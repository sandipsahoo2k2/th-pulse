import os
import json
import logging
from typing import Annotated, List, Union, Literal, Final, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ChatTicket")

load_dotenv()

# --- Tools ---

@tool
def validate_customer_tool(customer_id: str):
    """Validates a customer ID. Use this before creating any ticket."""
    logger.info(f"Tool Action: Validating Customer ID: {customer_id}")
    valid_ids = ["CUST123", "CUST456", "CUST789"]
    if str(customer_id).upper() in valid_ids:
        logger.info(f"Validation Result: SUCCESS for {customer_id}")
        return f"Customer {customer_id} is VALID."
    else:
        logger.info(f"Validation Result: FAILED for {customer_id}")
        return f"Customer {customer_id} is INVALID. Please provide a valid customer ID (e.g., CUST123)."

@tool
def create_technical_ticket_tool(customer_id: str, issue_description: str, priority: Literal["low", "medium", "high"]):
    """Creates a technical support ticket. ONLY use for bugs, errors, and system issues."""
    logger.info(f"Tool Action: Creating Technical Ticket for {customer_id}")
    ticket_id = f"TECH-{os.urandom(2).hex().upper()}"
    logger.info(f"Ticket Created: {ticket_id}")
    return json.dumps({
        "status": "Success",
        "ticket_id": ticket_id,
        "type": "Technical",
        "customer": customer_id,
        "description": issue_description,
        "priority": priority
    })

@tool
def create_billing_ticket_tool(customer_id: str, amount_disputed: float, invoice_number: str):
    """Creates a billing inquiry ticket. ONLY use for payments, invoices, and refunds."""
    logger.info(f"Tool Action: Creating Billing Ticket for {customer_id}")
    ticket_id = f"BILL-{os.urandom(2).hex().upper()}"
    logger.info(f"Ticket Created: {ticket_id}")
    return json.dumps({
        "status": "Success",
        "ticket_id": ticket_id,
        "type": "Billing",
        "customer": customer_id,
        "invoice": invoice_number,
        "amount": amount_disputed
    })

# --- Agents ---

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def make_agent(llm, tools, system_prompt: str, name: str):
    """Helper to create an agent node."""
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
    
    def agent_node(state: MessagesState):
        logger.info(f"--- Node: {name} Agent Turn ---")
        
        # 1. Filter out other named agents' messages
        filtered_messages = []
        other_agent_tool_call_ids = set()
        
        for msg in state["messages"]:
            msg_name = getattr(msg, "name", None)
            if isinstance(msg, AIMessage) and msg_name and msg_name != name:
                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        other_agent_tool_call_ids.add(tc["id"])
                continue
            if isinstance(msg, ToolMessage) and msg.tool_call_id in other_agent_tool_call_ids:
                continue
            filtered_messages.append(msg)
            
        # 2. Re-sequence to satisfy OpenAI's strict tool_call rules
        safe = []
        tool_msgs = {}
        for m in filtered_messages:
            if isinstance(m, ToolMessage):
                tool_msgs[m.tool_call_id] = m
                
        for m in filtered_messages:
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                # Only keep resolved tool calls, and pair them immediately
                resolved_tcs = [tc for tc in m.tool_calls if tc["id"] in tool_msgs]
                clean_msg = AIMessage(
                    content=str(m.content),
                    tool_calls=resolved_tcs,
                    name=getattr(m, "name", None),
                    id=m.id
                )
                safe.append(clean_msg)
                for tc in resolved_tcs:
                    safe.append(tool_msgs[tc["id"]])
            elif isinstance(m, ToolMessage):
                pass
            else:
                safe.append(m)
                
        response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + safe)
        
        final_response = AIMessage(
            content=response.content,
            tool_calls=getattr(response, "tool_calls", []),
            name=name,
            id=response.id,
            response_metadata=getattr(response, "response_metadata", {})
        )
        return {"messages": [final_response]}
    
    return agent_node

TECHNICAL_AGENT = "technical_agent"
BILLING_AGENT = "billing_agent"
SUPERVISOR = "supervisor"

tech_prompt = (
    "You are a Technical Support Agent. Your goal is to create technical tickets for BUGS, ERRORS, and SYSTEM ISSUES.\n"
    "1. You MUST validate the customer using 'validate_customer_tool' first.\n"
    "2. ONLY create tickets for technical issues. COMPLETELY IGNORE ANY BILLING OR PAYMENT ISSUES mentioned by the user, DO NOT mention them or other agents.\n"
    "3. Once the technical ticket is created, strictly confirm the TECH Ticket ID and stop talking.\n"
    "4. Do NOT tell the user you are passing them to another department."
)
billing_prompt = (
    "You are a Billing Support Agent. Your goal is to create billing tickets for PAYMENTS, INVOICES, and REFUNDS.\n"
    "1. You MUST validate the customer using 'validate_customer_tool' first.\n"
    "2. ONLY create tickets for billing issues. COMPLETELY IGNORE ANY TECHNICAL OR SYSTEM ISSUES mentioned by the user, DO NOT mention them or other agents.\n"
    "3. Once the billing ticket is created, strictly confirm the BILL Ticket ID and stop talking."
)

tech_node = make_agent(llm, [validate_customer_tool, create_technical_ticket_tool], tech_prompt, TECHNICAL_AGENT)
billing_node = make_agent(llm, [validate_customer_tool, create_billing_ticket_tool], billing_prompt, BILLING_AGENT)

# --- Supervisor Logic ---

def supervisor_node(state: MessagesState):
    """Supervisor decides who goes next OR if we are finished with a final message."""
    logger.info("--- Node: Supervisor Decision Turn ---")
    
    # Only check for tickets created AFTER the latest user message to allow new issues
    messages = state["messages"]
    new_messages = []
    for msg in reversed(messages):
        if msg.type == "human":
            break
        new_messages.append(msg)
    
    current_turn_history = " ".join([str(msg.content) for msg in new_messages if isinstance(msg, (ToolMessage, AIMessage))])
    
    # These flags now represent if a ticket was created during THIS specific user request
    has_tech_this_turn = "TECH-" in current_turn_history
    has_billing_this_turn = "BILL-" in current_turn_history

    system_prompt = (
        "You are a helpdesk supervisor coordinating Technical and Billing agents.\n"
        "Rules:\n"
        "1. PRIORITIZE the LATEST human message. If it contains a NEW request or problem, you MUST act on it.\n"
        "2. Ignore previous 'FINISH' or 'Resolved' messages if the user has described a new issue.\n"
        "3. Route to 'technical_agent' if there is a tech issue described in the latest message that hasn't been ticketed yet in this turn.\n"
        "4. Route to 'billing_agent' if there is a billing issue described in the latest message that hasn't been ticketed yet in this turn.\n"
        f"Current Turn Status: Tech Resolved: {has_tech_this_turn}, Billing Resolved: {has_billing_this_turn}.\n"
        "5. Return 'FINISH' ONLY if the latest user request is fully handled or if they are just saying thanks.\n"
        "6. ONLY provide a polite 'message' if choosing 'FINISH'. If routing, leave 'message' empty."
    )
    
    router_llm = llm.with_structured_output({
        "title": "router",
        "type": "object",
        "properties": {
            "next": {
                "type": "array", 
                "items": {
                    "type": "string", 
                    "enum": ["technical_agent", "billing_agent", "FINISH"]
                },
                "description": "The next agent(s) to route to. Can be multiple for parallel execution. Return ['FINISH'] to end."
            },
            "message": {
                "type": "string",
                "description": "Closing message. Only use if next is ['FINISH'] and appropriate."
            }
        },
        "required": ["next"]
    })
    
    # Strip tool_calls from AIMessages and modify ToolMessages for the router LLM
    safe_messages = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            safe_messages.append(AIMessage(content=msg.content, name=getattr(msg, "name", None)))
        elif isinstance(msg, ToolMessage):
            safe_messages.append(AIMessage(content=f"[Tool Result: {msg.content}]", name=msg.name))
        else:
            safe_messages.append(msg)
    
    try:
        response = router_llm.invoke([SystemMessage(content=system_prompt)] + safe_messages)
        decisions_raw = response.get("next", ["FINISH"])
        if not isinstance(decisions_raw, list):
            decisions_raw = [decisions_raw]
            
        decisions = []
        for d in decisions_raw:
            if d == "technical_agent" and has_tech_this_turn:
                logger.warning("Agent already created Tech ticket. Overriding to prevent loop.")
            elif d == "billing_agent" and has_billing_this_turn:
                logger.warning("Agent already created Billing ticket. Overriding to prevent loop.")
            else:
                if d not in decisions:
                    decisions.append(d)
                    
        if not decisions:
            decisions = ["FINISH"]
        elif len(decisions) > 1 and "FINISH" in decisions:
            decisions.remove("FINISH")
            
        msg_content = response.get("message") if decisions == ["FINISH"] else None
        
        logger.info(f"Supervisor Decision: {decisions}")
        
        if msg_content:
            logger.info(f"Supervisor Final Message: {msg_content}")
            return {"next_step": decisions, "messages": [AIMessage(content=msg_content)]}
            
        return {"next_step": decisions}
    except Exception as e:
        logger.error(f"Supervisor Error: {e}")
        return {"next_step": ["FINISH"]}

# --- Graph Construction ---

class GraphState(MessagesState):
    next_step: Optional[Union[str, List[str]]] = None

workflow = StateGraph(GraphState)

workflow.add_node(SUPERVISOR, supervisor_node)
workflow.add_node(TECHNICAL_AGENT, tech_node)
workflow.add_node(BILLING_AGENT, billing_node)
workflow.add_node("tech_tools", ToolNode([validate_customer_tool, create_technical_ticket_tool]))
workflow.add_node("billing_tools", ToolNode([validate_customer_tool, create_billing_ticket_tool]))

workflow.set_entry_point(SUPERVISOR)

workflow.add_conditional_edges(
    SUPERVISOR,
    lambda x: x.get("next_step", ["FINISH"]),
    {
        "technical_agent": TECHNICAL_AGENT,
        "billing_agent": BILLING_AGENT,
        "FINISH": END
    }
)

def make_agent_router(agent_name: str):
    def agent_router(state: GraphState):
        """
        If agent calls a tool, go to tools. 
        If agent provides a NEW Ticket ID in their current message, go to SUPERVISOR for handoff.
        Otherwise, STOP and wait for user input.
        """
        last_msg = state["messages"][-1]
        for msg in reversed(state["messages"]):
            if getattr(msg, "name", None) == agent_name:
                last_msg = msg
                break
                
        if getattr(last_msg, "tool_calls", None):
            return "tools"
        
        # We only want to loop back to supervisor if this SPECIFIC message 
        # just confirmed a ticket was created.
        content = str(last_msg.content)
        # Check if the agent just said they created a ticket
        if "created" in content.lower() and ("TECH-" in content or "BILL-" in content):
            logger.info(f"{agent_name}: New ticket confirmed. Returning to Supervisor for potential handoff.")
            return "supervisor"

        # Otherwise, they are likely asking a question or the task is already finished.
        logger.info(f"{agent_name}: move complete. Ending turn.")
        return "finish"
    return agent_router

workflow.add_conditional_edges(TECHNICAL_AGENT, make_agent_router(TECHNICAL_AGENT), {"tools": "tech_tools", "supervisor": SUPERVISOR, "finish": END})
workflow.add_conditional_edges(BILLING_AGENT, make_agent_router(BILLING_AGENT), {"tools": "billing_tools", "supervisor": SUPERVISOR, "finish": END})

workflow.add_edge("tech_tools", TECHNICAL_AGENT)
workflow.add_edge("billing_tools", BILLING_AGENT)

graph = workflow.compile(checkpointer=MemorySaver())

# --- FastAPI App ---

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html") as f:
        return f.read()

class ChatRequest(BaseModel):
    message: str
    thread_id: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    logger.info(f"--- Chat Request: {req.message} ---")
    config = {"configurable": {"thread_id": req.thread_id}, "recursion_limit": 20}
    
    try:
        # Run graph
        result = graph.invoke({"messages": [HumanMessage(content=req.message)]}, config=config)
        
        # Aggregate ALL AI messages generated after the human message we just sent
        all_messages = result.get("messages", [])
        last_human_index = -1
        # Loop backwards to find the index of the human message we just added
        for i in range(len(all_messages) - 1, -1, -1):
            if all_messages[i].type == "human":
                last_human_index = i
                break
        
        new_ai_messages = []
        if last_human_index != -1:
            for msg in all_messages[last_human_index + 1:]:
                # Only take final AI responses (those without tool calls)
                if msg.type == "ai" and msg.content and not getattr(msg, "tool_calls", None):
                    new_ai_messages.append(msg.content)
        
        if new_ai_messages:
            # Combine all responses into one message for the UI
            combined_reply = "\n\n".join(new_ai_messages)
            return {"reply": combined_reply}
                
        return {"reply": "I've processed your request. Is there anything else?"}
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
