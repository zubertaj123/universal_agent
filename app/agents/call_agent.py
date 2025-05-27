"""
LangGraph-based call center agent
"""
import asyncio
from typing import Dict, List, Optional, Any, TypedDict
from datetime import datetime
from enum import Enum

from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from app.core.config import settings
from app.agents.tools import get_call_center_tools
from app.services.llm_service import LLMService
from app.models.conversation import ConversationState, CallStatus
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class AgentState(TypedDict):
    """Agent state definition"""
    messages: List[Dict[str, Any]]
    current_speaker: str
    call_status: str
    customer_info: Dict[str, Any]
    claim_data: Dict[str, Any]
    emotion: str
    tools_output: List[Dict[str, Any]]
    next_action: str
    audio_queue: asyncio.Queue
    
class CallCenterAgent:
    """Main call center agent using LangGraph"""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.tools = get_call_center_tools()
        self.tool_executor = ToolExecutor(self.tools)
        self.graph = self._create_graph()
        
    def _create_graph(self) -> StateGraph:
        """Create the agent workflow graph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("listen", self.listen_node)
        workflow.add_node("understand", self.understand_node)
        workflow.add_node("retrieve_context", self.retrieve_context_node)
        workflow.add_node("generate_response", self.generate_response_node)
        workflow.add_node("execute_tools", self.execute_tools_node)
        workflow.add_node("speak", self.speak_node)
        workflow.add_node("end_call", self.end_call_node)
        
        # Add edges
        workflow.add_edge("listen", "understand")
        workflow.add_edge("understand", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        
        # Conditional edges
        workflow.add_conditional_edges(
            "generate_response",
            self.should_use_tools,
            {
                "tools": "execute_tools",
                "speak": "speak",
                "end": "end_call"
            }
        )
        
        workflow.add_edge("execute_tools", "generate_response")
        workflow.add_edge("speak", "listen")
        workflow.add_edge("end_call", END)
        
        # Set entry point
        workflow.set_entry_point("listen")
        
        return workflow.compile()
        
    async def listen_node(self, state: AgentState) -> AgentState:
        """Listen for customer input"""
        # This will be connected to STT service
        logger.debug("Listening for customer input...")
        state["current_speaker"] = "customer"
        return state
        
    async def understand_node(self, state: AgentState) -> AgentState:
        """Understand intent and extract information"""
        messages = state["messages"]
        if not messages:
            return state
            
        last_message = messages[-1]
        
        # Extract intent and entities
        understanding_prompt = self._create_understanding_prompt(last_message)
        response = await self.llm_service.complete(understanding_prompt)
        
        # Update state with understanding
        state["emotion"] = response.get("emotion", "neutral")
        state["next_action"] = response.get("intent", "clarify")
        
        return state
        
    async def retrieve_context_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant context from knowledge base"""
        # This would connect to your vector database
        logger.debug("Retrieving context...")
        return state
        
    async def generate_response_node(self, state: AgentState) -> AgentState:
        """Generate appropriate response"""
        # Build conversation history
        conversation = self._build_conversation_history(state)
        
        # Generate response
        system_prompt = self._create_system_prompt(state)
        response = await self.llm_service.chat(
            messages=[system_prompt] + conversation,
            tools=self.tools if state["next_action"] != "end" else None
        )
        
        # Update state
        state["messages"].append({
            "role": "assistant",
            "content": response.content,
            "timestamp": datetime.now().isoformat()
        })
        
        return state
        
    async def execute_tools_node(self, state: AgentState) -> AgentState:
        """Execute any required tools"""
        last_message = state["messages"][-1]
        
        if "tool_calls" in last_message:
            for tool_call in last_message["tool_calls"]:
                tool_result = await self.tool_executor.execute(
                    ToolInvocation(
                        tool=tool_call["name"],
                        tool_input=tool_call["arguments"]
                    )
                )
                
                state["tools_output"].append({
                    "tool": tool_call["name"],
                    "result": tool_result
                })
                
        return state
        
    async def speak_node(self, state: AgentState) -> AgentState:
        """Convert response to speech and send"""
        last_message = state["messages"][-1]
        
        # This will be connected to TTS service
        # For now, just put in audio queue
        await state["audio_queue"].put({
            "type": "speech",
            "content": last_message["content"],
            "emotion": state["emotion"]
        })
        
        state["current_speaker"] = "agent"
        return state
        
    async def end_call_node(self, state: AgentState) -> AgentState:
        """Handle call ending"""
        logger.info("Ending call...")
        state["call_status"] = "ended"
        
        # Generate summary
        summary = await self._generate_call_summary(state)
        state["call_summary"] = summary
        
        return state
        
    def should_use_tools(self, state: AgentState) -> str:
        """Determine next action based on response"""
        last_message = state["messages"][-1]
        
        if "tool_calls" in last_message and last_message["tool_calls"]:
            return "tools"
        elif state["next_action"] == "end" or state["call_status"] == "ending":
            return "end"
        else:
            return "speak"
            
    def _create_system_prompt(self, state: AgentState) -> SystemMessage:
        """Create system prompt for the agent"""
        prompt = f"""You are a professional call center agent for {settings.APP_NAME}.
        
Current call information:
- Customer emotion: {state.get('emotion', 'neutral')}
- Call status: {state.get('call_status', 'active')}
- Customer info: {state.get('customer_info', {})}

Your responsibilities:
1. Help customers with their inquiries professionally
2. Collect necessary information for claims
3. Show empathy and understanding
4. Use tools when needed to access information
5. Escalate to human agent when appropriate

Remember to:
- Be concise and clear
- Acknowledge customer emotions
- Ask clarifying questions when needed
- Confirm important information
"""
        return SystemMessage(content=prompt)
        
    def _build_conversation_history(self, state: AgentState) -> List[HumanMessage | AIMessage]:
        """Build conversation history for LLM"""
        history = []
        
        for msg in state["messages"]:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history.append(AIMessage(content=msg["content"]))
                
        return history
        
    def _create_understanding_prompt(self, message: Dict[str, str]) -> str:
        """Create prompt for understanding intent"""
        return f"""Analyze the following customer message and extract:
1. Primary intent (inquiry, complaint, claim, etc.)
2. Emotional state (neutral, happy, frustrated, angry, confused)
3. Key information mentioned
4. Suggested next action

Message: {message.get('content', '')}

Respond in JSON format.
"""
        
    async def _generate_call_summary(self, state: AgentState) -> Dict[str, Any]:
        """Generate call summary"""
        summary_prompt = f"""Generate a summary of this call:

Messages: {state['messages']}
Customer Info: {state.get('customer_info', {})}
Claim Data: {state.get('claim_data', {})}

Include:
1. Main issue/request
2. Resolution status
3. Follow-up actions needed
4. Customer satisfaction estimate
"""
        
        summary = await self.llm_service.complete(summary_prompt)
        return summary
        
    async def run(self, initial_state: AgentState) -> AgentState:
        """Run the agent workflow"""
        return await self.graph.ainvoke(initial_state)