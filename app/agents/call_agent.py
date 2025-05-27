"""
LangGraph-based call center agent
"""
import asyncio
import json
from typing import Dict, List, Optional, Any, TypedDict
from datetime import datetime
from enum import Enum

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from app.core.config import settings
from app.agents.tools import get_call_center_tools
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService
from app.models.conversation import ConversationState, CallStatus
from app.utils.text_processing import extract_intent, extract_phone_number, extract_email
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
    context: Dict[str, Any]
    intent_history: List[str]
    
class CallCenterAgent:
    """Main call center agent using LangGraph"""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.embedding_service = EmbeddingService()
        self.tools = get_call_center_tools()
        self.tool_executor = ToolExecutor(self.tools)
        self.graph = self._create_graph()
        self.conversation_memory = []
        
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
        logger.debug("Listening for customer input...")
        state["current_speaker"] = "customer"
        
        # Check if we have new messages to process
        if not state.get("messages"):
            # Initial state - no messages yet
            state["next_action"] = "greet"
        else:
            # Check if last message is from user
            last_message = state["messages"][-1]
            if last_message.get("role") == "user":
                state["next_action"] = "process_input"
            else:
                state["next_action"] = "wait"
                
        return state
        
    async def understand_node(self, state: AgentState) -> AgentState:
        """Understand intent and extract information"""
        messages = state["messages"]
        if not messages:
            return state
            
        last_message = messages[-1]
        if last_message.get("role") != "user":
            return state
            
        user_text = last_message.get("content", "")
        
        # Extract intent and entities
        intent_info = extract_intent(user_text)
        
        # Extract contact information
        phones = extract_phone_number(user_text)
        emails = extract_email(user_text)
        
        # Update customer info if found
        if phones and not state["customer_info"].get("phone"):
            state["customer_info"]["phone"] = phones[0]
        if emails and not state["customer_info"].get("email"):
            state["customer_info"]["email"] = emails[0]
            
        # Analyze emotion/sentiment
        emotion = self._analyze_emotion(user_text)
        state["emotion"] = emotion
        
        # Update intent history
        if "intent_history" not in state:
            state["intent_history"] = []
        primary_intent = intent_info.get("primary_intent", "unknown")
        state["intent_history"].append(primary_intent)
        
        # Determine next action
        state["next_action"] = self._determine_next_action(intent_info, state)
        
        # Store understanding context
        state["context"]["last_intent"] = intent_info
        state["context"]["extracted_entities"] = {
            "phones": phones,
            "emails": emails
        }
        
        logger.debug(f"Understood intent: {primary_intent}, emotion: {emotion}")
        return state
        
    async def retrieve_context_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant context from knowledge base"""
        logger.debug("Retrieving context...")
        
        messages = state["messages"]
        if not messages:
            return state
            
        last_message = messages[-1]
        if last_message.get("role") != "user":
            return state
            
        query = last_message.get("content", "")
        
        # Search knowledge base for relevant information
        try:
            search_results = await self.embedding_service.search(
                query=query,
                n_results=3
            )
            
            if search_results:
                relevant_context = []
                for result in search_results:
                    if result.get("similarity", 0) > 0.7:  # Only high-confidence results
                        relevant_context.append({
                            "content": result["content"][:300],  # Limit length
                            "source": result["metadata"].get("filename", "KB"),
                            "relevance": result.get("similarity", 0)
                        })
                        
                state["context"]["knowledge_base_results"] = relevant_context
                logger.debug(f"Retrieved {len(relevant_context)} relevant KB entries")
            else:
                state["context"]["knowledge_base_results"] = []
                
        except Exception as e:
            logger.error(f"Context retrieval error: {e}")
            state["context"]["knowledge_base_results"] = []
            
        return state
        
    async def generate_response_node(self, state: AgentState) -> AgentState:
        """Generate appropriate response"""
        # Build conversation history
        conversation = self._build_conversation_history(state)
        
        # Create system prompt
        system_prompt = self._create_system_prompt(state)
        
        # Prepare messages for LLM
        llm_messages = [{"role": "system", "content": system_prompt.content}]
        llm_messages.extend([
            {"role": msg.content if hasattr(msg, 'content') else msg["role"], 
             "content": msg.content if hasattr(msg, 'content') else msg["content"]}
            for msg in conversation
        ])
        
        try:
            # Generate response
            response = await self.llm_service.chat(
                messages=llm_messages,
                tools=self._get_available_tools(state) if state["next_action"] != "end" else None,
                temperature=0.7,
                max_tokens=300
            )
            
            # Handle response
            response_content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])
            
            # Create assistant message
            assistant_message = {
                "role": "assistant",
                "content": response_content,
                "timestamp": datetime.now().isoformat()
            }
            
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
                
            state["messages"].append(assistant_message)
            
            logger.debug(f"Generated response: {response_content[:100]}...")
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            # Fallback response
            fallback_response = self._get_fallback_response(state)
            state["messages"].append({
                "role": "assistant",
                "content": fallback_response,
                "timestamp": datetime.now().isoformat()
            })
            
        return state
        
    async def execute_tools_node(self, state: AgentState) -> AgentState:
        """Execute any required tools"""
        last_message = state["messages"][-1]
        
        if "tool_calls" not in last_message or not last_message["tool_calls"]:
            return state
            
        tool_results = []
        
        for tool_call in last_message["tool_calls"]:
            try:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("arguments", {})
                
                logger.debug(f"Executing tool: {tool_name} with args: {tool_args}")
                
                # Execute tool
                tool_result = await self._execute_tool(tool_name, tool_args)
                
                tool_results.append({
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": tool_result,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Update state based on tool result
                self._update_state_from_tool_result(state, tool_name, tool_result)
                
            except Exception as e:
                logger.error(f"Tool execution error for {tool_name}: {e}")
                tool_results.append({
                    "tool": tool_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                
        state["tools_output"].extend(tool_results)
        return state
        
    async def speak_node(self, state: AgentState) -> AgentState:
        """Convert response to speech and send"""
        last_message = state["messages"][-1]
        
        if last_message.get("role") != "assistant":
            return state
            
        content = last_message.get("content", "")
        
        if content.strip():
            # Queue for TTS
            await state["audio_queue"].put({
                "type": "speech",
                "content": content,
                "emotion": state.get("emotion", "neutral"),
                "voice_style": self._select_voice_style(state)
            })
            
            logger.debug(f"Queued speech: {content[:50]}...")
        
        state["current_speaker"] = "agent"
        return state
        
    async def end_call_node(self, state: AgentState) -> AgentState:
        """Handle call ending"""
        logger.info("Ending call...")
        state["call_status"] = "ended"
        
        # Generate call summary
        try:
            summary = await self._generate_call_summary(state)
            state["call_summary"] = summary
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            state["call_summary"] = {"error": str(e)}
        
        # Final message
        farewell_message = self._get_farewell_message(state)
        if farewell_message:
            await state["audio_queue"].put({
                "type": "speech",
                "content": farewell_message,
                "emotion": "professional"
            })
            
        return state
        
    def should_use_tools(self, state: AgentState) -> str:
        """Determine next action based on response"""
        if state["call_status"] == "ending":
            return "end"
            
        if not state["messages"]:
            return "speak"
            
        last_message = state["messages"][-1]
        
        if "tool_calls" in last_message and last_message["tool_calls"]:
            return "tools"
        elif state["next_action"] == "end":
            return "end"
        else:
            return "speak"
            
    def _create_system_prompt(self, state: AgentState) -> SystemMessage:
        """Create comprehensive system prompt"""
        # Get current context
        customer_info = state.get("customer_info", {})
        emotion = state.get("emotion", "neutral")
        call_status = state.get("call_status", "active")
        intent_history = state.get("intent_history", [])
        kb_results = state.get("context", {}).get("knowledge_base_results", [])
        
        # Build context section
        context_info = []
        if customer_info:
            context_info.append(f"Customer Info: {json.dumps(customer_info, indent=2)}")
        if emotion != "neutral":
            context_info.append(f"Customer Emotion: {emotion}")
        if intent_history:
            context_info.append(f"Intent History: {', '.join(intent_history[-3:])}")
        if kb_results:
            context_info.append("Relevant Information:")
            for result in kb_results[:2]:
                context_info.append(f"- {result['content'][:200]}...")
                
        context_section = "\n".join(context_info) if context_info else "No specific context available."
        
        prompt = f"""You are a professional call center agent for {settings.APP_NAME}.

CURRENT CONTEXT:
{context_section}

YOUR ROLE & RESPONSIBILITIES:
1. Help customers with insurance inquiries professionally and empathetically
2. Collect necessary information for claims processing
3. Schedule appointments and provide policy information
4. Use available tools to access customer data and process requests
5. Escalate to human agents when appropriate

COMMUNICATION GUIDELINES:
- Be concise and clear (keep responses under 100 words)
- Show empathy, especially when customer emotion is {emotion}
- Ask one question at a time to avoid overwhelming customers
- Confirm important information before proceeding
- Use professional yet friendly tone

TOOL USAGE:
- Use lookup_customer when you need customer information
- Use create_claim for new insurance claims
- Use check_claim_status for existing claims
- Use schedule_appointment for booking meetings
- Use search_knowledge_base for policy/procedure questions
- Use transfer_to_human when you cannot help or customer requests it

CURRENT SITUATION:
- Call Status: {call_status}
- Customer appears to be: {emotion}
- Recent intents: {', '.join(intent_history[-2:]) if intent_history else 'None'}

Respond helpfully and professionally. If you need to use tools, explain what you're doing."""

        return SystemMessage(content=prompt)
        
    def _build_conversation_history(self, state: AgentState) -> List[HumanMessage | AIMessage]:
        """Build conversation history for LLM"""
        history = []
        messages = state.get("messages", [])
        
        # Limit to last 10 messages to manage token usage
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        
        for msg in recent_messages:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history.append(AIMessage(content=msg["content"]))
                
        return history
        
    def _analyze_emotion(self, text: str) -> str:
        """Analyze emotional state from text"""
        text_lower = text.lower()
        
        # Frustrated/Angry indicators
        if any(word in text_lower for word in ["angry", "frustrated", "unacceptable", "ridiculous", "terrible", "awful", "hate"]):
            return "frustrated"
        
        # Confused indicators  
        if any(word in text_lower for word in ["confused", "don't understand", "what do you mean", "unclear", "explain"]):
            return "confused"
        
        # Urgent indicators
        if any(word in text_lower for word in ["urgent", "asap", "immediately", "right now", "emergency"]):
            return "urgent"
        
        # Satisfied/Happy indicators
        if any(word in text_lower for word in ["thank you", "great", "perfect", "excellent", "wonderful"]):
            return "satisfied"
        
        # Worried/Concerned indicators
        if any(word in text_lower for word in ["worried", "concerned", "scared", "nervous", "anxious"]):
            return "concerned"
        
        return "neutral"
        
    def _determine_next_action(self, intent_info: Dict, state: AgentState) -> str:
        """Determine the next action based on intent"""
        primary_intent = intent_info.get("primary_intent", "unknown")
        
        action_mapping = {
            "create_claim": "create_claim",
            "check_status": "check_status", 
            "schedule": "schedule_appointment",
            "complaint": "handle_complaint",
            "information": "provide_information",
            "transfer": "transfer_human",
            "greeting": "greet",
            "goodbye": "end"
        }
        
        return action_mapping.get(primary_intent, "clarify")
        
    def _get_available_tools(self, state: AgentState) -> List:
        """Get tools available based on current state"""
        # All tools available by default
        # Could be filtered based on customer status, agent permissions, etc.
        return self.tools
        
    async def _execute_tool(self, tool_name: str, tool_args: Dict) -> Any:
        """Execute a specific tool"""
        # Find the tool
        tool = None
        for t in self.tools:
            if t.name == tool_name:
                tool = t
                break
                
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
            
        # Execute tool
        if asyncio.iscoroutinefunction(tool.func):
            result = await tool.func(**tool_args)
        else:
            result = tool.func(**tool_args)
            
        return result
        
    def _update_state_from_tool_result(self, state: AgentState, tool_name: str, result: Any):
        """Update state based on tool execution result"""
        if tool_name == "lookup_customer" and result.get("found"):
            customer_data = result.get("customer", {})
            state["customer_info"].update(customer_data)
            
        elif tool_name == "create_claim" and result.get("success"):
            claim_data = result.get("claim", {})
            state["claim_data"] = claim_data
            
        elif tool_name == "transfer_to_human":
            state["call_status"] = "transferring"
            state["next_action"] = "end"
            
    def _select_voice_style(self, state: AgentState) -> str:
        """Select appropriate voice style based on context"""
        emotion = state.get("emotion", "neutral")
        
        voice_mapping = {
            "frustrated": "empathetic",
            "confused": "friendly", 
            "urgent": "professional",
            "satisfied": "friendly",
            "concerned": "empathetic",
            "neutral": "professional"
        }
        
        return voice_mapping.get(emotion, "professional")
        
    def _get_fallback_response(self, state: AgentState) -> str:
        """Get fallback response when LLM fails"""
        emotion = state.get("emotion", "neutral")
        
        if emotion == "frustrated":
            return "I understand your frustration. Let me transfer you to a human agent who can better assist you."
        elif emotion == "confused":
            return "I want to make sure I understand your needs correctly. Could you please tell me more about what you're looking for?"
        else:
            return "I apologize, but I'm having trouble processing your request right now. Let me connect you with a human agent."
            
    def _get_farewell_message(self, state: AgentState) -> str:
        """Get appropriate farewell message"""
        if state.get("call_status") == "transferring":
            return "I'm transferring you to a human agent now. Thank you for your patience."
        elif state.get("claim_data"):
            claim_id = state["claim_data"].get("id", "")
            return f"Your claim {claim_id} has been created. You'll receive confirmation via email. Thank you for calling."
        else:
            return "Thank you for calling. Have a great day!"
            
    async def _generate_call_summary(self, state: AgentState) -> Dict[str, Any]:
        """Generate comprehensive call summary"""
        messages = state.get("messages", [])
        customer_info = state.get("customer_info", {})
        claim_data = state.get("claim_data", {})
        tools_used = [tool["tool"] for tool in state.get("tools_output", [])]
        
        # Count message types
        user_messages = [m for m in messages if m.get("role") == "user"]
        agent_messages = [m for m in messages if m.get("role") == "assistant"]
        
        # Extract key information
        phone_numbers = []
        emails = []
        for msg in user_messages:
            phone_numbers.extend(extract_phone_number(msg.get("content", "")))
            emails.extend(extract_email(msg.get("content", "")))
            
        summary = {
            "call_duration_messages": len(messages),
            "customer_messages": len(user_messages),
            "agent_messages": len(agent_messages),
            "customer_info": customer_info,
            "claim_created": bool(claim_data),
            "claim_data": claim_data,
            "tools_used": tools_used,
            "emotions_detected": list(set(state.get("intent_history", []))),
            "contact_info_extracted": {
                "phones": list(set(phone_numbers)),
                "emails": list(set(emails))
            },
            "resolution_status": self._determine_resolution_status(state),
            "follow_up_needed": self._needs_follow_up(state),
            "generated_at": datetime.now().isoformat()
        }
        
        return summary
        
    def _determine_resolution_status(self, state: AgentState) -> str:
        """Determine if the call was resolved"""
        if state.get("call_status") == "transferring":
            return "transferred"
        elif state.get("claim_data"):
            return "claim_created"
        elif "schedule_appointment" in [t["tool"] for t in state.get("tools_output", [])]:
            return "appointment_scheduled"
        elif state.get("emotion") in ["satisfied", "neutral"]:
            return "resolved"
        else:
            return "unresolved"
            
    def _needs_follow_up(self, state: AgentState) -> bool:
        """Determine if follow-up is needed"""
        return (
            state.get("claim_data") or
            state.get("call_status") == "transferring" or
            state.get("emotion") == "frustrated"
        )
        
    async def run(self, initial_state: AgentState) -> AgentState:
        """Run the agent workflow"""
        try:
            # Ensure required fields exist
            if "context" not in initial_state:
                initial_state["context"] = {}
            if "tools_output" not in initial_state:
                initial_state["tools_output"] = []
            if "intent_history" not in initial_state:
                initial_state["intent_history"] = []
                
            # Execute the workflow
            result = await self.graph.ainvoke(initial_state)
            return result
        except Exception as e:
            logger.error(f"Agent workflow error: {e}")
            # Return safe state
            initial_state["call_status"] = "error"
            initial_state["error"] = str(e)
            return initial_state