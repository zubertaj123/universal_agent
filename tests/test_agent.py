"""
Tests for the call center agent
"""
import pytest
import asyncio
from app.agents.call_agent import CallCenterAgent, AgentState
from app.services.llm_service import LLMService
from app.services.speech_service import SpeechService

@pytest.fixture
def mock_llm_service(mocker):
    """Mock LLM service"""
    service = mocker.Mock(spec=LLMService)
    service.chat.return_value = asyncio.Future()
    service.chat.return_value.set_result({
        "content": "I understand your concern. Let me help you with that.",
        "role": "assistant"
    })
    return service

@pytest.fixture
def agent(mock_llm_service):
    """Create agent instance"""
    return CallCenterAgent(mock_llm_service)

@pytest.fixture
def initial_state():
    """Create initial agent state"""
    return {
        "messages": [],
        "current_speaker": "user",
        "call_status": "active",
        "customer_info": {},
        "claim_data": {},
        "emotion": "neutral",
        "tools_output": [],
        "next_action": "listen",
        "audio_queue": asyncio.Queue()
    }

@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """Test agent initialization"""
    assert agent is not None
    assert agent.graph is not None
    assert len(agent.tools) > 0

@pytest.mark.asyncio
async def test_listen_node(agent, initial_state):
    """Test listen node"""
    state = await agent.listen_node(initial_state)
    assert state["current_speaker"] == "customer"

@pytest.mark.asyncio
async def test_understand_node(agent, initial_state):
    """Test understand node"""
    initial_state["messages"] = [{
        "role": "user",
        "content": "I need to file a claim for my car accident"
    }]
    
    state = await agent.understand_node(initial_state)
    assert "emotion" in state
    assert "next_action" in state

@pytest.mark.asyncio
async def test_generate_response_node(agent, initial_state, mock_llm_service):
    """Test response generation"""
    initial_state["messages"] = [{
        "role": "user",
        "content": "Hello, I need help"
    }]
    
    state = await agent.generate_response_node(initial_state)
    assert len(state["messages"]) == 2
    assert state["messages"][-1]["role"] == "assistant"

@pytest.mark.asyncio
async def test_should_use_tools(agent, initial_state):
    """Test tool routing logic"""
    # Test with no tools
    initial_state["messages"] = [{
        "role": "assistant",
        "content": "How can I help you?"
    }]
    assert agent.should_use_tools(initial_state) == "speak"
    
    # Test with tools
    initial_state["messages"] = [{
        "role": "assistant",
        "content": "Let me look that up",
        "tool_calls": [{"name": "lookup_customer"}]
    }]
    assert agent.should_use_tools(initial_state) == "tools"
    
    # Test end condition
    initial_state["call_status"] = "ending"
    assert agent.should_use_tools(initial_state) == "end"

@pytest.mark.asyncio
async def test_full_workflow(agent, initial_state):
    """Test complete agent workflow"""
    initial_state["messages"] = [{
        "role": "user",
        "content": "I want to check my claim status"
    }]
    
    # Run the workflow
    final_state = await agent.run(initial_state)
    
    assert final_state is not None
    assert "messages" in final_state
    assert len(final_state["messages"]) > 1