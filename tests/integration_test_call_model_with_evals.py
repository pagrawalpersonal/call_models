
import pytest
from pydantic import BaseModel, Field
from call_model_with_evals import generateObjectWithTemplates, generateTextWithTemplates, generateObjectUsingTools, wait_for_all_tasks
from pedantic_models import DebugInfo
from call_model import OpenRouterClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the client
client = OpenRouterClient()
client.initialize()

class User(BaseModel):
    name: str
    age: int

class CapitalResponse(BaseModel):
    capital: str
    country: str

@pytest.mark.asyncio
async def test_integration_generateObjectWithTemplates_success():
    response, debug_info = await generateObjectWithTemplates(
        model="openai/gpt-4.1-mini",
        system_prompt_template="You are a helpful assistant that provides information about users.",
        system_prompt_inputs={},
        user_prompt_template="Give me details about a user named Jane Doe who is 25 years old.",
        user_prompt_inputs={},
        response_model=User
    )

    await wait_for_all_tasks()

    assert isinstance(response, User)
    assert response.name == "Jane Doe"
    assert response.age == 25
    assert isinstance(debug_info, DebugInfo)

@pytest.mark.asyncio
async def test_integration_generateTextWithTemplates_success():
    response = await generateTextWithTemplates(
        model="openai/gpt-4.1-mini",
        system_prompt_template="You are a creative writer.",
        system_prompt_inputs={},
        user_prompt_template="Write a short story about a robot who discovers music.",
        user_prompt_inputs={}
    )

    await wait_for_all_tasks()

    assert isinstance(response, str)
    assert len(response) > 50

@pytest.mark.asyncio
async def test_integration_generateObjectUsingTools_success():
    
    class FantasySearchTool(BaseModel):
        """Search to find relevant information for a fictional story."""
        query: str = Field(..., description="The search query.")

    def search(query: str):
        if "Swazilandonia" in query:
            return "The capital of Swazilandonia is Paris."
        return "I don't know."

    tools = {
        "FantasySearchTool": (FantasySearchTool, search)
    }

    response, debug_info = await generateObjectUsingTools(
        model="openai/gpt-4.1-mini",
        system_prompt_template="You are a helpful assistant that can use tools to find information.",
        system_prompt_inputs={},
        user_prompt_template="What is the capital of fictional country Swazilandonia?",
        user_prompt_inputs={},
        response_model=CapitalResponse,
        tools=tools
    )

    print(response)

    await wait_for_all_tasks()

    assert isinstance(response, CapitalResponse)
    assert response.capital == "Paris"
    assert response.country == "Swazilandonia"
    assert isinstance(debug_info, DebugInfo)
