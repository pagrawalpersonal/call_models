
import pytest
from pydantic import BaseModel
from call_model_with_evals import generateObjectWithTemplates, generateTextWithTemplates, generateObjectUsingTools
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
        model="qwen/qwen3-235b-a22b",
        system_prompt_template="You are a helpful assistant that provides information about users.",
        system_prompt_inputs={},
        user_prompt_template="Give me details about a user named Jane Doe who is 25 years old.",
        user_prompt_inputs={},
        response_model=User
    )

    assert isinstance(response, User)
    assert response.name == "Jane Doe"
    assert response.age == 25
    assert isinstance(debug_info, DebugInfo)

@pytest.mark.asyncio
async def test_integration_generateTextWithTemplates_success():
    response = await generateTextWithTemplates(
        model="qwen/qwen3-235b-a22b",
        system_prompt_template="You are a creative writer.",
        system_prompt_inputs={},
        user_prompt_template="Write a short story about a robot who discovers music.",
        user_prompt_inputs={}
    )

    assert isinstance(response, str)
    assert len(response) > 50

@pytest.mark.asyncio
async def test_integration_generateObjectUsingTools_success():
    
    def get_weather(city: str):
        return f"The weather in {city} is sunny."

    tools = {
        "get_weather": (BaseModel, get_weather)
    }

    response, debug_info = await generateObjectUsingTools(
        model="qwen/qwen3-235b-a22b",
        system_prompt_template="You are a helpful assistant that can use tools to find information.",
        system_prompt_inputs={},
        user_prompt_template="What is the capital of Germany?",
        user_prompt_inputs={},
        response_model=CapitalResponse,
        tools=tools
    )

    assert isinstance(response, CapitalResponse)
    assert response.capital == "Berlin"
    assert response.country == "Germany"
    assert isinstance(debug_info, DebugInfo)
