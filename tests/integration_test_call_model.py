
import pytest
from pydantic import BaseModel
from call_model import generateObject, generateText, RetryPolicy, OpenRouterClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the client
client = OpenRouterClient()
client.initialize()

class User(BaseModel):
    name: str
    age: int

@pytest.mark.asyncio
async def test_integration_generateObject_success():
    response = await generateObject(
        model="qwen/qwen3-235b-a22b",
        system_prompt="You are a helpful assistant that provides information about users.",
        user_prompt="Give me details about a user named John Doe who is 30 years old.",
        response_model=User,
        retry_policy=RetryPolicy.NO_RETRY
    )

    assert isinstance(response, User)
    assert response.name == "John Doe"
    assert response.age == 30

@pytest.mark.asyncio
async def test_integration_generateText_success():
    response = await generateText(
        model="qwen/qwen3-235b-a22b",
        system_prompt="You are a helpful assistant.",
        user_prompt="What is the capital of France?",
        retry_policy=RetryPolicy.NO_RETRY
    )

    assert isinstance(response, str)
    assert "Paris" in response
