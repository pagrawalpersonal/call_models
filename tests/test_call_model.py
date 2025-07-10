
import pytest
from unittest.mock import patch, AsyncMock
from pydantic import BaseModel
from call_model import generateObject, generateText, RetryPolicy

class MockResponse(BaseModel):
    text: str

@pytest.mark.asyncio
async def test_generateObject_success():
    with patch('call_models.call_model.instructor.from_openai') as mock_from_openai:
        mock_client = AsyncMock()
        mock_from_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MockResponse(text="test")

        response = await generateObject(
            model="test_model",
            system_prompt="system",
            user_prompt="user",
            response_model=MockResponse,
            retry_policy=RetryPolicy.NO_RETRY
        )

        assert isinstance(response, MockResponse)
        assert response.text == "test"
        mock_client.chat.completions.create.assert_called_once()

@pytest.mark.asyncio
async def test_generateText_success():
    with patch('call_model.AsyncOpenAI') as mock_async_openai:
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client
        
        # Create a mock for the response object with the expected structure
        mock_choice = AsyncMock()
        mock_choice.message.content = "test"
        mock_completion = AsyncMock()
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion

        response = await generateText(
            model="any-model",
            system_prompt="system",
            user_prompt="user",
            retry_policy=RetryPolicy.NO_RETRY
        )

        assert isinstance(response, str)
        assert response == "test"
        mock_client.chat.completions.create.assert_called_once()
