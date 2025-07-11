
import pytest
from unittest.mock import patch, AsyncMock
from pydantic import BaseModel
from call_model_with_evals import generateObjectWithTemplates, generateTextWithTemplates, generateObjectUsingTools
from pedantic_models import DebugInfo

class MockResponse(BaseModel):
    text: str

class MockDebugWrapper(BaseModel):
    text: str
    debug_info: DebugInfo

@pytest.mark.asyncio
async def test_generateObjectWithTemplates_success():
    with patch('call_model_with_evals.generateObject', new_callable=AsyncMock) as mock_generate_object:
        mock_debug_info = DebugInfo(prompt_difficulties="none", prompt_improvements="none")
        mock_response = MockResponse(text="test")
        
        # The mock should return an object with the original fields and 'debug_info' attribute
        mock_return = AsyncMock()
        mock_return.text = mock_response.text
        mock_return.debug_info = mock_debug_info
        
        # Mock the usage attribute
        mock_return.usage = {"prompt_tokens": 1, "completion_tokens": 1}
        
        mock_generate_object.return_value = mock_return

        response, debug_info = await generateObjectWithTemplates(
            model="anthropic/claude-3-haiku-20240307",
            system_prompt_template="system",
            system_prompt_inputs={},
            user_prompt_template="user",
            user_prompt_inputs={},
            response_model=MockResponse
        )

        assert isinstance(response, MockResponse)
        assert response.text == "test"
        assert isinstance(debug_info, DebugInfo)
        mock_generate_object.assert_called_once()

@pytest.mark.asyncio
async def test_generateTextWithTemplates_success():
    with patch('call_model_with_evals.generateText', new_callable=AsyncMock) as mock_generate_text:
        mock_generate_text.return_value = "test"

        response = await generateTextWithTemplates(
            model="anthropic/claude-3-haiku-20240307",
            system_prompt_template="system",
            system_prompt_inputs={},
            user_prompt_template="user",
            user_prompt_inputs={}
        )

        assert isinstance(response, str)
        assert response == "test"
        mock_generate_text.assert_called_once()

@pytest.mark.asyncio
async def test_generateObjectUsingTools_success():
    with patch('call_model_with_evals.generateObject', new_callable=AsyncMock) as mock_generate_object:
        # The mock should return an object with a 'choices' attribute
        mock_choice = AsyncMock()
        mock_choice.message.content = '{"type": "answer", "response": {"parameters": {"text": "test"}}}'
        mock_completion = AsyncMock()
        mock_completion.choices = [mock_choice]
        mock_generate_object.return_value = mock_completion

        response, debug_info = await generateObjectUsingTools(
            model="anthropic/claude-3-haiku-20240307",
            system_prompt_template="system",
            system_prompt_inputs={},
            user_prompt_template="user",
            user_prompt_inputs={},
            response_model=MockResponse,
            tools={}
        )

        assert isinstance(response, MockResponse)
        assert response.text == "test"
        mock_generate_object.assert_called_once()
