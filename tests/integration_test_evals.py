
import pytest
import tempfile
import os
from call_model_with_evals import generateObjectWithTemplates, wait_for_all_tasks
from run_eval import process_archive_file
from judge_eval import process_eval_file
from pedantic_models import DebugInfo
from pydantic import BaseModel
from call_model import OpenRouterClient
from dotenv import load_dotenv
import json
from pedantic_models import PydanticEncoder
from call_model_constants import LLM_CALL_ARCHIVE

# Load environment variables from .env file
load_dotenv()

# Initialize the client
client = OpenRouterClient()
client.initialize()

class User(BaseModel):
    name: str
    age: int

@pytest.mark.asyncio
async def test_integration_eval_flow():
    # 1. Call generateObjectWithTemplates
    response, debug_info = await generateObjectWithTemplates(
        model="openai/gpt-4.1-mini",
        system_prompt_template="You are a helpful assistant that provides information about users.",
        system_prompt_inputs={},
        user_prompt_template="Give me details about a user named Jane Doe who is 35 years younger than his father who is 60 years old.",
        user_prompt_inputs={},
        response_model=User
    )

    await wait_for_all_tasks()

    assert isinstance(response, User)
    assert response.name == "Jane Doe"
    assert response.age == 25
    assert isinstance(debug_info, DebugInfo)
    assert hasattr(debug_info, 'archive_id')
    archive_id = debug_info.archive_id

    # 2. Create a temporary file for the eval run
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".jsonl") as tmp_eval_file:
        eval_run_file = tmp_eval_file.name

    try:
        # 3. Call process_archive_file
        results = await process_archive_file(LLM_CALL_ARCHIVE, id_filter=archive_id, new_model="deepseek/deepseek-r1-0528")

        # Append results to eval_run_calls.jsonl
        with open(eval_run_file, 'a') as f:
            for result in results:
                f.write(json.dumps(result, cls=PydanticEncoder) + '\n')

        # 4. Call process_eval_file
        results = await process_eval_file(
            eval_file=eval_run_file,
            archive_id_filter=archive_id,
            judge_model="google/gemini-2.0-flash-lite-001")
        
        # Print results
        for result in results:
            print(f"\nEvaluation for ID: {result['archive_id']}")
            print(f"Model new: {result['model_new']}")
            print(f"Model original: {result['model_original']}")
            print(f"Judge score: {result['judge_score']}")
            print(f"Judge reasoning: {result['judge_reasoning']}")
            if result['judge_improvements']:
                print(f"Suggested improvements: {result['judge_improvements']}")
            print("-" * 80)

    finally:
        # Clean up the temporary file
        if os.path.exists(eval_run_file):
            os.remove(eval_run_file)
            
        await wait_for_all_tasks()

