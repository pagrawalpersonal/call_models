# LLM Model Calling Library

This project provides a robust library for interacting with Large Language Models (LLMs) through the OpenRouter API. It offers a structured and reliable way to make model calls, with built-in support for retries, template-based prompts, tool usage, and comprehensive logging and archiving.

## Main Features

- **Flexible Model Calls**: Generate either structured Pydantic objects (`generateObject`) or plain text (`generateText`).
- **Resilient API Interaction**: Built-in retry mechanisms with exponential backoff (`exponential`) or fixed intervals (`fixed`) to handle API errors gracefully.
- **Template-Based Prompts**: Easily construct dynamic prompts using Python's `Template` strings for both system and user messages.
- **Tool Integration**: Empower models to use external tools to gather information and perform actions, with a built-in multi-turn loop for complex tasks.
- **Debugging and Evaluation**: Automatically capture debug information from models about prompt difficulty and potential improvements.
- **Comprehensive Archiving**: All model calls are archived to a JSONL file (`evals/arhived_calls.jsonl`), including prompts, responses, and metadata, for easy analysis and debugging.
- **Detailed Logging**: In-depth logging of the entire process to `call_model.log` for troubleshooting.
- **Evaluation Framework**: A complete workflow for A/B testing models and prompts, including re-running archived calls and using a "judge" model to evaluate and compare responses.

## Setup

### 1. Install Packages

This project does not include a `requirements.txt` file. Based on the imports, you will need to install the following packages:

```bash
pip install instructor openai pydantic python-dotenv tenacity aiofiles
```

### 2. Configure Environment Variables

Create a `.env` file in the root of the project directory by copying the `env_sample` file:

```bash
cp env_sample .env
```

Open the `.env` file and add your API keys:

```
OPENROUTER_API_KEY=your_openrouter_api_key
GOOGLE_API_KEY=your_google_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## API Documentation

The core functionality is exposed through functions in `call_model.py` and `call_model_with_evals.py`.

### Basic Usage (`call_model.py`)

These functions provide the fundamental building blocks for making model calls.

#### `generateObject`

Generates a structured Pydantic object from a model's response.

**Usage:**
```python
import asyncio
from pydantic import BaseModel
from call_model import generateObject, OpenRouterClient

# Initialize the client
client = OpenRouterClient()
client.initialize()

class User(BaseModel):
    name: str
    age: int

async def main():
    response = await generateObject(
        model="openai/gpt-4.1-mini",
        system_prompt="You are a helpful assistant that provides information about users.",
        user_prompt="Give me details about a user named John Doe who is 30 years old.",
        response_model=User
    )
    print(response)
    # Expected output: name='John Doe' age=30

if __name__ == "__main__":
    asyncio.run(main())
```

#### `generateText`

Generates a plain text response from a model.

**Usage:**
```python
import asyncio
from call_model import generateText, OpenRouterClient

# Initialize the client
client = OpenRouterClient()
client.initialize()

async def main():
    response = await generateText(
        model="openai/gpt-4.1-mini",
        system_prompt="You are a helpful assistant.",
        user_prompt="What is the capital of France?",
    )
    print(response)
    # Expected output: A string containing "Paris"

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Usage (`call_model_with_evals.py`)

These functions build upon the basic ones to add support for prompt templates, tool usage, and debugging.

#### `generateObjectWithTemplates`

Generates a structured Pydantic object using prompt templates and captures debug information.

**Usage:**
```python
import asyncio
from pydantic import BaseModel
from call_model_with_evals import generateObjectWithTemplates, wait_for_all_tasks
from call_model import OpenRouterClient

# Initialize the client
client = OpenRouterClient()
client.initialize()

class User(BaseModel):
    name: str
    age: int

async def main():
    response, debug_info = await generateObjectWithTemplates(
        model="openai/gpt-4.1-mini",
        system_prompt_template="You are a helpful assistant that provides information about users.",
        system_prompt_inputs={},
        user_prompt_template="Give me details about a user named ${name} who is ${age} years old.",
        user_prompt_inputs={"name": "Jane Doe", "age": 25},
        response_model=User
    )
    await wait_for_all_tasks() # Ensure archive task completes
    print(response)
    print(debug_info)

if __name__ == "__main__":
    asyncio.run(main())
```

#### `generateTextWithTemplates`

Generates a plain text response using prompt templates.

**Usage:**
```python
import asyncio
from call_model_with_evals import generateTextWithTemplates, wait_for_all_tasks
from call_model import OpenRouterClient

# Initialize the client
client = OpenRouterClient()
client.initialize()

async def main():
    response = await generateTextWithTemplates(
        model="openai/gpt-4.1-mini",
        system_prompt_template="You are a creative writer.",
        system_prompt_inputs={},
        user_prompt_template="Write a short story about a ${animal} who discovers ${concept}.",
        user_prompt_inputs={"animal": "robot", "concept": "music"}
    )
    await wait_for_all_tasks() # Ensure archive task completes
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

#### `generateObjectUsingTools`

Generates a response by allowing the model to use a set of provided tools.

**Usage:**
```python
import asyncio
from pydantic import BaseModel, Field
from call_model_with_evals import generateObjectUsingTools, wait_for_all_tasks
from call_model import OpenRouterClient

# Initialize the client
client = OpenRouterClient()
client.initialize()

class CapitalResponse(BaseModel):
    capital: str
    country: str

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

async def main():
    response, debug_info = await generateObjectUsingTools(
        model="openai/gpt-4.1-mini",
        system_prompt_template="You are a helpful assistant that can use tools to find information.",
        system_prompt_inputs={},
        user_prompt_template="What is the capital of fictional country Swazilandonia?",
        user_prompt_inputs={},
        response_model=CapitalResponse,
        tools=tools
    )
    await wait_for_all_tasks() # Ensure archive task completes
    print(response)
    # Expected output: capital='Paris' country='Swazilandonia'

if __name__ == "__main__":
    asyncio.run(main())
```

## Evaluation Workflow

This project includes a powerful workflow for evaluating model responses, allowing you to A/B test different models or prompts and get quantitative feedback on their performance.

The flow consists of two main steps: running the evaluation and judging the results.

### 1. Running an Evaluation (`run_eval.py`)

This script re-runs a previously archived model call, potentially with a different model, to generate a new response for comparison.

**How it works:**
1.  It reads the `llm_call_archive.jsonl` file.
2.  You can filter for specific calls using the `--id` or `--tag` arguments.
3.  You can specify a new model to use with the `--model` argument.
4.  It executes the call again with the new parameters.
5.  The new response, along with the original context, is saved to `eval_calls.jsonl`.

**Usage:**

```bash
# Re-run the call with archive ID 'some_id' using a new model
python run_eval.py --id 'some_id' --model 'anthropic/claude-3-haiku-20240307'

# Re-run all calls with the tag 'test_prompts'
python run_eval.py --tag 'test_prompts'
```

### 2. Judging the Evaluation (`judge_eval.py`)

This script uses a "judge" model to compare the original response with the new one from the evaluation run.

**How it works:**
1.  It reads the `eval_calls.jsonl` file.
2.  For each entry, it calls a powerful "judge" model (e.g., GPT-4).
3.  The judge model provides a score from -1 (worse) to 1 (better), a detailed reason for its score, and suggestions for improvement.
4.  The results are printed to the console and saved in `judgement_calls.jsonl`.

**Usage:**

```bash
# Judge all evaluations in the default file
python judge_eval.py

# Judge a specific evaluation run by its eval_run_id
python judge_eval.py --eval_run_id 'some_eval_run_id'
```

This workflow provides a systematic way to iterate on and improve your prompts and model selections based on quantitative, AI-assisted feedback.

## Running Tests

The project uses `pytest` for testing. To run the tests, execute the following command in the project root:

```bash
pytest
```

The tests include both unit tests (mocking API calls) and integration tests (making real API calls). Ensure your `.env` file is configured correctly before running integration tests.

## Troubleshooting

### Log Files

All operations, including API requests, retries, errors, and evaluation runs, are logged to `call_model.log`. If you encounter unexpected behavior, this file is the first place to look for detailed information about the execution flow.

### Archive and Evaluation Files

The evaluation workflow generates several files that are crucial for debugging:

- **`evals/archived_calls.jsonl`**: The source of truth for all original model calls. Every call to an advanced function (e.g., `generateObjectWithTemplates`, `generateObjectUsingTools`) is archived here.
- **`eval_calls.jsonl`**: The output of `run_eval.py`. Contains the new responses generated during an evaluation run.
- **`judgement_calls.jsonl`**: The output of `judge_eval.py`. Contains the scores and reasoning from the judge model.

Inspecting these files allows you to trace the entire lifecycle of a model call, from its original invocation through evaluation and judgment.
