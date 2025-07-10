from typing import TypeVar, Type, List, Union, Optional, Dict, Any, Tuple
from pydantic import BaseModel, create_model, Field, ValidationError
from call_model import generateObject, generateText, RetryPolicy
import logging
import json
import time
import asyncio
from datetime import datetime
import aiofiles
import os
from string import Template
from typing import Literal
import ast
import re
import uuid
from pedantic_models import ArchiveInfo, DebugInfo, PydanticEncoder


logger = logging.getLogger(__name__)

LOG_FILE_PATH = "call_model.log"
LLM_CALL_ARCHIVE = "evals/archived_calls.jsonl"

def set_logging_level(level = logging.DEBUG):
    from openai import OpenAI
    import call_models.call_model as call_model
    """
    Sets the logging level for the current module, call_model module,
    and redirects OpenAI logging. All logs from these sources will also go to a file.
    """

    current_module_logger = logger

    # Ensure the file handler is created only once and re-used for all loggers
    file_handler = None
    if not any(isinstance(handler, logging.FileHandler) for handler in current_module_logger.handlers):
        file_handler = logging.FileHandler(LOG_FILE_PATH)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        #print(f"Created file handler for '{LOG_FILE_PATH}'")
    else:
        # If a file handler already exists on current_module_logger, retrieve it to reuse
        for handler in current_module_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                file_handler = handler
                break

    if file_handler:
        file_handler.setLevel(level) # Ensure the file handler's level is also updated

    # --- Configure current module's logger ---
    current_module_logger.setLevel(level)
    #print(f"Set '{current_module_logger.name}' logger level to {logging.getLevelName(level)}")
    if file_handler and file_handler not in current_module_logger.handlers:
        current_module_logger.addHandler(file_handler)
        #print(f"Added file handler to '{current_module_logger.name}' logger.")


    # --- Configure call_model.py's logger ---
    call_model_logger = logging.getLogger(call_model.__name__) # Get the logger named 'call_model'
    call_model_logger.setLevel(level)
    #print(f"Set '{call_model_logger.name}' logger level to {logging.getLevelName(level)}")
    # Remove any existing handlers from call_model_logger that might be sending to stderr/stdout
    for handler in list(call_model_logger.handlers):
        if isinstance(handler, logging.StreamHandler):
            call_model_logger.removeHandler(handler)
            #print(f"Removed StreamHandler from '{call_model_logger.name}' logger.")
    if file_handler and file_handler not in call_model_logger.handlers:
        call_model_logger.addHandler(file_handler)
        #print(f"Added file handler to '{call_model_logger.name}' logger.")


    # --- Configure OpenAI's logger ---
    openai_logger = logging.getLogger(OpenAI.__module__)
    openai_logger.setLevel(level)
    #print(f"Set '{openai_logger.name}' logger level to {logging.getLevelName(level)}")
    # Remove any existing handlers from OpenAI's logger that might be sending to stderr/stdout
    for handler in list(openai_logger.handlers):
        if isinstance(handler, logging.StreamHandler):
            openai_logger.removeHandler(handler)
            #print(f"Removed StreamHandler from '{openai_logger.name}' logger.")
    if file_handler and file_handler not in openai_logger.handlers:
        openai_logger.addHandler(file_handler)
        #print(f"Added file handler to '{openai_logger.name}' logger.")

T = TypeVar('T', bound=BaseModel)

class TextResponse(BaseModel):
    """Model for storing text response with debug information"""
    text: str
    debug_info: DebugInfo

def _create_debug_wrapper_model(base_model: Type[T]) -> Type[BaseModel]:
    """
    Create a wrapper model that includes debug information alongside the original model
    """
    return create_model(
        f'DebugWrapper_{base_model.__name__}',
        debug_info=(DebugInfo, ...),
        response=(base_model, ...)
    )

def _get_description_parameters(pydantic_model: Type[BaseModel]) -> dict:
    schema = pydantic_model.model_json_schema()
    parameters = {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", [])
    }

    return {
        "name": pydantic_model.__name__,
        "description": pydantic_model.__doc__.strip() if pydantic_model.__doc__ else "",
        "parameters": parameters["properties"]
    }

def try_extracting_json(content):
    extracted_content = _extract_json_from_codeblock(content)
    parsed_content = None

    '''
    if not extracted_content and content.startswith("```json"):
        # Remove ```json from start and ``` from end
        extracted_content = content[7:]  # Remove ```json
        extracted_content = extracted_content[:-3]
        extracted_content = extracted_content.rstrip("`")  # Remove trailing ```
        extracted_content = extracted_content.strip()  # Remove any extra whitespace
    else:
        extracted_content = content
    '''

    try:
        # Parse the JSON content
        parsed_content = json.loads(extracted_content)
        logger.debug(f"Parsed JSON content: {parsed_content}")
    except json.JSONDecodeError as e:
        logger.warn(f"Failed to parse JSON content First Attempt: {e}")

        try:
            parsed_content = ast.literal_eval(extracted_content)
        except Exception as ex:
            logger.warn(f"Failed to parse JSON content Second Attempt: {ex}")
            logger.warn(f"Raw content: {content}")
    
    return parsed_content or content

# Regex patterns for JSON extraction
_JSON_CODEBLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*)\s*```", re.DOTALL)
_JSON_PATTERN = re.compile(r"({[\s\S]*})")

def _extract_json_from_codeblock(content: str) -> str:
    """
    Extract JSON from a string that may contain markdown code blocks or plain JSON.

    This optimized version uses regex patterns to extract JSON more efficiently.

    Args:
        content: The string that may contain JSON

    Returns:
        The extracted JSON string
    """
    # First try to find JSON in code blocks
    match = _JSON_CODEBLOCK_PATTERN.search(content)
    if match:
        json_content = match.group(1).strip()
    else:
        logger.debug("extracting json - next pattern")
        # Look for JSON objects with the pattern { ... }
        match = _JSON_PATTERN.search(content)
        if match:
            json_content = match.group(1)
        else:
            # Fallback to the old method if regex doesn't find anything
            first_paren = content.find("{")
            last_paren = content.rfind("}")
            if first_paren != -1 and last_paren != -1:
                json_content = content[first_paren : last_paren + 1]
            else:
                json_content = content  # Return as is if no JSON-like content found

    return json_content

async def generateObjectUsingTools(
    model: str,
    system_prompt_template: str,
    system_prompt_inputs: Dict[str, str],
    user_prompt_template: str,
    user_prompt_inputs: Dict[str, str],
    response_model: Type[T],
    tools: Optional[Dict[str, Tuple[Type[BaseModel], callable]] ] = None,
    retry_policy: str = RetryPolicy.EXPONENTIAL,
    max_retries: int = 3,
    tag: Optional[str] = None,
    archive_path: Optional[str] = LLM_CALL_ARCHIVE,
    previous_messages: Optional[List[dict]] = None,
    is_eval_run: Optional[bool] = False
) -> T:

    system_prompt_template = system_prompt_template + """
    
    1. **Think step-by-step (Chain of Thought)** to decide what information is needed. You can only make ${remaining_turns} more back and forth turns.
    2. **If you can answer directly**, return:

    {
        "type": "answer",
        "thought": "Explain why you can answer without tools.",
        "response": ${response_model_params},
        "confidence": "High, Medium, Low confidence in your answer",
        "debug_info": {
            "prompt_difficulties": "What made this prompt hard to answer ",
            "prompt_improvements": "Changes to the prompt to improve the quality of output"
        }
    }

    3. **If you need a tool**, return the following. You can only make ${remaining_tool_calls} more tool calls. 
    If you have less than 2 tool calls remaining then focus on answering.

    {
        "type": "tool_calls",
        "thought": "Explain your reasoning for using tools and what you hope to retrieve.",
        "calls": [
            {
                "tool": "tool_name",
                "reason": "Explain why this specific tool is needed.",
                "parameters": {
                    "parameter1": "value"
                }
            }
        ],
        "debug_info": {
            "prompt_difficulties": "What made this prompt hard to answer ",
            "prompt_improvements": "Changes to the prompt to improve the quality of output"
        }
    }
    4. **Ensure the output is well formed JSON**. ONLY RETURN JSON output.

    Tools available:
    ${tools_params}
    """

    start_time = time.time()

    if not is_eval_run:
        response_model_params = _get_description_parameters(response_model)
        tools_params = [
            _get_description_parameters(tool_model) for tool_name, (tool_model, _) in tools.items()
        ]

        system_prompt_inputs["response_model_params"] = json.dumps(response_model_params, indent=2, cls=PydanticEncoder)
        system_prompt_inputs["tools_params"] = json.dumps(tools_params, indent=2, cls=PydanticEncoder)
    
    # Format the prompts with inputs if provided
    system_prompt = Template(system_prompt_template).safe_substitute(system_prompt_inputs) if system_prompt_inputs else system_prompt_template
    user_prompt = Template(user_prompt_template).safe_substitute(user_prompt_inputs) if user_prompt_inputs else user_prompt_template
    
    if "google" in model.lower() and previous_messages:
        previous_messages = [m for m in previous_messages if "tool_calls" not in m]
    
    if response_model and is_typed_dict(response_model):
        response_model: BaseModel = create_model(
            response_model.__name__,
            **{k: (v, ...) for k, v in response_model.__annotations__.items()},
        )

    messages = previous_messages or []
    max_iterations = 5  # Prevent infinite loops
    iteration = 0
    max_tool_calls = 10
    num_tool_calls = 0
    debug_info = None

    while iteration < max_iterations:
        iteration += 1
        logger.info(f"\n--- LLM Turn {iteration} ---")

        if not is_eval_run:
            system_prompt_inputs["remaining_turns"] = str(max_iterations - iteration)
            system_prompt_inputs["remaining_tool_calls"] = str(max_tool_calls - num_tool_calls)
            
        # Format the prompts with inputs if provided
        system_prompt = Template(system_prompt_template).safe_substitute(system_prompt_inputs) if system_prompt_inputs else system_prompt_template
        debug_info = None

        # Make a deep copy of messages before calling generateObject
        messages_copy = [msg.copy() for msg in messages] if messages else None
        
        action_response = await generateObject(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=None,
            retry_policy=retry_policy,
            max_retries=max_retries,
            previous_messages=messages
        )
        
        logger.debug(f"Turn {iteration} Response: {action_response.choices[0].message.content}")


        try:
            if is_eval_run:
                return action_response.choices[0].message.content, None
            
            parsed_content = try_extracting_json(action_response.choices[0].message.content)
            
            # Check if parsed_content is a string (error case)
            if isinstance(parsed_content, str):
                logger.warn(f"Error parsing response: {parsed_content}")
                return parsed_content, debug_info
                
            # Check if parsed_content has required fields
            if not isinstance(parsed_content, dict) or "type" not in parsed_content:
                logger.warn(f"Malformed response: {parsed_content}")
                return f"Error: Malformed response from model", debug_info

            if "debug_info" in parsed_content:
                try:
                    debug_info = DebugInfo(**parsed_content["debug_info"])
                except Exception as e:
                    logger.warn(f"Error creating User object: {e}")
                    debug_info = None

            if parsed_content["type"] == "answer":
                try:
                    response_object = response_model(**parsed_content["response"]["parameters"])
                    return response_object, debug_info
                except Exception as e:
                    logger.warn(f"Error creating response object: {e}")
                return parsed_content["response"], debug_info
            elif parsed_content["type"] == "tool_calls":
                if "calls" not in parsed_content:
                    logger.warn(f"Malformed tool_calls response: {parsed_content}")
                    return f"Error: Malformed tool_calls response from model", debug_info

                for tool in parsed_content["calls"]:
                    if not isinstance(tool, dict) or "tool" not in tool or "parameters" not in tool:
                        logger.warn(f"Malformed tool call: {tool}")
                        return f"Error: Malformed tool call in response", debug_info

                    tool_name = tool["tool"]
                    tool_args_json = tool["parameters"]

                    logger.info(f"LLM decided to call a tool: {tool_name} {tools.keys()}")

                    if tool_name in tools:
                        num_tool_calls += 1
                        try:
                            tool_model, execution_function = tools[tool_name]
                            tool_output_raw = execution_function(**tool_args_json)
                            
                            if asyncio.iscoroutine(tool_output_raw):
                                tool_output_raw = await tool_output_raw
                            
                            formatted_tool_output = f"Tool '{tool_name}' Output:\n{tool_output_raw}"
                            logger.debug(f"Tool output provided: {formatted_tool_output[:200]}...")

                            
                            if "google" not in model.lower():
                                messages.append({
                                    "role": "assistant", 
                                    "content": "", 
                                    "tool_calls": [{
                                        "id": f"call_{tool_name}", 
                                        "function": {
                                            "arguments": json.dumps(tool_args_json, indent=2, cls=PydanticEncoder), 
                                            "name": tool_name
                                        }, 
                                        "type": "function"
                                    }]
                                })
                            

                            messages.append({
                                "role": "tool", 
                                "tool_call_id": f"call_{tool_name}",
                                "name": tool_name, 
                                "content": formatted_tool_output
                            })
                            
                        except Exception as e:
                            logger.warn(f"Error executing tool '{tool_name}': {e}")
                            messages.append({
                                "role": "tool", 
                                "tool_call_id": f"call_{tool_name}", 
                                "name": tool_name, 
                                "content": f"Error executing tool: {e}"
                            })
                            return f"Error executing tool '{tool_name}': {e}", debug_info
                    else:
                        logger.warn(f"LLM requested unknown or unavailable tool: {tool_name}")
                        return f"LLM requested unknown or unavailable tool: {tool_name}", debug_info
            else:
                logger.warn(f"Unexpected response type: {parsed_content['type']}")
                return f"Error: Unexpected response type from model", debug_info

        except Exception as e:
            logger.warn(f"Error processing response: {e}")
            return f"Error processing model response: {str(e)}", debug_info
        finally:
            # Create archive info
            local_system_prompt_inputs = system_prompt_inputs
            if not is_eval_run:
                local_system_prompt_inputs = {
                        **system_prompt_inputs,
                        'response_model_params': json.dumps(response_model_params, indent=2, cls=PydanticEncoder),
                        'tools_params': json.dumps(tools_params, indent=2, cls=PydanticEncoder)
                    }
            archive_info = ArchiveInfo(
                timestamp=datetime.now(),
                model=model,
                system_prompt_template=system_prompt_template,
                system_prompt_inputs=local_system_prompt_inputs,
                user_prompt_template=user_prompt_template,
                user_prompt_inputs=user_prompt_inputs,
                response_model=json.dumps(response_model.model_json_schema()),
                tools=[{'name': name} for name in tools.keys()] if tools else None,
                tag=tag,
                time_taken=time.time() - start_time,
                input_tokens=getattr(action_response.usage, 'prompt_tokens', None),
                output_tokens=getattr(action_response.usage, 'completion_tokens', None),
                response=action_response.choices[0].message.content,
                debug_info=debug_info,
                messages=messages_copy
            )
            # Archive the call in a background task
            if archive_path:
                asyncio.create_task(_archive_call(archive_info, archive_path))

    return "Error: Maximum number of iterations reached without getting an answer", debug_info

def is_typed_dict(cls) -> bool:
    return (
        isinstance(cls, type)
        and issubclass(cls, dict)
        and hasattr(cls, "__annotations__")
    )

async def generateObjectWithTemplates(
    model: str,
    system_prompt_template: str,
    system_prompt_inputs: Dict[str, str],
    user_prompt_template: str,
    user_prompt_inputs: Dict[str, str],
    response_model: Type[T],
    retry_policy: str = RetryPolicy.EXPONENTIAL,
    max_retries: int = 3,
    tag: Optional[str] = None,
    archive_path: Optional[str] = LLM_CALL_ARCHIVE,
    previous_messages: Optional[List[dict]] = None
) -> Union[Tuple[T, DebugInfo], List[dict]]:
    """
    Generate a structured object using OpenRoute.ai model with instructor, supporting template-based prompts.
    Includes debug information about prompt quality and potential improvements.
    
    Args:
        model: The model name to use (e.g., "anthropic/claude-3-opus-20240229")
        system_prompt_template: Template string for the system prompt
        system_prompt_inputs: Dictionary of values to fill in the system prompt template
        user_prompt_template: Template string for the user prompt
        user_prompt_inputs: Dictionary of values to fill in the user prompt template
        response_model: The Pydantic model class for structured output
        api_key: OpenRoute.ai API key (optional, defaults to OPENROUTER_API_KEY env var)
        tools: Optional list of tool definitions that the model can use
        base_url: OpenRoute.ai API base URL (optional, defaults to OPENROUTER_BASE_URL env var)
        retry_policy: The retry policy to use (exponential, fixed, or no_retry)
        max_retries: Maximum number of retry attempts
        tag: Optional tag to identify this call in the archive
        archive_path: Path to the JSONL file for archiving calls
        
    Returns:
        Either a tuple containing:
            - An instance of the specified response_model with structured data
            - A DebugInfo object containing debug information
        Or a list of tool calls to execute
    """
    start_time = time.time()
    
    # Format the prompts with inputs if provided
    system_prompt = Template(system_prompt_template).safe_substitute(system_prompt_inputs) if system_prompt_inputs else system_prompt_template
    user_prompt = Template(user_prompt_template).safe_substitute(user_prompt_inputs) if user_prompt_inputs else user_prompt_template
    
    if response_model and is_typed_dict(response_model):
        response_model: BaseModel = create_model(
            response_model.__name__,
            **{k: (v, ...) for k, v in response_model.__annotations__.items()},
        )

    # Add debug instructions to the system prompt
    debug_system_prompt = f"""{system_prompt}

Additionally, please provide debug information about:
1. What aspects of the prompt made it difficult to answer (if any)
2. Suggestions for improving the prompt to get better results

This debug information will be used to improve future interactions but won't be shown to the end user."""

    # Create a wrapper model that includes debug information
    debug_wrapper_model = _create_debug_wrapper_model(response_model)
    
    if "google" in model.lower() and previous_messages:
        previous_messages = [m for m in previous_messages if "tool_calls" not in m]


    # Call generateObject with the debug wrapper model
    _response = await generateObject(
        model=model,
        system_prompt=debug_system_prompt,
        user_prompt=user_prompt,
        response_model=debug_wrapper_model,
        retry_policy=retry_policy,
        max_retries=max_retries,
        previous_messages=previous_messages
    )
    
    # If the response is a list of tool calls, return it as is
    if isinstance(_response, list):
        return _response
    
    #TODO archive the call when the debug_reponse is a list

    logger.info(f"Model response: {_response.response}")
    
    # Log the debug information
    logger.info("Debug information from model response:")
    logger.info(f"Prompt difficulties: {_response.debug_info.prompt_difficulties}")
    logger.info(f"Prompt improvements: {_response.debug_info.prompt_improvements}")
    
    response_model_schema = json.dumps(response_model.model_json_schema()) if hasattr(response_model, "model_json_schema") else ""
    
    # Create archive info
    archive_info = ArchiveInfo(
        timestamp=datetime.now(),
        model=model,
        system_prompt_template=system_prompt_template,
        system_prompt_inputs=system_prompt_inputs,
        user_prompt_template=user_prompt_template,
        user_prompt_inputs=user_prompt_inputs,
        response_model=response_model_schema,
        tools=None,
        tag=tag,
        time_taken=time.time() - start_time,
        input_tokens=getattr(_response, 'usage', {}).get('prompt_tokens'),
        output_tokens=getattr(_response, 'usage', {}).get('completion_tokens'),
        response=_response.response,
        debug_info=_response.debug_info
    )
    
    # Archive the call in a background task
    if archive_path:
        asyncio.create_task(_archive_call(archive_info, archive_path))
    
    # Return both the response and debug info
    return _response.response, _response.debug_info

async def _archive_call(archive_info: ArchiveInfo, archive_path: str):
    """Archive a model call to a JSONL file"""
    try:
        # Create directory if it doesn't exist and path contains directory component
        dirname = os.path.dirname(archive_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        # Append to JSONL file
        async with aiofiles.open(archive_path, 'a') as f:
            await f.write(archive_info.model_dump_json() + '\n')
            
    except Exception as e:
        logger.error(f"Error archiving model call: {str(e)}", exc_info=True)

async def generateTextWithTemplates(
    model: str,
    system_prompt_template: str,
    system_prompt_inputs: Dict[str, str],
    user_prompt_template: str,
    user_prompt_inputs: Dict[str, str],
    retry_policy: str = RetryPolicy.EXPONENTIAL,
    max_retries: int = 3,
    tag: Optional[str] = None,
    archive_path: Optional[str] = LLM_CALL_ARCHIVE
) -> Union[Tuple[str, DebugInfo], List[dict]]:
    """
    Generate text output using OpenRoute.ai model, supporting template-based prompts.
    Includes debug information about prompt quality and potential improvements.
    
    Args:
        model: The model name to use (e.g., "anthropic/claude-3-opus-20240229")
        system_prompt_template: Template string for the system prompt
        system_prompt_inputs: Dictionary of values to fill in the system prompt template
        user_prompt_template: Template string for the user prompt
        user_prompt_inputs: Dictionary of values to fill in the user prompt template
        api_key: OpenRoute.ai API key (optional, defaults to OPENROUTER_API_KEY env var)
        tools: Optional list of tool definitions that the model can use
        base_url: OpenRoute.ai API base URL (optional, defaults to OPENROUTER_BASE_URL env var)
        retry_policy: The retry policy to use (exponential, fixed, or no_retry)
        max_retries: Maximum number of retry attempts
        tag: Optional tag to identify this call in the archive
        archive_path: Path to the JSONL file for archiving calls
        
    Returns:
        Either a tuple containing:
            - A string containing the model's text response
            - A DebugInfo object containing debug information
        Or a list of tool calls to execute
    """
    start_time = time.time()
    
    # Format the prompts with inputs if provided
    system_prompt = Template(system_prompt_template).safe_substitute(system_prompt_inputs) if system_prompt_inputs else system_prompt_template
    user_prompt = Template(user_prompt_template).safe_substitute(user_prompt_inputs) if user_prompt_inputs else user_prompt_template
    
    # Call generateText with the TextResponse model
    response = await generateText(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        retry_policy=retry_policy,
        max_retries=max_retries
    )
    
    logger.debug(f"_generateTextWithTemplates: Response \n\n {response}\n\n")
    # If the response is a list of tool calls, return it as is
    if isinstance(response, list):
        return response
    
    # If the response is a TextResponse object, extract the text and debug info
    if isinstance(response, TextResponse):
        text = response.text
    else:
        # Handle the case where response is a string (no tools used)
        text = response
    
    # Create archive info
    archive_info = ArchiveInfo(
        timestamp=datetime.now(),
        model=model,
        system_prompt_template=system_prompt_template,
        system_prompt_inputs=system_prompt_inputs,
        user_prompt_template=user_prompt_template,
        user_prompt_inputs=user_prompt_inputs,
        response_model="",
        tools=None,
        tag=tag,
        time_taken=time.time() - start_time,
        input_tokens=getattr(response, 'usage', {}).get('prompt_tokens'),
        output_tokens=getattr(response, 'usage', {}).get('completion_tokens'),
        response=text,
        debug_info=None
    )
    
    # Archive the call in a background task
    if archive_path:
        asyncio.create_task(_archive_call(archive_info, archive_path))
    
    # Return both the text and debug info
    return text
