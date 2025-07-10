import json
import asyncio
from typing import Optional, List, Dict, Any
from call_models.call_model_with_evals import generateObjectWithTemplates, RetryPolicy, generateObjectUsingTools, LLM_CALL_ARCHIVE
from pedantic_models import ArchiveInfo, PydanticEncoder
import logging
from pydantic import create_model, BaseModel
from typing import Any, Optional, List
from datetime import datetime
import uuid
from constants import EVAL_CALLS_FILE


# Set up logging
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function to map JSON Schema types to Python types
def map_json_type_to_python(json_type: dict) -> Any:
    """A simple mapper for JSON schema types to Python types."""
    type_str = json_type.get("type")
    if type_str == "string":
        return str
    elif type_str == "integer":
        return int
    elif type_str == "number":
        return float
    elif type_str == "boolean":
        return bool
    elif type_str == "array":
        # Handle lists, recursively find the item type
        item_type = map_json_type_to_python(json_type.get("items", {}))
        return List[item_type]
    # Add other types as needed (e.g., object for nested models)
    return Any

def create_pydantic_model_from_schema(schema: str):
    """
    Dynamically creates a Pydantic model.
    """

    model_name = schema.get("title", "DynamicModel")
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    fields = {}
    for name, prop_schema in properties.items():
        python_type = map_json_type_to_python(prop_schema)
        
        # Determine if the field is required or has a default
        if name in required_fields:
            # Ellipsis (...) marks the field as required
            fields[name] = (python_type, ...)
        else:
            # Field is optional, provide a default value (e.g., None or from schema)
            default_value = prop_schema.get("default", None)
            fields[name] = (Optional[python_type], default_value)

    # Dynamically create the Pydantic model class
    DynamicModel = create_model(model_name, **fields)
    DynamicModel.__doc__ = schema.get("description")
    
    return DynamicModel

async def process_archive_file(
    archive_path: str,
    tag_filter: Optional[str] = None,
    id_filter: Optional[str] = None,
    new_model: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Process an archive file and run generateObjectWithTemplates for matching records.
    
    Args:
        archive_path: Path to the JSONL archive file
        tag_filter: Optional tag to filter records
        id_filter: Optional ID to filter records
        new_model: Optional model name to use instead of the archived model
        
    Returns:
        List of results from generateObjectWithTemplates calls
    """
    results = []
    
    try:
        with open(archive_path, 'r') as f:
            for line in f:
                try:
                    # Parse ArchiveInfo record
                    archive_data = json.loads(line.strip())
                    archive_info = ArchiveInfo(**archive_data)
                    
                    # Check if record matches filters
                    if tag_filter and archive_info.tag != tag_filter:
                        continue
                    if id_filter and archive_info.id != id_filter:
                        continue
                    
                    logger.info(f"Processing record with ID: {archive_info.id}")
                    print((f"Processing record with ID: {archive_info.id}"))
                    
                    # Use new_model if specified, otherwise use the archived model
                    model_to_use = new_model if new_model else archive_info.model
                    
                    hydrated_response_model = create_pydantic_model_from_schema(json.loads(archive_info.response_model)) if archive_info.response_model else None

                    if archive_info.tools :
                        response, debug_info = await generateObjectUsingTools(
                            model=model_to_use,
                            system_prompt_template=archive_info.system_prompt_template,
                            system_prompt_inputs=archive_info.system_prompt_inputs,
                            user_prompt_template=archive_info.user_prompt_template,
                            user_prompt_inputs=archive_info.user_prompt_inputs,
                            response_model=hydrated_response_model,
                            retry_policy=RetryPolicy.EXPONENTIAL,
                            max_retries=3,
                            tag=f"eval_{archive_info.id}",
                            previous_messages=archive_info.messages,
                            is_eval_run=True,
                        )
                    else:
                        response, debug_info = await generateObjectWithTemplates(
                            model=model_to_use,
                            system_prompt_template=archive_info.system_prompt_template,
                            system_prompt_inputs=archive_info.system_prompt_inputs,
                            user_prompt_template=archive_info.user_prompt_template,
                            user_prompt_inputs=archive_info.user_prompt_inputs,
                            response_model=hydrated_response_model,
                            retry_policy=RetryPolicy.EXPONENTIAL,
                            max_retries=3,
                            tag=f"eval_{archive_info.id}",
                            previous_messages=archive_info.messages,
                        )
                    
                    print(f"RESPONSE:\n{response}\nDebugInfo:\n{debug_info}")

                    results.append({
                        "archive_id": archive_info.id,
                        "eval_run_id": str(uuid.uuid4()),
                        "model_used": model_to_use,
                        "model_original": archive_info.model,
                        "response": response,
                        "debug_info": debug_info,
                        "original_response": archive_info.response,
                        "datetime": datetime.now().isoformat(),
                        "system_prompt_template": archive_info.system_prompt_template,
                        "system_prompt_inputs": archive_info.system_prompt_inputs,
                        "user_prompt_template": archive_info.user_prompt_template,
                        "user_prompt_inputs": archive_info.user_prompt_inputs,
                        "previous_messages": archive_info.messages
                    })
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON line: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Error processing record: {e}")
                    raise
                    
    except FileNotFoundError:
        logger.error(f"Archive file not found: {archive_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading archive file: {e}")
        raise
        
    return results

async def main():
    import argparse
    from call_models.call_model_with_evals import set_logging_level
    import logging
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.CRITICAL)

    set_logging_level(logging.DEBUG)
    
    parser = argparse.ArgumentParser(description='Process archive file and run evaluations')
    parser.add_argument('--archive', default=LLM_CALL_ARCHIVE, help=f'Path to the JSONL archive file (default: {LLM_CALL_ARCHIVE})')
    parser.add_argument('--tag', help='Filter records by tag')
    parser.add_argument('--id', help='Filter records by ID')
    parser.add_argument('--model', help='Model to use instead of the archived model')
    parser.add_argument('--output', help='Path to save results (optional)')
    
    args = parser.parse_args()
    
    results = await process_archive_file(
        archive_path=args.archive,
        tag_filter=args.tag,
        id_filter=args.id,
        new_model=args.model
    )
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, cls=PydanticEncoder)
    else:
        # Append results to eval_run_calls.jsonl
        with open(EVAL_CALLS_FILE, 'a') as f:
            for result in results:
                f.write(json.dumps(result, cls=PydanticEncoder) + '\n')
        try:
            print(json.dumps(results, indent=2, cls=PydanticEncoder))
        except TypeError as e:
            print("Error serializing results to JSON:", e)
            print("Raw results:", results)

    # 1. Get the current task (which is this main_fixed coroutine)
    current_task = asyncio.current_task()

    # 2. Get all other tasks running on the loop
    all_other_tasks = {
        task for task in asyncio.all_tasks() if task is not current_task
    }

    if all_other_tasks:
        print(f"Main: Found {len(all_other_tasks)} background task(s). Waiting for them to complete.")
        # 3. Wait for all of them to finish
        await asyncio.gather(*all_other_tasks)

if __name__ == "__main__":
    asyncio.run(main())
