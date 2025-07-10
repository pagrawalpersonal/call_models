import json
import asyncio
import argparse
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from call_models.call_model_with_evals import generateObjectWithTemplates, RetryPolicy
import logging
from string import Template
from constants import JUDGEMENT_MODEL, EVAL_CALLS_FILE, JUDGEMENT_CALLS_FILE

# Set up logging
logger = logging.getLogger(__name__)


class JudgeResponse(BaseModel):
    """Model for the judge's evaluation response"""
    score: float = Field(description="Score from -1 to 1 indicating how well the new response matches the original")
    reasoning: str = Field(description="Detailed reasoning for the score")
    improvements: Optional[str] = Field(description="Suggestions for improving the response", default=None)
    
async def evaluate_response(
    original_response: str,
    new_response: str,
    system_prompt_template: str,
    system_prompt_inputs: Dict[str, Any],
    user_prompt_template: str,
    user_prompt_inputs: Dict[str, Any],
    judge_model: str = "openai/gpt-4.1"
) -> Tuple[JudgeResponse, Optional[Dict[str, Any]]]:
    """Evaluate a new response against the original response using a judge model."""
    
    system_prompt = """You are an expert evaluator of AI model responses. Your task is to 
    look at the prompts used to call two different models, and compare the new response against an original response and determine how well they compare in terms of:
1. Semantic meaning and content
2. Completeness of information in terms of answering the prompts.
3. Accuracy of facts
4. Clarity and coherence

Provide a score from -1 to 1 where:
- 1 means the new response is significantly better.
- 0 means responses are essentially identical in meaning and quality
- -1 means the responses are completely different or the new response is significantly worse

Also provide detailed reasoning for your score and any suggestions for improvement."""

    # Format the original prompts with their inputs
    formatted_system_prompt = Template(system_prompt_template).safe_substitute(system_prompt_inputs) if system_prompt_inputs else system_prompt_template
    formatted_user_prompt = Template(user_prompt_template).safe_substitute(user_prompt_inputs) if user_prompt_inputs else user_prompt_template

    user_prompt = f"""Context:
**System Prompt**:
{formatted_system_prompt}

**User Prompt**:
{formatted_user_prompt}

**Original Response**:
{original_response}

**New Response**:
{new_response}

Please evaluate how well the new response matches the original response, taking into account the context provided by the system and user prompts."""

    #print(f"Original Response:\n{original_response}\n New Response:{new_response}\n")

    response, debug_info = await generateObjectWithTemplates(
        model=judge_model,
        system_prompt_template=system_prompt,
        system_prompt_inputs={},
        user_prompt_template=user_prompt,
        user_prompt_inputs={},
        response_model=JudgeResponse,
        retry_policy=RetryPolicy.EXPONENTIAL,
        max_retries=3,
        tag="judge_eval"
    )

    return response, debug_info

async def process_eval_file(
    eval_file: str,
    judge_model: str,
    archive_id_filter: Optional[str] = None,
    eval_id_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Process evaluation results and run judge evaluation."""
    results = []
    
    try:
        with open(eval_file, 'r') as f:
            for line in f:
                try:
                    eval_data = json.loads(line.strip())
                    
                    # Skip if ID filter is specified and doesn't match
                    if archive_id_filter and eval_data.get("archive_id") != archive_id_filter:
                        continue

                    if eval_id_filter and eval_data.get("eval_run_id", "") != eval_id_filter:
                        continue
                    
                    logger.info(f"Processing evaluation for ID: {eval_data.get('archive_id')}")
                    
                    # Run judge evaluation
                    judge_response, debug_info = await evaluate_response(
                        original_response=eval_data["original_response"],
                        new_response=eval_data["response"],
                        system_prompt_template=eval_data["system_prompt_template"],
                        system_prompt_inputs=eval_data["system_prompt_inputs"],
                        user_prompt_template=eval_data["user_prompt_template"],
                        user_prompt_inputs=eval_data["user_prompt_inputs"],
                        judge_model=judge_model
                    )
                    
                    result = {
                        "archive_id": eval_data["archive_id"],
                        "model_new": eval_data["model_used"],
                        "model_original": eval_data["model_original"] if "model_original" in eval_data else "",
                        "judge_score": judge_response.score,
                        "judge_reasoning": judge_response.reasoning,
                        "judge_improvements": judge_response.improvements,
                        "original_response": eval_data["original_response"],
                        "new_response": eval_data["response"],
                        "datetime": eval_data["datetime"],
                        "debug_info": debug_info
                    }
                    
                    results.append(result)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON line: {e}")
                    raise
                    
    except Exception as e:
        logger.error(f"Error processing evaluation file: {e}")
        raise
        
    return results

async def main():
    from call_models.call_model_with_evals import set_logging_level
    import logging
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.CRITICAL)

    set_logging_level(logging.DEBUG)

    parser = argparse.ArgumentParser(description="Evaluate model responses using a judge model")
    parser.add_argument("--eval-file", default=EVAL_CALLS_FILE, help="Path to the evaluation results JSONL file")
    parser.add_argument("--judge-model", default=JUDGEMENT_MODEL, help="Model to use for judging")
    parser.add_argument("--id", help="Optional ID to filter evaluations. This is id of the original archived model call.")
    parser.add_argument("--eval_run_id", help="Optional eval run ID to filter evaluations.")
    parser.add_argument("--output", default=JUDGEMENT_CALLS_FILE, help="Optional output file for results")
    
    args = parser.parse_args()
    
    results = await process_eval_file(
        eval_file=args.eval_file,
        judge_model=args.judge_model,
        archive_id_filter=args.id,
        eval_id_filter=args.eval_run_id
    )
    
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
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

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
