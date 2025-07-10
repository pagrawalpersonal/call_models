from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from google.genai.errors import ServerError, ClientError
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import GOOGLE_API_KEY
from json_manipulations import _extract_json_from_codeblock
import json
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)
import time

logger = logging.getLogger(__name__)

client = genai.Client(api_key=GOOGLE_API_KEY)

def should_retry_gemini_error(exception):
    """
    Determine if a Gemini API error should be retried
    """
    if isinstance(exception, ServerError):
        # Retry on server errors (5xx status codes)
        return exception.code >= 500
    elif isinstance(exception, ClientError):
        # Retry on rate limiting (429) and specific client errors
        return exception.code == 429
    # Also retry on general connection/timeout errors
    elif isinstance(exception, (ConnectionError, TimeoutError)):
        return True
    return False

def log_retry_attempt(retry_state):
    """Log retry attempt details"""
    seconds = retry_state.seconds_since_start if retry_state.seconds_since_start is not None else 0
    next_sleep = retry_state.next_action.sleep if retry_state.next_action and retry_state.next_action.sleep is not None else 0
    error_msg = str(retry_state.outcome.exception()) if retry_state.outcome is not None else "Unknown error"
    logger.warning(
        f"Gemini API retry attempt {retry_state.attempt_number} after {seconds:.2f}s. "
        f"Next attempt in {next_sleep:.2f}s. "
        f"Last error: {error_msg}"
    )

def log_after_retry(retry_state):
    """Log after retry completion"""
    seconds = retry_state.seconds_since_start if retry_state.seconds_since_start is not None else 0
    if retry_state.outcome is None:
        logger.error("Gemini API retry completed with no outcome")
        return
        
    if retry_state.outcome.failed:
        error_msg = str(retry_state.outcome.exception()) if retry_state.outcome.exception() is not None else "Unknown error"
        logger.error(
            f"All Gemini API retry attempts failed after {seconds:.2f}s. "
            f"Total attempts: {retry_state.attempt_number}. "
            f"Last error: {error_msg}"
        )
    else:
        logger.info(
            f"Gemini API request succeeded after {retry_state.attempt_number} attempts "
            f"and {seconds:.2f}s"
        )

# Retry decorator for Gemini API calls
gemini_retry_decorator = retry(
    stop=stop_after_attempt(5),  # More attempts for Gemini due to frequent overloading
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception(should_retry_gemini_error),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    after=log_after_retry,
    before=log_retry_attempt
)

def getText(query, model = "gemini-2.5-flash", useSearch=False, output_format="json"):
    google_search_tool = Tool(
        google_search = GoogleSearch()
    )

    if useSearch:
        config=GenerateContentConfig(
            tools=[google_search_tool],
            response_modalities=["TEXT"]
        )
    else:
        if output_format == "json":
            config=GenerateContentConfig(
                response_mime_type='application/json'
            )
        elif output_format == "markdown":
            config=GenerateContentConfig(
                response_mime_type='text/plain'
            )
        else:
            config=GenerateContentConfig(
                response_mime_type='text/plain'
            )
        

    @gemini_retry_decorator
    def make_api_call():
        return client.models.generate_content(
            model=model,
            contents=query,
            config=config
        )

    retry_count = 0
    while retry_count < 3:
        logger.info(f"Making Gemini API call with model: {model}, useSearch: {useSearch}")
        response = make_api_call()
        if response.candidates[0].content.parts is not None:
            break
        retry_count += 1
        time.sleep(1)

    response_text = "\n".join(each.text for each in response.candidates[0].content.parts)

    if "```json" in response_text:
        #print("^^^^^ REMOVING CODE BLOCK ^^^^")
        response_text = _extract_json_from_codeblock(response_text)


    if getattr(response.candidates[0], "grounding_metadata") and response.candidates[0].grounding_metadata:
        debug_info = {
            "web_search_queries": response.candidates[0].grounding_metadata.web_search_queries if getattr(response.candidates[0].grounding_metadata, "web_search_queries") else "",
            "domains": response.candidates[0].grounding_metadata.grounding_chunks if getattr(response.candidates[0].grounding_metadata, "grounding_chunks") else "",
            "full": response.candidates[0].grounding_metadata.model_dump_json()
        }
    else:
        debug_info = {}

    return response_text, debug_info