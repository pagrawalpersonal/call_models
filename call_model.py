from typing import TypeVar, Type, List, Union, Optional, Callable, Dict
import instructor
from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError
from pydantic import BaseModel
from instructor import patch
from dotenv import load_dotenv
import os
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
    retry_if_exception_type,
    before_sleep_log,
    retry_if_exception
)
import time
from call_model_config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

# Define hook functions for instructor
def log_kwargs(**kwargs):
    logger.info(f"Instructor: Function called with kwargs: {kwargs}")

def log_exception(exception: Exception):
    logger.info(f"Instructor: exception occurred: {str(exception)}")

def log_parseerror(exception: Exception):
    logger.info(f"Instructor: parsing error occurred: {str(exception)}")

class RetryPolicy:
    """Different retry policies for API calls"""
    EXPONENTIAL = "exponential"  # Exponential backoff
    FIXED = "fixed"  # Fixed interval
    NO_RETRY = "no_retry"  # No retries

def log_retry_attempt(retry_state):
    """Log retry attempt details"""
    seconds = retry_state.seconds_since_start if retry_state.seconds_since_start is not None else 0
    next_sleep = retry_state.next_action.sleep if retry_state.next_action and retry_state.next_action.sleep is not None else 0
    error_msg = str(retry_state.outcome.exception()) if retry_state.outcome is not None else "Unknown error"
    logger.warning(
        f"Retry attempt {retry_state.attempt_number} after {seconds:.2f}s. "
        f"Next attempt in {next_sleep:.2f}s. "
        f"Last error: {error_msg}"
    )

def log_after_retry(retry_state):
    """Log after retry completion"""
    seconds = retry_state.seconds_since_start if retry_state.seconds_since_start is not None else 0
    if retry_state.outcome is None:
        logger.error("Retry completed with no outcome")
        return
        
    if retry_state.outcome.failed:
        error_msg = str(retry_state.outcome.exception()) if retry_state.outcome.exception() is not None else "Unknown error"
        logger.error(
            f"All retry attempts failed after {seconds:.2f}s. "
            f"Total attempts: {retry_state.attempt_number}. "
            f"Last error: {error_msg}"
        )
    else:
        logger.info(
            f"Request succeeded after {retry_state.attempt_number} attempts "
            f"and {seconds:.2f}s"
        )

class OpenRouterClient:
    """Singleton client for OpenRouter API with separate clients for each mode"""
    _instance = None
    _clients: Dict[instructor.Mode, AsyncOpenAI] = {}
    _api_key: Optional[str] = None
    _base_url: Optional[str] = None
    _client_untyped: AsyncOpenAI = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenRouterClient, cls).__new__(cls)
        return cls._instance
    
    def initialize(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the client configuration"""
        self._api_key = api_key or OPENROUTER_API_KEY
        self._base_url = base_url or OPENROUTER_BASE_URL
        
        if not self._api_key:
            raise ValueError("API key must be provided either as an argument or through OPENROUTER_API_KEY environment variable")
        
        logger.info("Initializing OpenRouter client configuration")
    
    def get_client(self, mode: instructor.Mode) -> AsyncOpenAI:
        """
        Get or create a client for the specified mode
        
        Args:
            mode: The instructor mode to use
            
        Returns:
            An OpenAI client configured with the specified mode
        """
        if mode not in self._clients:
            if not self._api_key:
                self.initialize()
            
            logger.info(f"Creating new client for mode: {mode}")
            self._clients[mode] = instructor.from_openai(
                AsyncOpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url
                ),
                mode=mode
            )
            self._clients[mode].on("completion:kwargs", log_kwargs)
            self._clients[mode].on("completion:error", log_exception)
            self._clients[mode].on("parse:error", log_parseerror)
        
        return self._clients[mode]
    
    def get_client_untyped(self) -> AsyncOpenAI:
        """
        Get or create a client for the specified mode
        
        Args:
            mode: The instructor mode to use
            
        Returns:
            An OpenAI client configured with the specified mode
        """
        if not self._client_untyped:
            if not self._api_key:
                self.initialize()
            
            logger.info(f"Creating new untyped client")
            self._client_untyped = AsyncOpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url
                )
        
        return self._client_untyped

def get_retry_decorator(policy: str = RetryPolicy.EXPONENTIAL, max_attempts: int = 3) -> Callable:
    """
    Get a retry decorator based on the specified policy
    
    Args:
        policy: The retry policy to use (exponential, fixed, or no_retry)
        max_attempts: Maximum number of retry attempts
        
    Returns:
        A retry decorator configured with the specified policy
    """
    if policy == RetryPolicy.NO_RETRY:
        return lambda x: x  # No-op decorator
        
    retry_exceptions = (
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        APIError
    )
    
    if policy == RetryPolicy.EXPONENTIAL:
        return retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            retry=retry_if_exception_type(retry_exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=log_after_retry,
            before=log_retry_attempt
        )
    elif policy == RetryPolicy.FIXED:
        return retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_fixed(5),  # 5 seconds between retries
            retry=retry_if_exception_type(retry_exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=log_after_retry,
            before=log_retry_attempt
        )
    else:
        raise ValueError(f"Unknown retry policy: {policy}")

async def generateObject(
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
    retry_policy: str = RetryPolicy.EXPONENTIAL,
    max_retries: int = 3,
    previous_messages: Optional[List[dict]] = None
) -> Union[T, List[dict]]:
    """
    Generate a structured object using OpenRoute.ai model with instructor.
    Can return either a structured response or a list of tool calls to execute.
    
    Args:
        model: The model name to use (e.g., "anthropic/claude-3-opus-20240229")
        system_prompt: The system prompt to guide the model's behavior
        user_prompt: The user prompt containing the task
        response_model: The Pydantic model class for structured output
        api_key: OpenRoute.ai API key (optional, defaults to OPENROUTER_API_KEY env var)
        tools: Optional list of tool definitions that the model can use
        base_url: OpenRoute.ai API base URL (optional, defaults to OPENROUTER_BASE_URL env var)
        retry_policy: The retry policy to use (exponential, fixed, or no_retry)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Either an instance of the specified response_model with structured data,
        or a list of tool calls to execute
        
    Raises:
        ValueError: If API key is not provided
        APIError: If the API call fails after all retries
    """
    try:
        # Initialize or get the singleton client
        client = OpenRouterClient()
        mode = instructor.Mode.JSON
        openai_client = client.get_client(mode)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        if previous_messages:
            messages.extend(previous_messages) 
        
        # Get retry decorator based on policy
        retry_decorator = get_retry_decorator(retry_policy, max_retries)
        
        @retry_decorator
        async def make_api_call():
            return await openai_client.chat.completions.create(
                model=model,
                response_model=response_model,
                messages=messages
            )
        
        logger.info(f"Making API call with retry policy: {retry_policy}, max retries: {max_retries}")
        return await make_api_call()
        
    except Exception as e:
        logger.error(f"Error in generateObject: {str(e)}", exc_info=True)
        raise

async def generateText(
    model: str,
    system_prompt: str,
    user_prompt: str,
    retry_policy: str = RetryPolicy.EXPONENTIAL,
    max_retries: int = 3
) -> Union[str, List[dict]]:
    """
    Generate text output using OpenRoute.ai model.
    Can return either plain text or a list of tool calls to execute.
    
    Args:
        model: The model name to use (e.g., "anthropic/claude-3-opus-20240229")
        system_prompt: The system prompt to guide the model's behavior
        user_prompt: The user prompt containing the task
        api_key: OpenRoute.ai API key (optional, defaults to OPENROUTER_API_KEY env var)
        tools: Optional list of tool definitions that the model can use
        base_url: OpenRoute.ai API base URL (optional, defaults to OPENROUTER_BASE_URL env var)
        retry_policy: The retry policy to use (exponential, fixed, or no_retry)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Either a string containing the model's text response,
        or a list of tool calls to execute
        
    Raises:
        ValueError: If API key is not provided
        APIError: If the API call fails after all retries
    """
    try:
        # Initialize or get the singleton client
        client = OpenRouterClient()
        mode = instructor.Mode.JSON
        openai_client = client.get_client_untyped()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Get retry decorator based on policy
        retry_decorator = get_retry_decorator(retry_policy, max_retries)
        
        @retry_decorator
        async def make_api_call():
            response = await openai_client.chat.completions.create(
                model=model,
                messages=messages
            )
            #print(f"OpenAI response: {response}")
            return response.choices[0].message.content
        
        logger.info(f"Making API call with retry policy: {retry_policy}, max retries: {max_retries}")
        return await make_api_call()
        
    except Exception as e:
        logger.error(f"Error in generateText: {str(e)}", exc_info=True)
        raise
