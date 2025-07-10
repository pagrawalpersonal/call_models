import re
import logging

logger = logging.getLogger(__name__)

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