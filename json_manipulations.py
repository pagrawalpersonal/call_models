import re
import logging
import json
import re
from typing import List, Tuple
from enum import Enum

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


class DataType(Enum):
    """Represents the current data type being processed"""
    OBJECT = "object"      # Inside a JSON object {}
    ARRAY = "array"        # Inside a JSON array []
    STRING = "string"      # Inside a string ""
    KEY = "key"           # Processing a key in an object
    VALUE = "value"       # Processing a value in an object/array


class JSONRecoveryTool:
    """Tool to recover malformatted JSON using stack-based parsing"""
    
    def __init__(self):
        self.stack: List[DataType] = []
        self.brace_count = 0
        self.bracket_count = 0
        self.quote_count = 0
        self.escaped = False
        self.in_string = False
        
    def reset_state(self):
        """Reset the parsing state"""
        self.stack = []
        self.brace_count = 0
        self.bracket_count = 0
        self.quote_count = 0
        self.escaped = False
        self.in_string = False
    
    def extract_json_from_response(self, content: str) -> str:
        """
        Extract JSON content from a response that may contain other text.
        Looks for content after "Response:" and attempts to extract valid JSON.
        """
        # Find the Response: marker
        response_marker = "Response:"
        if response_marker in content:
            json_start = content.find(response_marker) + len(response_marker)
            json_content = content[json_start:].strip()
        else:
            # If no Response: marker, try to find JSON-like content
            json_content = content.strip()
        
        return json_content
    
    def is_json_start(self, content: str) -> bool:
        """Check if content starts with valid JSON structure"""
        content = content.strip()
        return content.startswith('{') or content.startswith('[')
    
    def find_json_start(self, content: str) -> int:
        """Find the start of JSON content"""
        for i, char in enumerate(content):
            if char in '{[':
                return i
        return -1
    
    def find_json_end(self, content: str, start_idx: int) -> int:
        """Find the end of JSON content using stack-based approach"""
        self.reset_state()
        last_valid_end = start_idx
        
        for i in range(start_idx, len(content)):
            char = content[i]
            
            if self.escaped:
                self.escaped = False
                continue
                
            if char == '\\':
                self.escaped = True
                continue
                
            if char == '"' and not self.escaped:
                self.in_string = not self.in_string
                self.quote_count += 1
                continue
                
            if self.in_string:
                continue
                
            if char == '{':
                self.brace_count += 1
                self.stack.append(DataType.OBJECT)
            elif char == '}':
                self.brace_count -= 1
                if self.stack and self.stack[-1] == DataType.OBJECT:
                    self.stack.pop()
                # Track the last position where we had balanced braces/brackets
                if self.brace_count >= 0 and self.bracket_count >= 0:
                    last_valid_end = i + 1
            elif char == '[':
                self.bracket_count += 1
                self.stack.append(DataType.ARRAY)
            elif char == ']':
                self.bracket_count -= 1
                if self.stack and self.stack[-1] == DataType.ARRAY:
                    self.stack.pop()
                # Track the last position where we had balanced braces/brackets
                if self.brace_count >= 0 and self.bracket_count >= 0:
                    last_valid_end = i + 1
                    
            # If we encounter clearly non-JSON content after a complete structure, stop here
            if (not self.in_string and 
                self.brace_count == 0 and self.bracket_count == 0 and
                char not in '{}[]",: \n\t\r' and 
                not char.isalnum() and 
                char not in '+-.eE'):
                break
                
        # Return the last valid position, or the full content if we processed everything
        return max(last_valid_end, start_idx)
    
    def fix_common_issues(self, json_content: str) -> str:
        """Fix common JSON formatting issues"""
        if not json_content.strip():
            return "{}"
            
        # Remove any trailing text after the JSON
        json_content = self.trim_to_json(json_content)
        
        # Fix common issues
        json_content = self.fix_invalid_escapes(json_content)  # Fix invalid escapes first
        json_content = self.fix_unescaped_quotes(json_content)
        json_content = self.fix_missing_closers(json_content)
        json_content = self.fix_extra_chars(json_content)
        json_content = self.fix_missing_commas(json_content)
        json_content = self.fix_line_breaks_in_strings(json_content)
        
        return json_content
    
    def trim_to_json(self, content: str) -> str:
        """Trim content to just the JSON part"""
        start_idx = self.find_json_start(content)
        if start_idx == -1:
            return "{}"
        
        # Find the last closing brace/bracket of the correct type
        if content[start_idx] == '{':
            last_idx = content.rfind('}')
        elif content[start_idx] == '[':
            last_idx = content.rfind(']')
        else:
            last_idx = len(content) - 1
        
        if last_idx == -1:
            # fallback to previous logic
            end_idx = self.find_json_end(content, start_idx)
            return content[start_idx:end_idx]
        else:
            return content[start_idx:last_idx+1]
    
    def fix_missing_closers(self, content: str) -> str:
        """Add missing closing brackets/braces"""
        self.reset_state()
        result = []
        
        for char in content:
            result.append(char)
            
            if self.escaped:
                self.escaped = False
                continue
                
            if char == '\\':
                self.escaped = True
                continue
                
            if char == '"' and not self.escaped:
                self.in_string = not self.in_string
                continue
                
            if self.in_string:
                continue
                
            if char == '{':
                self.brace_count += 1
                self.stack.append(DataType.OBJECT)
            elif char == '}':
                if self.stack and self.stack[-1] == DataType.OBJECT:
                    self.stack.pop()
                    self.brace_count -= 1
                else:
                    if self.stack and self.stack[-1] == DataType.ARRAY:
                        self.stack.pop()
                        self.brace_count -= 1
                        result[len(result) - 1] = ']'
            elif char == '[':
                self.bracket_count += 1
                self.stack.append(DataType.ARRAY)
            elif char == ']':
                
                if self.stack and self.stack[-1] == DataType.ARRAY:
                    self.stack.pop()
                    self.bracket_count -= 1
                else:
                    if self.stack and self.stack[-1] == DataType.OBJECT:
                        self.stack.pop()
                        self.brace_count -= 1
                        result[len(result) - 1] = '}'
        
        # Add missing closers
        while self.bracket_count > 0:
            result.append(']')
            self.bracket_count -= 1
            
        while self.brace_count > 0:
            result.append('}')
            self.brace_count -= 1
            
        return ''.join(result)
    
    def fix_extra_chars(self, content: str) -> str:
        """Remove extra characters that appear after valid JSON"""
        # Find the last valid JSON structure
        self.reset_state()
        last_valid_pos = 0
        
        for i, char in enumerate(content):
            if self.escaped:
                self.escaped = False
                continue
                
            if char == '\\':
                self.escaped = True
                continue
                
            if char == '"' and not self.escaped:
                self.in_string = not self.in_string
                continue
                
            if self.in_string:
                continue
                
            if char == '{':
                self.brace_count += 1
            elif char == '}':
                self.brace_count -= 1
                if self.brace_count == 0 and self.bracket_count == 0:
                    last_valid_pos = i + 1
            elif char == '[':
                self.bracket_count += 1
            elif char == ']':
                self.bracket_count -= 1
                if self.brace_count == 0 and self.bracket_count == 0:
                    last_valid_pos = i + 1
        
        # Remove trailing non-JSON characters
        if last_valid_pos > 0:
            content = content[:last_valid_pos]
        
        # Remove any trailing whitespace and non-JSON characters
        content = content.rstrip()
        content = re.sub(r'[^}\]]*$', '', content)
        
        return content
    
    def fix_missing_commas(self, content: str) -> str:
        """Add missing commas between array/object elements"""
        # This is a simplified version - in practice, this would need more sophisticated parsing
        # to avoid adding commas in strings
        lines = content.split('\n')
        result = []
        
        for i, line in enumerate(lines):
            
            # Add comma if line ends with } or ] and next line starts with { or [
            if (i < len(lines) - 1 and 
                line.strip().endswith(('}', ']')) and 
                lines[i + 1].strip().startswith(('{', '['))):
                result.append(line)
                result.append(',')
            elif (i < len(lines) - 1 and 
                line.strip().endswith((',')) and 
                lines[i + 1].strip().startswith(('}', ']'))):
                result.append(line[:len(line)-1])
            else:
                result.append(line)
        
        return '\n'.join(result)
    
    def fix_unescaped_quotes(self, content: str) -> str:
        """Fix unescaped quotes in strings by escaping them intelligently"""
        self.reset_state()
        result = []
        i = 0
        
        while i < len(content):
            char = content[i]
            
            if not self.in_string and char != '"':
                result.append(char)
                i += 1
                continue
            
            if self.escaped:
                self.escaped = False
                if char not in '\"\\\t\n\r':
                    result.append('\\')
                result.append(char)
                i += 1
                continue
                
            if char == '\\':
                self.escaped = True
                result.append(char)
                i += 1
                continue
                
            if char == '"' and not self.escaped:
                # Check if this quote can be escaped or if it's a legitimate string boundary
                if self.in_string:
                    # We're inside a string, check if this quote should end the string
                    # Look ahead to see if this quote is followed by valid JSON structure
                    can_end_string = False
                    if i < len(content) - 1:
                        next_char = content[i + 1]
                        # Valid characters that can follow a closing quote in JSON
                        if next_char in '}],:':
                            can_end_string = True
                        elif next_char in ' \n\t\r':
                            # Check if there's a valid JSON character after whitespace
                            j = i + 1
                            while j < len(content) and content[j] in ' \n\t\r':
                                j += 1
                            if j < len(content) and content[j] in '}],:':
                                can_end_string = True
                    
                    if can_end_string:
                        # This quote can legitimately end the string
                        self.in_string = False
                        result.append(char)
                        i += 1
                    else:
                        # This quote should be escaped
                        result.append('\\"')
                        i += 1
                else:
                    # We're not in a string, this starts a new string
                    self.in_string = True
                    result.append(char)
                    i += 1
                continue
            
            # If we're inside a string, add the character as-is
            if self.in_string:
                result.append(char)
                i += 1
                continue
            
            result.append(char)
            i += 1
        
        if self.in_string:
            result.append('"')
        return ''.join(result)
    
    def fix_line_breaks_in_strings(self, content: str) -> str:
        """Fix line breaks that occur in the middle of JSON strings"""
        self.reset_state()
        result = []
        i = 0
        
        while i < len(content):
            char = content[i]
            
            if self.escaped:
                self.escaped = False
                result.append(char)
                i += 1
                continue
                
            if char == '\\':
                self.escaped = True
                result.append(char)
                i += 1
                continue
                
            if char == '"' and not self.escaped:
                self.in_string = not self.in_string
                result.append(char)
                i += 1
                continue
                
            if self.in_string:
                # If we're in a string and encounter a newline, replace it with a space
                if char == '\n':
                    result.append('\\n')
                else:
                    result.append(char)
                i += 1
                continue
                
            result.append(char)
            i += 1
        
        return ''.join(result)
    
    def fix_invalid_escapes(self, content: str) -> str:
        """Fix invalid escape sequences in the JSON"""
        result = []
        i = 0
        
        while i < len(content):
            char = content[i]
            
            if char == '\\' and i < len(content) - 1:
                next_char = content[i + 1]
                # Valid escape sequences in JSON
                valid_escapes = ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']
                
                if next_char in valid_escapes:
                    # Valid escape sequence, keep it
                    result.append(char)
                    result.append(next_char)
                    i += 2
                else:
                    # Invalid escape sequence, remove the backslash
                    result.append(next_char)
                    i += 2
            else:
                result.append(char)
                i += 1
        
        return ''.join(result)
    
    def validate_and_fix(self, content: str) -> Tuple[str, bool]:
        """
        Validate JSON and attempt to fix it if invalid.
        Returns (fixed_json, is_valid)
        """
        # First, try to parse as-is
        try:
            json.loads(content)
            return content, True
        except json.JSONDecodeError:
            pass
        
        
        # Extract JSON from response
        json_content = self.extract_json_from_response(content)
        
        # Fix common issues
        fixed_content = self.fix_common_issues(json_content)
        
        # Try to parse the fixed content
        try:
            json.loads(fixed_content)
            return fixed_content, True
        except json.JSONDecodeError as e:
            # If still invalid, try more aggressive fixes
            aggressive_fixed = self.aggressive_fix(fixed_content)
            try:
                json.loads(aggressive_fixed)
                return aggressive_fixed, True
            except json.JSONDecodeError:
                return f"Error Validating: {e}\n" + fixed_content, False
    
    def aggressive_fix(self, content: str) -> str:
        """Apply more aggressive fixes for severely malformed JSON"""
        # Remove any trailing text that's clearly not JSON
        content = re.sub(r'[^}\]]*$', '', content)
        
        # Ensure we have matching braces/brackets
        open_braces = content.count('{')
        close_braces = content.count('}')
        open_brackets = content.count('[')
        close_brackets = content.count(']')
        
        # Add missing closers
        while close_braces < open_braces:
            content += '}'
            close_braces += 1
            
        while close_brackets < open_brackets:
            content += ']'
            close_brackets += 1
        
        return content
    
    def recover_json(self, file_path: str) -> Tuple[str, bool]:
        """
        Recover JSON from a file containing malformatted JSON.
        Returns (recovered_json, success)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return f"Error reading file: {e}", False
        
        return self.validate_and_fix(content)
