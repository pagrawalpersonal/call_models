from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import json
import uuid

class PydanticEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle Pydantic models"""
    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

class QueryAnalysisResult(BaseModel):
    """Model for query analysis results with retrieval strategy recommendations"""
    query_type: str = Field(..., description="The identified category of the user's query")
    retrieval_strategy: str = Field(..., description="The recommended retrieval strategy or a combination of strategies")
    multi_hop: bool = Field(..., description="True if the query requires synthesizing information from multiple documents or steps")
    keywords: List[str] = Field(..., description="List of extracted keywords or named entities")
    sub_queries: List[str] = Field(..., description="List of rewritten or decomposed questions if multi_hop is true")

    def __str__(self) -> str:
        """String representation of query analysis result"""
        return json.dumps(self.model_dump(), indent=2, cls=PydanticEncoder)

    def __repr__(self) -> str:
        """String representation for debugging"""
        return json.dumps(self.model_dump(), indent=2, cls=PydanticEncoder)

class DebugInfo(BaseModel, extra="allow"):
    """Model for storing debug information about the prompt and response"""
    prompt_difficulties: Optional[str] = None
    prompt_improvements: Optional[str] = None

    def __str__(self) -> str:
        """String representation of debug info"""
        return json.dumps(self.model_dump(), indent=2, cls=PydanticEncoder)

    def __repr__(self) -> str:
        """String representation for debugging"""
        #return f"\"DebugInfo(prompt_difficulties={self.prompt_difficulties}, prompt_improvements={self.prompt_improvements})\""
        return json.dumps(self.model_dump(), indent=2, cls=PydanticEncoder)
    
    
class ArchiveInfo(BaseModel):
    """Model for storing archive information about the API call"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    model: str
    system_prompt_template: str
    system_prompt_inputs: Dict[str, str]
    user_prompt_template: str
    user_prompt_inputs: Dict[str, str]
    response_model: str
    tools: Optional[List[dict]]
    tag: Optional[str]
    time_taken: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    response: Any
    debug_info: Optional[DebugInfo]
    messages: Optional[List[dict]] = None 