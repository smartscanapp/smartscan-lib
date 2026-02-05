from typing import Optional
from pydantic import Field, BaseModel

class ClassificationResult(BaseModel):
    item_id: str
    label: Optional[str] = Field(default=None)

