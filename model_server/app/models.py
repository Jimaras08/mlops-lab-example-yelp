from typing import Dict

from pydantic import BaseModel


class ModelInput(BaseModel):
    text: str
    metadata: Dict = {}
