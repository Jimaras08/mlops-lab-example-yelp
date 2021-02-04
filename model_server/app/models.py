from pydantic import BaseModel


class ModelInput(BaseModel):
    text: str
