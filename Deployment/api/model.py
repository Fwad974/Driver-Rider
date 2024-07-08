from datetime import datetime
from pydantic import BaseModel


class Input(BaseModel):
    dest: int
    origin: int
    Time: int
    Income: int
    Comment: str
    Created_at: datetime