from typing import Optional
from pydantic import BaseModel

class Model(BaseModel):
    name: str
    estimate_number:Optional[int]
    random_state: Optional[int]
    random_state_train: Optional[int]
    test_size:Optional[float]