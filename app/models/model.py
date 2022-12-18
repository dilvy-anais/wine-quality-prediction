from typing import Optional
from pydantic import BaseModel

class Model(BaseModel):
    name: str
    test_sample_size:float
    train_sample_size:float
    accuracy:Optional[float]
