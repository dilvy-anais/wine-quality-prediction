from typing import Optional

from pydantic import BaseModel,Field

class WineInput(BaseModel):
    fixed_acidity: float = Field(0.0, gt=0.0, description="Fixed acidity must not be a negative value")
    volatile_acidity: float = Field(0.0, gt=0.0, description="Volatile acidity must not be a negative value")
    citric_acidity: float = Field(0.0, gt=0.0, description="Citric acidity must be greater than 0.")
    residual_sugar: float = Field(0.0, gt=0.0, description="Residual sugar must be greater than 0.")
    chlorides: float = Field(0.0, gt=0.0, description="Chlorides must not be a negative value.")
    free_sulfur_dioxide: int = Field(0, gt=0, description="Free sulfur dioxide  must be greater than 0.")
    total_sulfur_dioxide: int = Field(0, gt=0, description="Total sulfur dioxide be a negative value.")
    density: float = Field(0.0, gt=0.0, description="Density must be greater than 0.")
    ph: float = Field(0.0, gt=0.0, lt=14.0, description="pH must be between 0 and 14.")
    sulphates: float = Field(0.0, gt=0.0, description="Sulphates must not be a negative value")
    alcohol:float =Field(0.0,gt=0.0,description="Alcohol must not be a negative value")

class Wine(WineInput):
    quality:int= Field(0,gt=0,lt=11,description="Quality score must be between 0 and 10")