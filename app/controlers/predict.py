from fastapi import APIRouter
from app.models.wine import Wine

router = APIRouter(
    prefix="/predict",
    tags=["prediction"]
)

@router.post("/")
async def predict_quality_score(wine: Wine = None)->dict:
    """
    This function call predicting model to determine the quality score thanks to wine characteristic
    Args:
        wine (Wine): instance of wine BaseModel
    Returns:
        dict: dictionary contains quality score
    """

    quality_score = 8.7 # Replace by prediction Emmy function
    return {"quality_score": quality_score}

@router.get("/")
async def get_perfect_wine() -> Wine:
    """
    This function call predicting model to find ideal wine

    Return:
         Wine: Wine with ideals characteristics
    """
    # Replace by prediction Emmy function
    return {"quality score": 10}