from fastapi import APIRouter,HTTPException
from app.models.wine import Wine
from app.predictions.train_models import *
router = APIRouter(
    prefix="/predict",
    tags=["prediction"]
)

@router.post("/")
async def predict_quality_score(wine: Wine = None)->int:
    """
    This function call predicting model to determine the quality score thanks to wine characteristic
    Args:
        wine (Wine): instance of wine BaseModel
    Returns:
        int: quality score
    """
    try :
        result = info_vin_a_predire(wine.volatile_acidity,wine.chlorides,wine.free_sulfur_dioxide,wine.total_sulfur_dioxide,wine.density,wine.ph,wine.sulphates,wine.alcohol)
        return int(result[0])
    except :
        raise HTTPException(status_code=500, detail="Could not calculate quality score")


@router.get("/")
async def get_perfect_wine() -> Wine:
    """
    This function call predicting model to find ideal wine

    Return:
         Wine: Wine with ideals characteristics
    """
    # Replace by prediction Emmy function
