from fastapi import APIRouter,HTTPException
from app.models.wine import WineInput,Wine
from app.predictions.prediction_model import model

router = APIRouter(
    prefix="/predict",
    tags=["prediction"]
)
@router.post("/")
async def predict_quality_score(wine: WineInput = None)->int:
    """
    This function call predicting model to determine the quality score thanks to wine characteristic
    Args:
        wine (Wine): instance of wine BaseModel
    Returns:
        int: quality score
    """
    try :
        result = model.info_wine_to_predict(wine.volatile_acidity,wine.chlorides,wine.free_sulfur_dioxide,wine.total_sulfur_dioxide,wine.density,wine.ph,wine.sulphates,wine.alcohol)
        return result
    except Exception as e :
        print(e)
        raise HTTPException(status_code=500, detail="Could not calculate quality score")


@router.get("/")
async def get_perfect_wine() -> WineInput:
    """
    This function call predicting model to find ideal wine

    Return:
         Wine: Wine with ideals characteristics
    """

    return model.wine_perfect()
