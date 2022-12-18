from fastapi import APIRouter,HTTPException
from app.models.wine import Wine
from app.predictions.prediction_model import model
from app.models.model import Model
from fastapi.responses import FileResponse


router = APIRouter(
    prefix="/model",
    tags=["AI management"]
)

@router.get("/")
async def get_serialize_model():
    """
    Allowing to obtain file with .h5 extension containing the latest version of predicting model.

    """
    try:
        return FileResponse(path="app/data/random_forest.joblib",filename="random_forest.joblib",media_type="application/octet-stream")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail="Serialize model not found.")



@router.get("/description")
async def get_AI_model_description()->Model:
    """
        Get predicting model description.
        Args :
            N/A
        Return :
            Models contains AI model parameter
    """

    return Model(name="Random Forest Model",test_sample_size=model.test_sample,train_sample_size=model.train_sample,accuracy= model.accuracy)

@router.put("/", status_code=201)
async def add_data_to_dataframe(wine: Wine):
    """
        Add a new line to dataframe.
        Args:
            wine: wine model contains all information to add to dataframe
        Returns:
            N/A
    """
    try:
        model.add_wine_dataFrame(wine.fixed_acidity,wine.volatile_acidity,wine.citric_acidity,wine.residual_sugar,wine.chlorides,wine.free_sulfur_dioxide,wine.total_sulfur_dioxide,wine.density,wine.ph,wine.sulphates, wine.alcohol, wine.quality)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Could not add this wine to the dataframe.")

@router.post("/retrain")
async def retrain_model()->float:
    """
        Retrain predicting model
        Args:
            N/A
        Returns:
            Accuracy rate after training model

    """
    data = model.analyse_model()
    model.train_Random_Forest(data)
    return(model.accuracy)
