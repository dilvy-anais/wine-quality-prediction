from fastapi import APIRouter,HTTPException
from app.models.wine import Wine
from app.predictions.train_models import *
from app.models.model import Model

router = APIRouter(
    prefix="/model",
    tags=["AI management"]
)

@router.get("/")
async def get_serialize_model():
    """
    Allowing to obtain file with .h5 extension containing the latest version of predicting model.

    """
    pass

@router.get("/description")
async def get_IA_model_description()->Model:
    """
        Get predicting model description.
        Args :
            N/A
        Return :
            Models contains AI model parameter
    """
    estimation_number,random_state,test_size, random_state_train = parametre_model()
    return Model(name="Random Forest Model", estimate_number=estimation_number, random_state=random_state, test_size=test_size, random_state_train=random_state_train)

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
        ajout_vin_dataFrame(wine.volatile_acidity,wine.citric_acidity,wine.residual_sugar,wine.chlorides,wine.free_sulfur_dioxide,wine.total_sulfur_dioxide,wine.density,wine.ph,wine.sulphates, wine.alcohol,data_wine)
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
    data = analyse_model(data_wine)
    test, predict, model = entrainement_Random_Forest(data)
    enregistrer_model(model)
    accuracy = metrique_model(test, predict)
    return(accuracy)
