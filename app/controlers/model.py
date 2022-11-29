from fastapi import APIRouter

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
async def get_IA_model_description():
    """
        Get predicting model description.


    """

@router.put("/")
async def add_data_to_dataframe(status_code=201):
    """
        Add a new line to dataframe.
    """
    pass
@router.post("/retrain")
async def retrain_model():
    """
        Retrain predicting model
    """
    pass