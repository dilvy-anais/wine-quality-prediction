from fastapi import FastAPI
from app.controlers import model, predict

# Solution :  https://fastapi.tiangolo.com/advanced/sub-applications/
app = FastAPI()

#root_path="/api"
app.include_router(model.router)
app.include_router(predict.router)

