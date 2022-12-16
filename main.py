from fastapi import FastAPI
from app.controlers import model, predict

# Solution :  https://fastapi.tiangolo.com/advanced/sub-applications/
app = FastAPI()
subapi = FastAPI()

#root_path="/api"
subapi.include_router(model.router)
subapi.include_router(predict.router)

app.mount("/api", subapi)

