import os
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from dotenv import load_dotenv
from api import router
from app import app, logger
import xgboost as xgb
from transformers import AutoTokenizer, AutoModel

import numpy as np


__author__ = "Fwad abdi"

app.include_router(router)


@app.on_event("startup")
def startup() -> None:
    load_dotenv()
    try:
        app.thread_pool = ThreadPoolExecutor()
    except Exception as error:
        raise Exception(error)
    try:
        app.model=  xgb.Booster()
        app.model.load_model('final_xgboost_model.json')
    except Exception as error:
        raise Exception(error)
    try:
        app.tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
        app.bert = AutoModel.from_pretrained("HooshvareLab/bert-fa-base-uncased")

    except Exception as error:
        raise Exception(error)

    logger.info("App started successfully")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
