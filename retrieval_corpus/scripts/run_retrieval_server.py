import os
import logging
from typing import List, Union, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from flashrag.config import Config
from flashrag.utils import get_retriever
from flashrag.retriever import BaseTextRetriever


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {
            "level": "INFO",
            "handlers": ["console"],
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
    },
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Document(BaseModel):
    id: str
    contents: str

class SingleRetrieveResponseWithScore(BaseModel):
    documents: List[Document]
    scores: List[float]

class SingleRetrieveResponseNoScore(BaseModel):
    documents: List[Document]

class RetrieveRequest(BaseModel):
    query: Union[str, List[str]]
    tok_k: int = 3
    return_score: bool = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API Service Starting...")
    try:
        cur_file_path = os.path.dirname(os.path.realpath(__file__))
        setting_config_file = os.path.join(cur_file_path, "coa_config.yaml")
        use_config = Config(config_file_path=setting_config_file)
        
        app.state.retriever = get_retriever(config=use_config)
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {str(e)}")
        raise
    yield
    logger.info("API Service Shutting Down...")

app = FastAPI(title="Document Retrieval API", description="API for retrieving documents based on queries.", lifespan=lifespan)

@app.post("/retrieve", 
          response_model=Union[SingleRetrieveResponseWithScore, SingleRetrieveResponseNoScore, 
                              List[SingleRetrieveResponseWithScore], List[SingleRetrieveResponseNoScore]])
async def retrieve_docs_endpoint(request: RetrieveRequest):
    try:
        query = request.query
        tok_k = request.tok_k
        return_score = request.return_score
        retriever = app.state.retriever

        if isinstance(query, str):
            retrieved_result = retriever.search(query=query, num=tok_k, return_score=return_score)
            if return_score:
                documents, scores = retrieved_result
                return SingleRetrieveResponseWithScore(
                    documents=[Document(**doc) for doc in documents],
                    scores=scores
                )
            else:
                documents = retrieved_result
                return SingleRetrieveResponseNoScore(
                    documents=[Document(**doc) for doc in documents]
                )
        elif isinstance(query, list):
            retrieved_results = retriever.batch_search(query=query, num=tok_k, return_score=return_score)
            if return_score:
                docs_list, scores_list = retrieved_results
                return [
                    SingleRetrieveResponseWithScore(
                        documents=[Document(**doc) for doc in docs],
                        scores=scores
                    )
                    for docs, scores in zip(docs_list, scores_list)
                ]
            else:
                docs_list = retrieved_results
                return [
                    SingleRetrieveResponseNoScore(
                        documents=[Document(**doc) for doc in docs]
                    )
                    for docs in docs_list
                ]
        else:
            raise ValueError("Query must be a string or a list of strings.")
    except ValueError as ve:
        logger.error(f"ValueError occurred: {str(ve)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tests.run_retrieve_service:app", host="0.0.0.0", port=8000, log_config=LOGGING_CONFIG)