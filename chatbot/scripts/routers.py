import os
import logging
import platform
from pathlib import Path
from typing import List, Optional, Any
import pandas as pd
from semantic_router import Route
from semantic_router.encoders import BaseEncoder, HuggingFaceEncoder
from semantic_router.layer import RouteLayer
from encoders import TextEmbeddingGecko, BAAICrossEncoder, BAAILLMEmbedder
PROJECT_ROOT_DIR = str(Path(__file__).parent.parent.parent)  # set project root directory


def read_api_key(service_provider):
    """
    Options: "openai", "nvidia", "cohere"
    """
    api_key_path = f"scripts/server/.{service_provider}_api_key.txt"
    if platform.system() == 'Windows':
        api_key_path = api_key_path.split('/')
        api_key_path = os.path.join(PROJECT_ROOT_DIR, *api_key_path)
    else:
        api_key_path = os.path.join(PROJECT_ROOT_DIR, api_key_path)
    with open(api_key_path, 'r') as f:
        access_token = f.read().rstrip()
    if not access_token:
        return f"Error 401: No {service_provider} API-KEY saved."

    return access_token


class TextEmbeddingGeckoRouter(BaseEncoder):
    client: Optional[Any] = None
    type: str = "google"

    def __init__(
        self,
        name: Optional[str] = type,
        score_threshold: float = 0.75
    ):
        super().__init__(name=name, score_threshold=score_threshold)
        self.client = TextEmbeddingGecko()

    def __call__(self, docs: List[str]) -> List[List[float]]:
        predictions = self.client.encode(docs=docs)
        return [prediction["embeddings"]["values"] for prediction in predictions]


class MsMarcoMiniLML6V2Router(HuggingFaceEncoder):
    name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    score_threshold: float = 0.95


class BAAICrossEncoderRouter(BaseEncoder):
    client: Optional[Any] = None
    type: str = "baaicrossencoder"

    def __init__(
            self,
            name: Optional[str] = type,
            score_threshold: float = 0.75,
    ):
        super().__init__(name=name, score_threshold=score_threshold)
        self.client = BAAICrossEncoder()

    def __call__(self, docs: List[str]) -> List[List[float]]:
        predictions = [self.client.encode(doc) for doc in docs]
        return predictions


class BAAILLMEmbedderRouter(BaseEncoder):
    client: Optional[Any] = None
    type: str = "baaillmembedder"

    def __init__(
        self,
        name: Optional[str] = type,
        score_threshold: float = 0.75,
    ):
        super().__init__(name=name, score_threshold=score_threshold)
        self.client = BAAILLMEmbedder()

    def __call__(self, docs: List[str]) -> List[List[float]]:
        predictions = [self.client.encode(doc) for doc in docs]
        return predictions


class SemanticRouter:
    def __init__(self):
        try:
            valid_requests_sheet = "data/valid_requests.csv"
            try:
                valid_df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, *valid_requests_sheet.split('/')))
                valid_requests = Route(
                    name="valid",
                    utterances=[str(v) for v in valid_df["query"].values])
            except:
                valid_requests = Route(
                    name="valid",
                    utterances=["hey"])
            routes = [valid_requests]

            encoder = BAAILLMEmbedderRouter()
            self.routelayer = RouteLayer(encoder=encoder, routes=routes)
            logging.info("Router successfully initialised.")
        except Exception as e:
            raise Exception(str(e))

    def __call__(self, prompt):
        return self.routelayer(prompt).name
