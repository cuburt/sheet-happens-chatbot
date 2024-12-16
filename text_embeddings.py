from typing import List
from langchain.schema.embeddings import Embeddings
from torch import Tensor
import logging
from pathlib import Path
from encoders import BAAILLMEmbedder, AllMiniLML6V2, TextEmbeddingGecko, MultilingualEmbeddingGecko, TextMultimodalEmbeddingGecko
PROJECT_ROOT_DIR = str(Path(__file__).parent) #set project root directory


class TextEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = eval(f"{model_name}()")

    def embed_query(self, text: str) -> List[float]:
        try:
            if self.model_name in ['BAAILLMEmbedder']:
                embeddings = self.model.encode(text, normalize_embeddings=True)
            else:
                embeddings = self.model.encode(text)
            return embeddings.tolist()
        except Exception as e:
            logging.error(str(e))

    def embed_documents(self, texts: List[str]) -> List[List[Tensor]]:
        try:
            if self.model_name in ['BAAILLMEmbedder']:
                embeddings = [self.model.encode(text, normalize_embeddings=True) for text in texts]
            else:
                embeddings = [self.model.encode(text) for text in texts]
            return embeddings
        except Exception as e:
            logging.error(str(e))
