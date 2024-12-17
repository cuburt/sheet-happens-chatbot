from typing import Dict, List
import json
import subprocess
import platform
import requests
from pathlib import Path
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from log import logger
PROJECT_ROOT_DIR = str(Path(__file__).parent) #set project root directory


class EmbedRequest:
    def __call__(self, model_id:str, data:Dict):
        try:
            region = "us-central1"
            api_endpoint = f"{region}-aiplatform.googleapis.com"
            project_id = "hclsw-gcp-xai"
            assert model_id in ["textembedding-gecko@003", "textembedding-gecko-multilingual@001", "multimodalembedding@001", "text-embedding-004"], f"Sorry {model_id} is not currently supported."
            script_file_path = os.path.join(PROJECT_ROOT_DIR, "scripts/server", "access_token.sh")
            cmd = ['/bin/bash', script_file_path]
            if platform.system() == 'Windows':
                cmd = ['C:/Program Files/Git/bin/bash.exe', script_file_path]
            access_token = subprocess.run(cmd, capture_output=True, text=True)
            assert access_token.stdout, access_token.stderr
            headers = {"Authorization": f"Bearer {access_token.stdout}", "Content-Type": "application/json"}
            url = f"https://{api_endpoint}/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model_id}:predict"
            return requests.post(url=url, headers=headers, data=json.dumps(data)).json()
        except Exception as e:
            logger.error(str(e))


class MsMarcoMiniLML6V2(CrossEncoder):
    def __init__(self):
        super().__init__('cross-encoder/ms-marco-MiniLM-L-6-v2')


class AllMiniLML6V2(SentenceTransformer):
    def __init__(self):
        super().__init__('sentence-transformers/all-MiniLM-L6-v2')


# class BAAILLMEmbedder(SentenceTransformer):
#     def __init__(self):
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         logger.info(f"Running on {device}")
#         model_path = os.path.join(PROJECT_ROOT_DIR, 'models', 'BAAI', 'llm-embedder')
#         if not os.path.exists(os.path.join(model_path, "config.json")):
#             model_path = "BAAI/llm-embedder"
#         super().__init__(model_path, device=device)
#
#     def predict(self, pairs):
#         query_embeddings = self.encode([p[0] for p in pairs], normalize_embeddings=True)
#         doc_embeddings = self.encode([p[1] for p in pairs], normalize_embeddings=True)
#         scores = []
#         for q_emb, d_emb in zip(query_embeddings, doc_embeddings):
#             similarity = cosine_similarity([q_emb], [d_emb])[0][0] * 100
#             scores.append(similarity)
#         return scores
#
#
# class BAAICrossEncoder(SentenceTransformer):
#     def __init__(self):
#         super().__init__("BAAI/bge-multilingual-gemma2", model_kwargs={"torch_dtype": torch.float16})
#
#     def predict(self, pairs):
#         instruction = 'Given a web search query, retrieve relevant passages that answer the query.'
#         prompt = f'<instruct>{instruction}\n<query>'
#         # Compute the query and document embeddings
#         query_embeddings = self.encode([p[0] for p in pairs], prompt=prompt)
#         doc_embeddings = self.encode([p[1] for p in pairs])
#         # Compute the cosine similarity between the query and document embeddings
#         similarities = self.similarity(query_embeddings, doc_embeddings)
#         return similarities.tolist()


class TextEmbeddingGecko:
    @staticmethod
    def encode(docs: List[str], prompt:str=None):
        try:
            data = {}
            embed_request = EmbedRequest()
            if docs:
                data = {
                    "instances": [{
                        "content": doc,
                        "task_type": "SEMANTIC_SIMILARITY"
                    } for doc in docs]
                }
                res = embed_request(model_id="textembedding-gecko@003", data=data)
                response = res['predictions']
            else:
                data = {
                    "instances": [{
                        "content": prompt,
                        "task_type": "SEMANTIC_SIMILARITY"
                    }]
                }
                res = embed_request(model_id="textembedding-gecko@003", data=data)
                response = res['predictions'][0]['embeddings']['values']
            assert response, "No Response from Google Embeddings"
            return response
        except Exception as e:
            logger.error(str(e))

    def predict(self, pairs):
        queries, docs = [p[0] for p in pairs], [p[1] for p in pairs]
        query_embeddings = self.encode(queries)
        doc_embeddings = self.encode(docs)
        return list((query_embeddings @ doc_embeddings.T) * 100)


class MultilingualEmbeddingGecko:
    @staticmethod
    def encode(docs: List[str], prompt:str=None):
        try:
            data = {}
            embed_request = EmbedRequest()
            if docs:
                data = {
                    "instances": [{
                        "content": doc,
                        "task_type": "SEMANTIC_SIMILARITY"
                    } for doc in docs]
                }
                res = embed_request(model_id="textembedding-gecko-multilingual@001", data=data)
                response = res['predictions']
            else:
                data = {
                    "instances": [{
                        "content": prompt,
                        "task_type": "SEMANTIC_SIMILARITY"
                    }]
                }
                res = embed_request(model_id="textembedding-gecko-multilingual@001", data=data)
                response = res['predictions'][0]['embeddings']['values']
            assert response, "No Response from Google Embeddings"
            return response
        except Exception as e:
            logger.error(str(e))

    def predict(self, pairs):
        queries, docs = [p[0] for p in pairs], [p[1] for p in pairs]
        query_embeddings = self.encode(queries)
        doc_embeddings = self.encode(docs)
        return list((query_embeddings @ doc_embeddings.T) * 100)


class TextMultimodalEmbeddingGecko:
    @staticmethod
    def encode(prompt, base64_encoded_img):
        try:
            data = {
                "instances": [{
                    "text": prompt,
                    "image": {"bytesBase64Encoded": base64_encoded_img}
                }]
            }
            embed_request = EmbedRequest()
            res = embed_request(model_id="multimodalembedding@001", data=data)
            response = res['predictions'][0]['textEmbedding'], res['predictions'][0]['textEmbedding']
            assert response, "No Response from Google Embeddings"
            return response
        except Exception as e:
            logger.error(str(e))
