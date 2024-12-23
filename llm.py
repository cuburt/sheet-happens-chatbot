import os
import time
import json
import requests
from gguf_llm import models
import mlx.core as mx
from typing import Any, List, Mapping, Optional, Dict
from pathlib import Path
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from log import logger
PROJECT_ROOT_DIR = str(Path(__file__).parent)  # set project root directory



class GoogleRequest:
    def __call__(self, model_id: str, data: Dict):
        try:
            google_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemma-2-27b-it"]
            if model_id not in google_models:
                return f"Sorry {model_id} is not currently supported."
            access_token = os.getenv("GEMINI_APIKEY") if "GEMINI_APIKEY" in os.environ else None
            if not access_token:
                return "Error 401: No Google API-KEY saved."
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={access_token}"
            headers = {"Content-Type": "application/json"}
            response = requests.post(url=url, headers=headers, data=json.dumps(data)).json()
            return response

        except Exception as e:
            logger.error(str(e))


class GeminiLLM(LLM):
    temperature: float = 1
    max_output_tokens: int = 2048
    top_p: float = 1

    @property
    def _llm_type(self) -> str:
        return "gemini-1.5-pro"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")
        try:
            data = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": self.temperature, "maxOutputTokens": self.max_output_tokens,
                                     "topP": self.top_p}
            }
            llm_request = GoogleRequest()
            start = time.time()
            res = [llm_request(model_id=self._llm_type, data=data)]
            logger.info(f"LLM RT: {(time.time() - start):.4f}")
            if len(res) >= 1 and 'error' in res and 'code' in res['error'] and 'message' in res[0]['error']:
                return f"Error {res[0]['error']['code']}: {res[0]['error']['message']}"
            elif len(res) >= 1 and 'candidates' in res[0]:
                return ''.join([r['candidates'][0]['content']['parts'][0]['text'] for r in res if
                                len(r['candidates']) > 0 and "content" in r['candidates'][0] and len(
                                    r['candidates'][0]['content']['parts']) > 0 and "text" in
                                r['candidates'][0]['content']['parts'][0]])
            else:
                return str(res)

        except Exception as e:
            logger.error(str(e))


class GeminiFlashLLM(LLM):
    temperature: float = 1
    max_output_tokens: int = 2048
    top_p: float = 1

    @property
    def _llm_type(self) -> str:
        return "gemini-1.5-flash"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")
        try:
            data = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": self.temperature, "maxOutputTokens": self.max_output_tokens, "topP": self.top_p}
            }
            llm_request = GoogleRequest()
            start = time.time()
            res = [llm_request(model_id=self._llm_type, data=data)]
            logger.info(f"LLM RT: {(time.time() - start):.4f}")
            if len(res) >= 1 and 'error' in res and 'code' in res['error'] and 'message' in res[0]['error']:
                return f"Error {res[0]['error']['code']}: {res[0]['error']['message']}"
            elif len(res) >= 1 and 'candidates' in res[0]:
                return ''.join([r['candidates'][0]['content']['parts'][0]['text'] for r in res if len(r['candidates']) > 0 and "content" in r['candidates'][0] and len(r['candidates'][0]['content']['parts']) > 0 and "text" in r['candidates'][0]['content']['parts'][0]])
            else:
                return str(res)

        except Exception as e:
            logger.error(str(e))


class Gemma2LLM(LLM):
    temperature: float = 1
    max_output_tokens: int = 2048
    top_p: float = 1

    @property
    def _llm_type(self) -> str:
        return "gemma-2-27b-it"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")
        try:
            data = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": self.temperature, "maxOutputTokens": self.max_output_tokens, "topP": self.top_p}
            }
            llm_request = GoogleRequest()
            start = time.time()
            res = [llm_request(model_id=self._llm_type, data=data)]
            logger.info(f"LLM RT: {(time.time() - start):.4f}")
            if len(res) >= 1 and 'error' in res and 'code' in res['error'] and 'message' in res[0]['error']:
                return f"Error {res[0]['error']['code']}: {res[0]['error']['message']}"
            elif len(res) >= 1 and 'candidates' in res[0]:
                return ''.join([r['candidates'][0]['content']['parts'][0]['text'] for r in res if len(r['candidates']) > 0 and "content" in r['candidates'][0] and len(r['candidates'][0]['content']['parts']) > 0 and "text" in r['candidates'][0]['content']['parts'][0]])
            else:
                return str(res)

        except Exception as e:
            logger.error(str(e))


class TinyLlama:
    def __init__(self):
        mx.random.seed(1)
        self.model, self.tokenizer = models.load("tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
                                                 "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")

    def generate(self, prompt: str, max_tokens: int, temp: float = 0.0):
        # Encode the prompt
        prompt_encoded = self.tokenizer.encode(prompt)

        # Initialization
        tic = time.time()
        tokens = []
        skip = 0
        for token, n in zip(
                models.generate(prompt_encoded, self.model, temp),
                range(max_tokens),
        ):
            if token == self.tokenizer.eos_token_id:
                break

            if n == 0:
                prompt_time = time.time() - tic
                tic = time.time()

            tokens.append(token.item())
            # s = tokenizer.decode(tokens)
            # print(s[skip:], end="", flush=True)
            # skip = len(s)

        print(self.tokenizer.decode(tokens)[skip:], flush=True)
        # gen_time = time.time() - tic
        # print("=" * 10)

        if len(tokens) == 0:
            print("No tokens generated for this prompt")

        # prompt_tps = prompt_encoded.size / prompt_time
        # gen_tps = (len(tokens) - 1) / gen_time
        # print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
        # print(f"Generation: {gen_tps:.3f} tokens-per-sec")
        return self.tokenizer.decode(tokens)[skip:]


class TinyLlamaLLM(LLM):
    model = TinyLlama()
    temperature: float = 0.3
    max_output_tokens: int = 256
    top_p: float = 1

    @property
    def _llm_type(self) -> str:
        return "tinyllama"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")
        outputs = self.model.generate(prompt=str(prompt), max_tokens=self.max_output_tokens, temp=self.temperature)

        return outputs