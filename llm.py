import os
import time
import json
import platform
import requests
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
            access_token = os.getenv("GEMINI_APIKEY")
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