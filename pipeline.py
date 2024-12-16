from pathlib import Path
from chain import Chain
import threading
import os
from langchain_community.document_loaders import PyPDFLoader
# Directory to save uploaded files
PROJECT_ROOT_DIR = str(Path(__file__).parent)
UPLOAD_FOLDER = "data/uploads"
os.makedirs(os.path.join(PROJECT_ROOT_DIR, UPLOAD_FOLDER), exist_ok=True)

class ModelPipeline: 
    def __init__(self):
        self.chunklen = 0
        # moved config here to isolate references in the same file and reduce coupling
        self.chain = None
        self.ready = False
        self.ainitialize()

    def ainitialize(self):
        # Avoids TCP probe failure by immediate initialization of server with "unready" endpoints.
        initialize_thread = threading.Thread(target=self.initialize, daemon=True, name="initialize")
        initialize_thread.start()

    def initialize(self):
        self.chain = Chain()
        self.ready = True

    def upload_data(self, files):
        """Model Pipeline: Load Data, Preprocess Data, Form Causality Graph, Model Probability"""
        try:
            if not self.ready:
                raise Exception("Pipeline not ready. Please wait...")
            pages = []
            attached_text = ""
            for file in files:
                file_path = os.path.join(PROJECT_ROOT_DIR, UPLOAD_FOLDER, file.name)
                print("FILEPATH: ", file_path)
                with open(file_path, "wb") as f:
                    f.write(file.read())
                # Read PDF content into PyPDFLoader
                loader = PyPDFLoader(file_path)
                pages = loader.load_and_split()
            if pages:
                attached_text = ''.join([page.page_content for page in pages])
                self.chain.attached_text = attached_text
            res = self.chain.vectorstore.insert_docs(pages)
            return {"ids": res}
        except Exception as e:
            raise Exception(str(e))

    def generate_prediction(self, model, query, enable_rag=True, session_id=None):
        try:
            if not self.ready:
                raise Exception("Endpoint not ready. Please wait...")
            message_id, llm_response, source_documents = self.chain(query=query, llm=model, enable_rag=enable_rag, session_id=session_id)
            response = {"prediction": {"id": message_id, "answer": llm_response, "source_documents": source_documents}}
            return response
        except Exception as e:
            raise Exception(str(e))
