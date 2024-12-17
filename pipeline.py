import os
import threading
from pathlib import Path
from typing import List, Iterable
import nltk
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.schema.document import Document
from langchain.text_splitter import NLTKTextSplitter
from chain import Chain

PROJECT_ROOT_DIR = str(Path(__file__).parent)
# Directory to save uploaded files
UPLOAD_FOLDER = "data/uploads"
os.makedirs(os.path.join(PROJECT_ROOT_DIR, UPLOAD_FOLDER), exist_ok=True)


class ModelPipeline: 
    def __init__(self):
        self.chunk_overlap = 5
        self.chunk_size = 512
        self.chain = None
        self.ready = False
        self.ainitialize()
        self.aload_pdf([str(file) for file in Path(os.path.join(PROJECT_ROOT_DIR, UPLOAD_FOLDER)).rglob("*.pdf")])

    def ainitialize(self):
        # Avoids TCP probe failure by immediate initialization of server with "unready" endpoints.
        initialize_thread = threading.Thread(target=self.initialize, daemon=True, name="initialize")
        initialize_thread.start()

    def initialize(self):
        nltk.download('punkt')
        nltk.download('punkt_tab')
        self.chain = Chain()
        self.ready = True

    def chunk_data(self, docs: Iterable[Document]) -> List[Document]:
        text_splitter = NLTKTextSplitter(chunk_size=self.chunk_size,chunk_overlap=self.chunk_overlap)
        docs = text_splitter.split_documents(docs)
        return docs

    def load_pdf(self, files):
        docs = []
        for file in files:
            if isinstance(file, str):
                file_path = file
            else:
                file_path = os.path.join(PROJECT_ROOT_DIR, UPLOAD_FOLDER, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.read())
            # Read PDF content into PyPDFLoader
            print("PDF filepath: ", file_path)
            loader = UnstructuredPDFLoader(file_path)
            pages = loader.load_and_split()
            if isinstance(pages, list):
                docs.extend(pages)
            else:
                docs.append(pages)
        self.chain.vectorstore.insert_docs(self.chunk_data(docs), insert_batch_size=5)

    def aload_pdf(self, files):
        load_pdf_thread = threading.Thread(target=self.load_pdf, daemon=True, name="load_pdf", args=[files])
        load_pdf_thread.start()

    def process_document(self, files):
        """Model Pipeline: Load Data, Preprocess Data, Form Causality Graph, Model Probability"""
        try:
            if not self.ready:
                raise Exception("Pipeline not ready. Please wait...")
            self.load_pdf(files)
        except Exception as e:
            raise Exception("pipeline.py @ upload_data() " + str(e))

    def generate_prediction(self, model, params, query, enable_rag=True, session_id=None):
        try:
            if not self.ready:
                raise Exception("Endpoint not ready. Please wait...")
            message_id, llm_response, source_documents = self.chain(query=query, params=params, llm=model, enable_rag=enable_rag, session_id=session_id)
            response = {"prediction": {"id": message_id, "answer": llm_response, "source_documents": source_documents}}
            return response
        except Exception as e:
            raise Exception("pipeline.py @ generate_prediction() " + str(e))
