import os
import logging
import threading
import platform
from typing import List, Dict
from pathlib import Path
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings import FakeEmbeddings
from langchain.docstore.in_memory import InMemoryDocstore
import faiss
from text_embeddings import TextEmbeddings


def get_most_recently_updated_file(folders):
    most_recent_file = None
    most_recent_time = None

    for folder in folders:
        # Get a list of all files in the folder with their full paths
        files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

        for file in files:
            file_mtime = os.path.getmtime(file)
            if most_recent_time is None or file_mtime > most_recent_time:
                most_recent_time = file_mtime
                most_recent_file = file

    return Path(most_recent_file).parent


def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


PROJECT_ROOT_DIR = str(Path(__file__).parent)


def is_file_too_big(file_path, max):
    if platform.system() == 'Windows':
        dir_path = file_path.split('/')
        dir_path = os.path.join(PROJECT_ROOT_DIR, *dir_path)
    else:
        dir_path = os.path.join(PROJECT_ROOT_DIR, file_path)
    size = sum([os.path.getsize(file) for file in os.scandir(dir_path)])
    logging.info(f"{file_path}'s size in byte is {size}")
    return size >= max


class FAISSstore:
    def __init__(self, docs: Dict[str, List[Document]]=None, db_dir: str=None, db_file: str=None, model: str="AllMiniLML6V2", max_filesize_bytes:float=0.0, insert_batch_size:int=100):
        logging.info(f"Initialising {db_file} vectorstore...")
        vectorstore_dir = os.path.join(PROJECT_ROOT_DIR, db_dir)
        if not os.path.exists(vectorstore_dir):
            os.mkdir(vectorstore_dir)

        self.local_dir = os.path.join(PROJECT_ROOT_DIR, db_dir, f"{db_file}-{model}")
        self.vectorstore: FAISS
        if docs and "documents" in docs and not docs["documents"] and db_dir and db_file:
            try:
                if not os.path.exists(self.local_dir):
                    # vectorstore directory
                    vectorestore_folder = os.path.join(str(Path(__file__).parent.parent.parent), db_dir)
                    candidate_vectorstores = [os.path.join(vectorestore_folder, dir) for dir in os.listdir(vectorestore_folder) if db_file in dir]
                    assert candidate_vectorstores, f"No existing vectorstore for {db_file}."
                    self.local_dir = get_most_recently_updated_file(candidate_vectorstores)
                    assert self.local_dir, f"Corrupt vectorstore for {db_file}."
                    self.embed_model = TextEmbeddings(os.path.basename(os.path.normpath(self.local_dir)).strip(f"{db_file}-"))
                else:
                    self.embed_model = TextEmbeddings(model)
            except AssertionError as e:
                logging.error(str(e))

            try:
                self.vectorstore = FAISS.load_local(folder_path=self.local_dir, embeddings=self.embed_model, allow_dangerous_deserialization=True)
            except Exception as e:
                logging.warning(str(e))
                self.vectorstore = FAISS.load_local(folder_path=self.local_dir, embeddings=self.embed_model)
            logging.info("Vectorstore successfully loaded.")

        elif docs and "documents" in docs and docs["documents"] and not is_file_too_big(docs["filepath"], max_filesize_bytes):
            self.embed_model = TextEmbeddings(model)
            self.vectorstore = FAISS.from_documents(docs["documents"], self.embed_model)
            if not os.path.exists(self.local_dir):
                os.mkdir(self.local_dir)
            self.vectorstore.save_local(folder_path=self.local_dir)
            logging.info(f"VECTORSTORE saved at: {self.local_dir}\nFileSize: {convert_bytes(sum([os.path.getsize(file) for file in os.scandir(self.local_dir)]))}")

        elif docs and "documents" in docs and docs["documents"] and is_file_too_big(docs["filepath"], max_filesize_bytes):
            self.embed_model = TextEmbeddings(model)
            logging.info("Asynchronous vectorstore builder initialised.")
            self.vectorstore = FAISS.from_documents(docs["documents"][:insert_batch_size], self.embed_model)
            if not os.path.exists(self.local_dir):
                os.mkdir(self.local_dir)
            self.vectorstore.save_local(folder_path=self.local_dir)
            logging.info(f"Initial vectorstore saved at: {self.local_dir}\nFileSize: {convert_bytes(sum([os.path.getsize(file) for file in os.scandir(self.local_dir)]))}\nAsynchronous update begins...")
            # Allows parallel inference and vectorstore build.
            insert_docs_thread = threading.Thread(target=self.insert_docs, daemon=True, name="insert_docs", args=[docs["documents"][insert_batch_size:], insert_batch_size])
            insert_docs_thread.start()

        else:
            # Initialize FAISS index with the dimension of the embeddings
            embedding_dim = 384  # Replace with your embedding dimension
            index = faiss.IndexFlatL2(embedding_dim)

            # Create an in-memory docstore
            docstore = InMemoryDocstore({})

            # Map to track the association between FAISS vectors and document IDs
            index_to_docstore_id = {}

            # Create a dummy embedding function
            embedding_model = FakeEmbeddings(size=embedding_dim)

            # Initialize the FAISS vector store
            self.vectorstore = FAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=embedding_model)
            self.vectorstore.save_local(folder_path=self.local_dir)

    def insert_dict(self, texts: List[Dict], metadata_cols=None):
        docs = []
        try:
            for i in range(0, len(texts)):
                doc = Document(page_content=texts[i]['text'], metadata= {m: texts[i].get(m, "/") if m in texts[i] else "/" for m in metadata_cols})
                docs.append(doc)
            res = self.vectorstore.add_documents(docs)
            if self.local_dir:
                self.vectorstore.save_local(str(self.local_dir))
        except Exception as e:
            logging.error(str(e))
            res = str(e)

        return res

    def insert_docs(self, docs: List[Document], insert_batch_size=1):
        try:
            responses = []
            # Process the list in batches of 5
            for i in range(0, len(docs), insert_batch_size):
                batch = docs[i:i + insert_batch_size]
                logging.info(f"Embedding {len(batch)} documents...")
                # Process each batch (replace this comment with your processing logic)
                res = self.vectorstore.add_documents(batch)
                responses.append(res)
                logging.info(f"Documents embeddings for: {res}")
                if self.local_dir:
                    self.vectorstore.save_local(str(self.local_dir))
                    logging.info(
                        f"VECTORSTORE updated at: {self.local_dir}\nFileSize: {convert_bytes(sum([os.path.getsize(file) for file in os.scandir(self.local_dir)]))}")
            return responses
        except Exception as e:
            logging.error(str(e))
