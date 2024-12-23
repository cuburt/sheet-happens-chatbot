import time
from pathlib import Path
from typing import List, Any, Tuple, Dict
from langchain.callbacks.manager import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever, Document
from langchain_community.document_transformers import LongContextReorder
from langchain_community.vectorstores import FAISS
from encoders import MsMarcoMiniLML6V2
from log import logger
PROJECT_ROOT_DIR = str(Path(__file__).parent)


def sort_documents_by_score(documents: List[Document], scores: List[Tuple[float, str]]) -> List[Document]:
    # Create a dictionary for quick lookup of scores by page_content
    score_dict = {content: score for score, content in scores}

    # Sort documents based on the scores using the score dictionary
    sorted_documents = sorted(documents, key=lambda doc: score_dict.get(doc.page_content, float('inf')))

    return sorted_documents


class RetrieverFactory(BaseRetriever):
    vectorstore: FAISS
    router: Any = None
    """VectorStore to use for retrieval."""
    search_type: str = "similarity"
    """Type of search to perform. Defaults to "similarity"."""
    k: int = 5
    skip_longcontext_reorder: bool = False
    cross_encoder = MsMarcoMiniLML6V2()

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        reranked_docs = []

        # start = time.time()
        # route = self.router(query)
        # logger.info(f"Routed to {route} chain.")
        # logger.info(f"Router RT: {(time.time() - start):.4f}")
        #
        # if not route or route == "valid":
        #     # route can be replaced by None self.rerouter.to_reroute() == False
        #     return []

        try:
            start = time.time()
            # relevant_docs = self.vectorstores[route].vectorstore.search(query, search_type="similarity_score_threshold", search_kwargs={"score_threshold": .8, "k": self.k})
            # relevant_docs = self.vectorstores[route].vectorstore.search(query, search_type="similarity", search_kwargs={"k": self.k})
            relevant_docs = self.vectorstore.search(query, search_type=self.search_type, k=self.k)
            logger.info(f"Bi-Encoder RT: {(time.time() - start):.4f}")
            page_contents = [doc.page_content for doc in relevant_docs]
        except Exception as e:
            logger.error("retriever.py @ _get_relevant_documents() bi-encoder rank " + str(e))
            relevant_docs, page_contents = [], []

        try:
            start = time.time()
            pairs = []
            for content in page_contents:
                pairs.append([query, content])
            assert len(pairs) > 0, "No relevant documents to rerank."
            scores = self.cross_encoder.predict(pairs)
            scored_docs = zip(scores, page_contents)
            sorted_contents = sorted(scored_docs, reverse=True)
            reranked_docs = sort_documents_by_score(relevant_docs, sorted_contents)
            logger.info(f"Cross-Encoder RT: {(time.time() - start):.4f}")
            relevant_docs = reranked_docs
        except AssertionError as ae:
            logger.warning(str(ae))
        except Exception as e:
            logger.error("retriever.py @ _get_relevant_documents() cross-encoder rank " + str(e))

        try:
            if not self.skip_longcontext_reorder:
                reordering = LongContextReorder()
                start = time.time()
                relevant_docs = reordering.transform_documents(relevant_docs)
                logger.info(f"LC-Reranker RT: {(time.time() - start):.4f}")
        except Exception as e:
            logger.error("retriever.py @ _get_relevant_documents() longcontext reorder " + str(e))

        return list(relevant_docs if not reranked_docs else reranked_docs)

    async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: AsyncCallbackManagerForRetrieverRun,
            **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError()
