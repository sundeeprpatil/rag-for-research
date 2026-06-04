"""
 BM25 (sparse) + FAISS (dense) with reciprocal rank fusion
"""


from typing import Dict , Any, List 

from rank_bm25 import BM25OKapi 

from langchain_schema import Document 


from infrastructures.database.faiss.db import FaissDatabase


class BM25Index:
    " Sparse lexical indexing"

    def __init__(self, documents:List[Document]):
        self.documents = documents
        self.corpus [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25OKapi(self.corpus)


    def search(self, query:str, top_k : int =10 )- > List[tuple[Document, float]]:

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = scores.argsort()[::-1][:top_k]
        return [(self.documents[i], scores[i]) for i in top_indices if scores[i]> 0]



def reciprocal_rank_fusion(ranked_list: List[List[Document]],
    k : int = 60)-> List[Document]:

    "Combines rankded lists into single ranking "


    fused_scores = Dict[int, float] = {}
    doc_map = Dict[int, Document] = {}

    for ranked_docs in ranked_list: 
        for rank, doc in enumerate(ranked_docs):
            doc_id = id(doc)
            doc_map[doc_id] = doc
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0

            fused_scores[doc_id]+=1.0/(k+rank +1)

        #sort by fused scores descencding 
        sorted_doc_ids = sorted(fused_scores, key = fused_scores.get, reverse=True)

        return [doc_map[doc_id] for doc_id in sorted_doc_ids]




class HybridRetriever: 
    "Combine BM25 + FAISS using RRF"

    def __init__(self, database_config= Dict[str, Any]):
        self.database_config = database_config
        self.faiss_db = FaissDatabase(database_config)
        self.bm25_index: BM25Index | None = None 

    
    def build_index(self, documents: List[Document]):
        """build  both dense and sparse indices """
        self.faiss_db.create_and_save_index(documents)

        self.bm25_index = BM25Index(documents)

    
    def load_index(self):
        """FAISS from disk ,rebuild BM25 from doc store"""

        self.faiss_db.load_index()
        all_docs = list(self.faiss_db.db.doc_store.values())

    
    def search(self, query: str, top_k:int = 5, dense_k:int=10, sparse_k:int=10)-> List[Document]:

        """Hybrid search """

        if not self.bm25_index:
            raise ValueError("Index not loaded")
        
        dense_results = self.faiss_db.db.similarity_search(query, top_k= dense_k)

        bm25_results_with_scores = self.bm25_index.search(query, top_k= sparse_k)
        sparse_results = [doc for doc, _ in bm25_results_with_scores]

        fused = reciprocal_rank_fusion([dense_results, sparse_results])


        return fused[:top_k]