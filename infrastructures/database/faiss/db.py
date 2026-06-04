
from config_ import DATASET_CONFIG

from typing import Dict, Any, List 

from langchain.schema import Document 
from langchain_core.embeddings import Embeddings 
from sentence_transformers import SentenceTransformer 
import faiss 
import os 
import numpy as np 
import pickle



class Faiss:
    """
    Class representing Faiss DB
    """

    def __init__(self, database_config: Dict[str, Any]) -> None:
        
       
        #6-layer language model for sentence embeddings 
        self.model = SentenceTransformer("all-MiniLM-L6-V2")


        self.index_file_name = database_config["db_base_path"]

        self.index_meta_file_name = database_config["db_base_path"].split(".")[0] + "_metadata.pkl"
        self.index_params = {
            "nlist": database_config.get("nlist", 100),
            "metric_type": faiss.METRIC_L2,
            # IVF training needs ~39 vectors per cluster; below this we use flat L2
            "min_vectors_for_ivf": database_config.get("min_vectors_for_ivf", 39),
        }

        self.index = None 
        self.dimension = None 
        
        #for incremental updates 
        self.id_to_embedding = {}
        self.next_id = 0 # store the id of the last document that was indexed
        self.doc_store = {} 

    def _create_embedding_for_documents(self, documents: List[Document]):

        texts = [doc.page_content for doc in documents]

        embedding = self.model.encode(texts, convert_to_tensor= True).cpu().numpy()

        ids = np.arange(self.next_id, self.next_id + len(embedding))

        for i, id in enumerate(ids):
            self.id_to_embedding[int(id)] = embedding[i]



        return embedding, ids

    def save_index(self):


        storage_dir = os.path.dirname(self.index_file_name)
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

        if self.index and self.index_meta_file_name:
            faiss.write_index(self.index, self.index_file_name)
            print(f'Index stored successfully')
            #TODO : store also metadata as we incremental increase 
            docs_with_metadata = {
                "docstore" : self.doc_store,
                 "next_id" : self.next_id,  
                "id_to_embedding": self.id_to_embedding
            }
            with open (self.index_meta_file_name, 'wb') as f: 
                pickle.dump(docs_with_metadata, f)



    def create_index(self, documents : List[Document]):

        """
        Creates a faiss index with the given documents 
        """

        embeddings, ids = self._create_embedding_for_documents(documents)
        self.dimension = embeddings.shape[1]
        n_vectors = len(embeddings)

        faiss.omp_set_num_threads(1)  # TODO: for now cpu
        metric_type = self.index_params["metric_type"]
        min_for_ivf = self.index_params["min_vectors_for_ivf"]

        if n_vectors < min_for_ivf:
            base_index = faiss.IndexFlatL2(self.dimension)
        else:
            # FAISS requires n_vectors >= nlist for IVF training
            nlist = min(self.index_params["nlist"], n_vectors)
            quantizer = faiss.IndexFlatL2(self.dimension)
            base_index = faiss.IndexIVFFlat(
                quantizer, self.dimension, nlist, metric_type
            )
            base_index.train(embeddings)

        # Flat/IVF indexes do not support add_with_ids; IndexIDMap adds that API
        self.index = faiss.IndexIDMap(base_index)
        self.index.add_with_ids(
            embeddings.astype(np.float32),
            ids.astype(np.int64),
        )
        self.next_id = int(ids[-1]) + 1
        self.doc_store.update({int(id_): doc for id_, doc in zip(ids, documents)})

    def load_index(self):

        self.index = faiss.read_index(self.index_file_name)
        if self.index.ntotal > 0:
            embedding_vector = np.zeros((1, self.index.d))
            self.dimension = embedding_vector.shape[1]
            print(f'index is loaded')
        else:
            raise ValueError('Index file is empty ')

        with open(self.index_meta_file_name, 'rb') as file_name:
            docs_with_metadata = pickle.load(file_name)
            print(f' index metadata loaded')

        self.doc_store = docs_with_metadata["docstore"]
        self.next_id = docs_with_metadata["next_id"]
        self.id_to_embedding = docs_with_metadata["id_to_embedding"]






    def similarity_search(self, query : str, top_k: int = 5,
                          extra_chunks_required:int = None ):
        """
        Search the faiss index for the given query
        """

        if not self.index:
            raise ValueError("Index is not created")
    
        query_embedding = self.model.encode(query)
        distance, indices  =  self.index.search(np.reshape(query_embedding, [1, self.dimension]), 5)


        
        results = self.get_chunks_by_indices(indices[0], extra_chunks_required)

        return results 


    def get_chunks_by_indices(self, indices, extra_chunks_required= None):

        results = []

        new_chunk_ids = list(dict.fromkeys(indices))

        for chunk_uid in new_chunk_ids:
            if chunk_uid != -1: 
                results.append(self.doc_store[int(chunk_uid)])

        
        return results 



class FaissDatabase:

    def __init__(self, database_config :Dict[str, Any]) -> None:
        
        self.db = Faiss(database_config)

    
    def create_and_save_index(self, documents : List[Document]):

        self.db.create_index(documents)

        self.db.save_index()

    def load_index(self):

        self.db.load_index()

    def get_relevant_documents(self, query):

        return self.db.similarity_search(query)
