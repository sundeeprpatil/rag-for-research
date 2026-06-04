import os 

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings 


class SentenceTransformerWrapper(Embeddings):
    """
    A wrapper class for the sentence transformer model
    """

    def __int__(self, model_name_or_path):
        self.embedding_model = SentenceTransformer(model_name_or_path)

    def embed_documents(self, texts):
        """
        Args : list of documents 
        Return : each element of list which is a document is encoded with SentenceTransformer 
        """
        #embedd 
        return self.embedding_model.encode(texts, convert_to_tensor = True).cpu().numpy()


