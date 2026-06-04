import config_
from ingestion.loaders import JSONLoader 
from ingestion.utils import TextSplitter, FileScanner , TextCleaner
from typing import Dict, Any, List 
from langchain.schema import Document 


class DatasetParsers:
    def __init__(self, dataset_config : Dict[str, Any]= config_.DATASET_CONFIG):
        """ Initializes the DatasetParser object 

        Args: dataset_config (dict) : configuration parameters
        
        """

        dataset_dir_path = dataset_config["dataset_dir"]

        required_document_types = dataset_config["req_doc_types"]

        pdf_loader_to_use = dataset_config["use_pdf_loader"]

        text_splitter_to_use = dataset_config["use_text_splitter"]

        file_extensions_map = {
            "json": [".json"],
           # "pdf": [".pdf"],
        }

        self.loaders = {
            ".json" : JSONLoader(),
            #".pdf" : PDFLoader(loader_to_use = pdf_loader_to_use),
        }


        self.req_doc_extensions = []
        for doc_type in required_document_types: 
            self.req_doc_extensions.extend(file_extensions_map[doc_type])

        self.files_dict = self.scan(dataset_dir_path)

        self.splitter = TextSplitter(text_splitter_to_use)


    
    def scan(self, root: str ) -> Dict[str, List[str]]:
        """
        Returns the dictionary where the key is a file type and value is a list of files of specified type
        """

        scanner = FileScanner()
        files_dict = scanner.scan(root, self.req_doc_extensions)
        return files_dict 


    def load_and_split(self) -> List[Document]:

        """
        Loads all documents in the dataset directory 

        Returns: 
        list of all loaded documents 
        """

        all_documents = []

        for document_type, files in self.files_dict.items():
            documents = self.load_by_file_type(document_type, files)
            all_documents.extend(documents)

        #all_documents = self._split(all_documents)
        return all_documents 
    

    def load_by_file_type(self, document_type:str, files : List[str]) -> List[Document]:
        """
        Loads the files of a specific document type using the corresponding loader

        Args: 

            document_type (str): document type to load
            files(list of str): list of file paths

        Returns:
            List of Documents 


        """


        loader = self.loaders.get(document_type)
        if not loader: 
            raise ValueError(f"No loader found found for document type {document_type}")
        
        pages = loader.load(files)

        return pages 
    

    def clean_documents (self, documents: List[Document]) -> List[Document]:
        """
        Cleans the content of the given documents

        Args:
         List of documents of type Document

        Return : 
         List of documents of type Document  
        """
        cleaner = TextCleaner()
        for d in documents:
            d.page_content = cleaner.clean(d.page_content)

        return documents
    


    def _split(self, docs : List[Document])-> List[Document]:
        """
        Take a list of documents and split into list of documents based on chunk 
        """
    
        return self.splitter.split(docs)




