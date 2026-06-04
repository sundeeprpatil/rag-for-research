import re 
from typing import List 

from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain.schema import Document 

TEXT_SPLITTERS = {
    "spacysemanticsplitter": {
        "chunk_size": 500,
    },
    "recursivecharacter":{
        "chunk_size": 500,
        "chunk_overlap":0,
        "separators": ["\n\n", "\n", " ", ""],
    },

    "charactersplitter":{
        "separator":"\n\n",
        "chunk_size": 500,
        "chunk_overlap":0,
        "length_function": len,
        "is_separator_regex": False,
    }
}

class SpacySemanticSplitter:
    """ Class to semantically split documents into smaller chunks using spacy NLP library.
    The splitting is performed at sentence boundaries, aiming to keep the chunk size below a
    specified limit while maintaining semantic coherence 
    
    Attributes: 
        chunk_size : int, optional
        nlp: spacy language model

    """

    def __init__(self, chunk_size: int):
    
        self.chunk_size = chunk_size
        self.nlp = spacy.load('en_core_web_sm')


    def _trim_spaces(self, text: str):

        """
        Replace multiple spaces and tabs with single space.
        Useful  for chunking to calculate number of valid characters
        when there are more spaces than characters 
        
        Parameters: 
            text: str input text
        Returns :
            text: cleaned version
        """


        if text:
            text = re.sub(r'{1,}', ' ', text)
            text = re.sub(r'\t{1,}', ' ', text)
            
        return text 



    def  merge_chunks(self, text: str, chunk_size : int, max_chunk_size_allowed : int = 750)-> List[str]:
        """
            Splits the input text into chunks based on th provided chunk size ensuring that
            each chunks size does not exceed specific limits 

        Args: 
            text(str): input text 
            chunk_size (int)
            max_chunk_size_allowed (int, optional)

        Returns:
            List[str] : A list of text chunks each confined to limits
        """

        if text is None:
            return []
        
        doc = self.nlp(text)

        sentences = [sent.text for sent in doc.sents]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            trimmed_sentence = self._trim_spaces(sentence)
            actual_chunk_len = len(current_chunk)
            trimmed_chunk_len = len(self._trim_spaces(current_chunk))
            trimmed_sentence_len = len(trimmed_sentence)

            if trimmed_chunk_len + trimmed_sentence_len <= chunk_size and \
                actual_chunk_len + len(sentence) <= max_chunk_size_allowed:
                current_chunk += sentence 

            else:
                while len(current_chunk) + len(sentence) > max_chunk_size_allowed:
                    remaining_space = max_chunk_size_allowed - len(current_chunk)
                    sentence_to_add = sentence[:remaining_space]
                    current_chunk += sentence_to_add
                    sentence = sentence[remaining_space:]
                    chunks.append(current_chunk.rstrip())
                    current_chunk = ""

                if current_chunk: 
                    chunks.append(current_chunk.rstrip())
                
                current_chunk = sentence 

            
        if current_chunk:
            chunks.append(current_chunk.rstrip())

        
        #Merge small chunks with the next chunk if possible but do not exceed max_chunk_size_allowed    
        merged_chunks = []
        for chunk in chunks:
            trimmed_chunk = self._trim_spaces(chunk)
            actual_chunk_len = len(chunk)
            trimmed_chunk_len = len(trimmed_chunk)

            if merged_chunks and trimmed_chunk_len + len(self._trim_spaces(merged_chunks[-1])) <= chunk_size \
                  and actual_chunk_len + len(merged_chunks[-1]) <= max_chunk_size_allowed:
                merged_chunks[-1] += chunk
            else:
                merged_chunks.append(chunk)

        
        return merged_chunks




    def split_documents(self, pages: List[Document]) -> List[Document]:
        """
        Splits the content of each page into smaller chunks based on semantic boundaries

        Parameters:
        pages : List[Document]: List of Document objects containing page_content (str) and metadata (dict)
        Returns:
         List[Document] where each list element has page_content limited to chunk size 

        """

        split_docs = []
        for page in pages :
            text = page.page_content
            doc_metadata = page.metadata 

            chunks = self.merge_chunks(text, self.chunk_size)

            for chunk in chunks:
                split_docs.append(Document (page_content = chunk , metadata = doc_metadata.copy()))

        
        return split_docs
    


class TextSplitter:

    def __init__(self, text_splitter_to_use : str = "recursivecharacter"):

        """
        Initialize text splitter object along with the splitter to use along with it

        """

        self.text_splitter_to_use = text_splitter_to_use
        self.transforms = TEXT_SPLITTERS


    def spacysemanticsplitter(self, pages : List[Document]) -> List[Document]:

        """
        Applies Spacy Semantic Text Splitter on Documents of pages
        
        Args: 
            pages (list of Documents)
        
        Returns:
            list: document chunks
        """

        splitter = SpacySemanticSplitter (
            **self.transforms["spacysemanticsplitter"]
        )


        return splitter.split_documents(pages)
    



    def recursivecharacter(self, pages: List[Document])-> List[Document]:
        
        splitter = RecursiveCharacterTextSplitter(
            **self.transforms["recursivecharacter"]
        )
        return splitter.split_documents(pages)
    
    


    def charactersplitter(self, pages: List[Document])-> List[Document]:

        print('character splitter to be implemented')

    
    def add_chunk_info_to_metadata(self, documents: List[Document]) -> List[Document]:

        """
        Splits the pages using a specified text splitter 

        Args:
         documents : List[Document]

        Returns:

            List[Document] with content = chunk, metadata = chunk number and number of chunks in doc

        """


        number_of_chunks_in_doc = len(documents)

        for id in range(number_of_chunks_in_doc):
            documents[id].metadata["chunk_number"] = id+ 1
            documents[id].metadata["total_chunks"] = number_of_chunks_in_doc 



        return documents 
    


    def split(self, pages : List[Document])-> List[Document]:

        """
        Splits the pages with specified splitter 

        Returns: list of document chunks 
        """


        router = {
            "spacysemanticsplitter" : self.spacysemanticsplitter,
            "recursivecharacter": self.recursivecharacter ,
            "charactersplitter": self.charactersplitter,

        }

        split_pages = router[self.text_splitter_to_use](pages)

        modified_chunks = self.add_chunk_info_to_metadata(split_pages)

        return modified_chunks
    




