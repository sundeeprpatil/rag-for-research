import re 
import string 
import unicodedata 

from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 



class TextCleaner :
    """
    Class that provides methods to clean up text 
    """

    def __init__(self, text: str) -> None:
        self.text = text 

    def clean(self):
        """
        
        """
        self.remove_non_ascii() 
        self.remove_trailing_spaces_and_newlines()

        return self.text 

    def remove_non_ascii(self):
        """
        Remove non-ascii from a given text.
        """

        self.text = unicodedata.normalize("NFKD", self.text)\
            .encode("ascii", "ignore") \
            .decode("utf-8", "ignore")
        

    def remove_trailing_spaces_and_newlines(self):
        """
            Additional clean up 
        """


        text = re.sub(r'[ \t]+(?=\n)', '', self.text) # match space or tab before newline and remove

        text = re.sub(r'\n+', '\n', text) # multiple new line with one new line

        self.text = text.rstrip() # remove trailing spaces


