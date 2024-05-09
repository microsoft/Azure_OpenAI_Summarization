
import logging
import os
import json
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter



class Utility:
    def __init__(self,text) -> None:
        self.text = text

    def get_chunks(self,chunk_size,chunk_overlap,text_separator= ["\n\n", "\n", " ", ""]):
        try:
            # Using RecursiveCharacterTextSplitter  - lang chain's default text splitter
            text_splitter = RecursiveCharacterTextSplitter(separators = text_separator,chunk_size = chunk_size, chunk_overlap = chunk_overlap)
            chunks = text_splitter.create_documents([self.text])
            return chunks
        except Exception as e:
            #Handle all other exceptions
            logging.error({"Msg": "Failed in get_chunks method ", "error": str(e)})
            return []


    #Helper function to get prompt text from prompt library
    def get_prompt(self,file_name):
        logging.info(" Processing Get Prompt:")
        try:
            with open(file_name,"r") as f:
                return f.read()
        except Exception as e:
            logging.error({"Msg": "Failed in get_prompt method ", "error": str(e)})
            return ''