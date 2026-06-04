"""
Test each RAG component independently 
"""

import os 

from dotenv import load_dotenv
from flask.wrappers import Response
from langchain_core.callbacks.manager import H
from torch.utils.data import Dataset


load_dotenv(".env")


def test_ingestion():

    import config_

    from ingestion.ingest import IngestDataset

    ingest = IngestDataset(dataset_config=config_.DATASET_CONFIG)

    print(f" URL : {ingest.base_url}")
    print(f" Query : {ingest.query}")
    print(f" DataPath : {ingest.datapath}")
    
    file_path = os.path.join(ingest.datapath, ingest.name)
    

    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path)
        print(f" Data already exitst: {file_path} ({size_mb:.1f} MB)  - Skipped ")
    else:
        print(f" Downloading started")
        result = ingest.BatteryData()


def test_parser():

    import config_

    from ingestion.parser import DatasetParsers

    dp = DatasetParsers(config_.DATASET_CONFIG)

    print(f" Files found: {dp.files_dict}")

    docs = dp.load_and_split()
    print(f" Total chunks : {len(docs)}")

    return docs



def test_faiss(docs):

    import config_

    from infrastructures.database.faiss.db import FaissDatabase

    db = FaissDatabase(config_.DATABASE)

    db.create_and_save_index(docs)

    print(f" Index saved to : {config_.DATABASE['db_base_path']}")

    # reload for search
    db2 = FaissDatabase(config_.DATABASE)
    db2.load_index()
    print(f" Index loaded successfully")

    query = "lithius ion battery degradation "
    results = db2.get_relevant_documents(query)

    print(f" Results : {len(results)}")
    
    return results 

def test_generation(results):

    import config_ 

    from generation import generation 

    query = "what causes battery degradation"

    print(f" Model: {config_.BASE_LLM} ({config_.LLM[config_.BASE_LLM]})")

    print (f" Query : {query}")

    response = generation(query, config_.PROMPT, results)

    print(f" Response : {Response}")
    return response 


def test_hybrid_retriever(docs):
    import config_ 

    from infrastructures.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever(config_.DATABASE)

    print(f" Build hybrid index ")
    retriever.build_index(docs)

    #reload from disk

    retriever2 = HybridRetriever(config_.DATABASE)

    retriever2.load_index()

    query = "what causes battery degradation"

    results = retriever2.search(query, top_k= 5)

    for i, doc in enumerate(results): 

        print(f" [{i+1}] {doc.page_content[:80]}")

    return results 


if __name__ == "__main__":

    print(" Battery Research - RAG Component tests")
    try :
        test_ingestion()
    except Exception as e: 
        print(f" Ingestion failed {e} ")
    try :
        docs = test_parser()
    except Exception as e: 
        print(f" Parsing failed {e} ")

    try :
        relevant_docs = test_faiss(docs)
    except Exception as e: 
        print(f" Something wrong in Faiss {e} ")
    
    try :
        relevant_docs = test_hybrid_retriever(docs)
    except Exception as e: 
        print(f" Something wrong in Hybrid retriever {e} ")

    try :
        response = test_generation(relevant_docs)
    except Exception as e: 
        print(f" Something wrong in Generation {e} ")
    
    