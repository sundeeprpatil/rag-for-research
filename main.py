import logging

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

from ingestion.parser import DatasetParsers
from ingestion.ingest import IngestDataset
import config_
from generation import generation

from infrastructures.database.faiss.db import FaissDatabase
from infrastructures.retrieval.hybrid import HybridRetriever


load_dotenv(".env")
#configure debugging
logging.basicConfig(filename='app.log', level=logging.DEBUG)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getdata')
def getdata():
    """
    Get data from semantic scholar via API 
    """
    dataset_config = config_.DATASET_CONFIG
    database_config = config_.DATABASE
    ingest  = IngestDataset(dataset_config= dataset_config)
    result = ingest.BatteryData()
    result.jsonify({"status":"ok","message": result or "Data Ingestion Complete"})


@app.route('/train')
def preprocess_and_train():
    """
    Load the data from local store or DB and further preprocess for training   
    """
    dataset_config = config_.DATASET_CONFIG
    database_config = config_.DATABASE

    dp = DatasetParsers(dataset_config)
    cleaned_chunks_docs = dp.load_and_split()
    
    print('Data parsed and split into chunks for training')

    print(f' Begin training : hybrid FAISS and BM25 ')

    retriever = HybridRetriever(database_config)

    retriever.build_index(cleaned_chunks_docs)
    

    print(f'Training Complete = Dense Indices + Sparse Indices built ')

    return 'Training Complete = Dense Indices + Sparse Indices built'

    


@app.route('/query', methods = ['POST'])
def query():
    user_query = request.json.get('query', '')
    print(f' starting to process {user_query}')

    database_config = config_.DATABASE
    prompt_config = config_.PROMPT


    retriever = HybridRetriever(database_config)
    retriever.load_index()
    result = retriever.search(user_query, top_k=5)

    print(f"query {user_query} \n")
    print(f"result - hybrid RRF  {result}")

    response = generation(user_query, prompt_config, result)


    # Convert Document objects into a serializable format (e.g., dicts)
    results_serialized = [
        {"content": doc.page_content, "metadata": doc.metadata} \
            for doc in result
    ]

    url_list = [doc.metadata['url'] for doc in result]

    return jsonify({"results":  results_serialized,
                    "generate": {"text": response['output_text'],
                                 "url": url_list  }
                        })



if __name__ == "__main__":
    import argparse 
    
    parser = argparse.ArgumentParser(description='Flask App')
    parser.add_argument('--port', type=int, default=5001,
                        help='Custom Port number')
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port, debug= True)
else:
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
