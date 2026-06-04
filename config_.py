
import os

BASE_LLM =  os.environ.get("BASE_LLM", 'flan-t5-small') # 'gpt2' #

DATASET_CONFIG = {
"dataset_dir" : "dataset",
"req_doc_types": ["json"], #, "pdf"],
"use_text_splitter": "recursivecharacter",
"use_pdf_loader": "pymupdf"
}

DATABASE = { \
    "db_base_path": "indexes/faiss/db.index"}

RETRIEVALS = {

    "mmr_search_kwargs": {
        "k": 5,
        "fetch_k":20,
        "lambda_multi": 0.2,
        "extra_chunks_required": 3
    },
    "similarity_topk_search_kwargs":{
        "top_k": 5,
        "extra_chunks_required": 3
    },
    "hybrid":{
        "top_k": 5,
        "dense_k":10,
        "sparse_k":10,
        "rrf_k": 60 , # reciprocal rank fusion
    
    }
}

PROMPT = """ <info> You are an expert assistant and well knowledable in lithium-ion battery technology.
    You provide detailed answers, suggest follow-up and engage in meaningful conversations.
    The knowledge is coming from set of documents retrieved from vector database. 
    You can suggest follow up questions and engage in meaningful convesations
    with users. Ensure coherence and avoid repitition of information.</info>. 
    The human asked the following question <question>{human_input}<\question>.
    These are the context relevant to the question from the human <context>{context}<\context>.    
"""


LLM = {
"distillBert" : "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
"flan-t5-small" : "google/flan-t5-small" ,
"gpt-neo-125M": "EleutherAI/gpt-neo-125M",
"gpt2":  "gpt2",

}
