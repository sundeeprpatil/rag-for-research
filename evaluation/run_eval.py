"""
Standalone evaluation runner 

"""

import os
import sys 


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from dotenv import load_dotenv 

load_dotenv(".env")

import config_

from infrastructures.database.faiss.db import FaissDatabase

from generation import generation 

from evaluation.metrics import run_evaluation, save_eval_results

from evaluation.test_set import build_eval_samples, EVAL_QUESTIONS 


from langchain_openai import ChatOpenAI, OpenAIEmbeddings




JUDGE_LLM = ChatOpenAI(
    model = 'gpt-4o-mini',
    base_url = "https://models.inference.ai.azure.com",
    api_key = os.environ['GITHUB_TOKEN'],
    temperature =0  # stay factual 

)

JUDGE_EMBEDDINGS = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url = "https://models.inference.ai.azure.com",
    api_key=os.environ['GITHUB_TOKEN'],
)

def run_pipeline(question: str) -> dict :
    """
    Run the full retrieval + generation for single question
    """

    prompt_config = config_.PROMPT
    db = FaissDatabase(config_.DATABASE)

    db.load_index()

    docs = db.get_relevant_documents(question)

    response = generation(question, prompt_config, docs)


    return {
        "answer": response["output_text"],
        "contexts":[doc.page_content for doc in docs ],

    }



def running_evals():

    print(f" Running RAG pipeline for evaluation")

    print(f"Running {len(EVAL_QUESTIONS)} evaluation questions ")

    print(f" Model : {config_.LLM[config_.BASE_LLM]}")


    samples= build_eval_samples(run_pipeline)

    result = run_evaluation(samples, llm= JUDGE_LLM, embeddings= JUDGE_EMBEDDINGS)


    print(f" Faithfullness : {result.faithfulness:.4f }")

    print(f" Answer relevancy: {result.answer_relevancy:.4f}")

    print(f" Context precision : {result.context_precision:.4f}")

    print(f" context recall : {result.context_recall:.4f }")


    file_path = save_eval_results(result)

    print(f" Results saved at {file_path}")


if __name__ == "__main__":

    running_evals()    