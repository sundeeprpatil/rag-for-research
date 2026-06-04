from config_ import PROMPT
from config_ import LLM
from config_ import BASE_LLM
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, pipeline


from langchain_core.prompts import PromptTemplate 
from langchain.chains.question_answering import load_qa_chain 

import torch 
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline 

#Model Architecture mapping 
SEQ2SEQ_MODELS=["t5","flan"]
CAUSAL_MODELS=["gpt","neo","llama","opt"]

def _load_model(model_id:str):

    model_id_lower = model_id.lower()
    if any(tag in model_id_lower for tag in SEQ2SEQ_MODELS):
        return AutoModelForSeq2SeqLM.from_pretrained(model_id)
    else:
        return AutoModelForCausalLM.from_pretrained(model_id)


def _get_task(model_id: str)->str: 
    model_id_lower =model_id.lower()
    if any(tag in model_id_lower for tag in SEQ2SEQ_MODELS):
        return "text2text-generation"
    return "text-generation"



def generation(query, config_prompt, docs):

    model_id = LLM[BASE_LLM]

    tokenizer  = AutoTokenizer.from_pretrained(model_id)
    model = _load_model(model_id)
    task = _get_task(model_id)
    
 

    qa_pipeline = pipeline(
        task, 
        model=model , 
        tokenizer = tokenizer,
        max_new_tokens= 500,  # Set max length of generated text
        do_sample = True,  # Control randomness of outputs
        temperature = 0.2,
    )
    llm = HuggingFacePipeline(pipeline=qa_pipeline)

    prompt = PromptTemplate(

        input_variables = ["human_input", "context"],
        template = config_prompt,
    )


    chain = load_qa_chain(
        llm= llm ,
        prompt = prompt,
        verbose = True, 
        metadata = {"application-type": "question-answering"},
    )


    response = chain.invoke( \
        { 
        "input_documents" : docs,
        "human_input" : query
        }    
    )

    print(f"{BASE_LLM}: {response}")

    return response


