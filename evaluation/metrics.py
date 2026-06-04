"""
RAGAS Evaluation metrics
"""


from dataclasses import dataclass, field 
from typing import Optional
import json 

import json 
import os 

from datetime import datetime 

from ragas import evaluate 

from ragas.metrics import (
    faithfulness, 
    answer_relevancy,
    context_precision,
    context_recall

)

from datasets import Dataset

@dataclass 
class EvalSample: 
    """single sample evaluation """
    question: str
    answer: str  # generated answer 
    contexts:list[str] # retrieved context chunks 
    ground_truth:str # expected answer 


@dataclass 
class EvalResult: 
    """Aggregated evaluation result"""
    faithfulness: float 
    answer_relevancy:float 
    context_precision:float 
    context_recall: float 
    num_samples: int 
    timestamp: str = field(default_factory = lambda : datetime.utcnow().isoformat())


    def to_dict(self)->dict:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision":self.context_precision,
            "context_recall": self.context_recall,
            "num_samples": self.num_samples,
            "timestamp": self.timestamp,

        }

    


def run_evaluation(samples: list[EvalSample], llm=None, embeddings=None)->EvalResult:
    """
    Run RAGAS eval on list of samples 
    """

    dataset = Dataset.from_dict({
        "question": [s.question for s in samples],
        "answer": [s.answer for s in samples],
        "contexts":[s.contexts for s in samples],
        "ground_truth": [s.ground_truth for s in samples]
    })
    
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    eval_kwargs = {"dataset": dataset, "metrics":metrics}

    if llm:
        eval_kwargs["llm"]= llm
    if embeddings:
        eval_kwargs["embeddings"]= embeddings

    
    result = evaluate(**eval_kwargs)

    return EvalResult(
        faithfulness=float(result["faithfulness"]),
        answer_relevancy=float(result["answer_relevancy"]),
        context_precision=float(result["context_precision"]),
        context_recall = float(result["context_recall"]),
        num_samples= len(samples)
    )


def save_eval_results (result: EvalResult, output_dir : str ="eval_results"):
    """ Persist the evaluation results """
    os.makedirs(output_dir, exist_ok= True)

    filename = f"eval_{datetime.utcnow().strftime('%Y%m%d_%H_%m')}.json"

    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(result.to_dict(), f, indent =2 )
    
    return filepath 




