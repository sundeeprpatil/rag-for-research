"""
Curated question-answer pairs acting as ground truth
"""

from evaluation.metrics import EvalSample, run_evaluation 


EVAL_QUESTIONS = [
    {
        "question": " what caused lithium-ion battery degradation",
        "ground_truth": ("Lithium-ion battery degrades by multiple mechanism "
        "SEI layer growth, lithium plating, active material loss, electrolyte decomposition"
        "and mechanical stress from repeated charge-discharge cycles")
    },

    {
        "question": "what is thermal runaway",
        "ground_truth":( "Thermal runaway is a uncontrollable self-heating chain reaction that occurs when"
        "internal temperature exceeds a critical threshold, leading to decomposition of"
        "cell components, gas generation and potentially fire or explosion"
        )
    }
]


def build_eval_samples(run_rag_pipeline) -> list[EvalSample]:

    """
    Build evaluation metrics by running each question through pipeline
    
    """

    samples = []

    for item in EVAL_QUESTIONS:
        result = run_rag_pipeline(item["question"])
        samples.append(
            EvalSample(
                question=item["question"],
                answer =result["answer"],
                contexts=result["contexts"],
                ground_truth=item["ground_truth"]
            )

        )

        return samples 