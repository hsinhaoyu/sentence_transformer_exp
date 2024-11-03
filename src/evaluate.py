"""
Given a test set and a model, evaluate the model on an IR task.
"""
import logging
import utils
import pandas as pd
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.evaluation import SequentialEvaluator

def get_val(v):
    """
    If v is a Numpy float, turn to python flot
    """
    try:
        return v.item()
    except:
        return v


def print_and_save_eval(res, prefix = "", project_name = "dev", matryoshka_dims = None):
    """
    Print a summary of evaluation metrics of interest.
    Only use cosine correlation. dot product is very similar
    """
    metrics = ["accuracy@10", "precision@10", "recall@10", "ndcg@10", "mrr@10"]
    
    # Turn the dimensions into strings, because dimensions are embedded into model names
    if matryoshka_dims is None:
        matryoshka_dims = ["all"]
    else:
        matryoshka_dims = list(map(lambda i: str(i), matryoshka_dims))

    def model_name(m, d):
        if d == "all":
            return project_name + "_cosine_" + m
        else:
            return project_name + "_" + d + "_cosine_" + m

    summary = dict(zip(matryoshka_dims, [[]] * len(matryoshka_dims)))
    for d in matryoshka_dims:
        for m in metrics:
            k = model_name(m, d)
            summary[d] = summary[d] + [get_val(res[k])]

    df = pd.DataFrame(summary)
    df['metrics'] = metrics
    df = df.set_index('metrics')

    # printing
    logging.info(df)

    # saving
    path = utils.parent_dir('evals')

    filename = prefix + "_metrics.csv"
    df.to_csv(path / filename)

def evaluate_ir(queries,
                relevant_docs,
                corpus, model,
                project_name = 'dev',
                print_and_save = True,
                model_name = "base_model"):
    """
    Build a IR evaluator based on queries, relevant_docs and corpus, and use it to evaluate a sentence transformer (model)
    """
    
    ir_evaluator = InformationRetrievalEvaluator(
        queries = queries,
        corpus = corpus,
        relevant_docs = relevant_docs,
        name = project_name
    )

    logging.info("\nEvaluating...")
    base_model_metrics = ir_evaluator(model)

    if print_and_save:
        print_and_save_eval(base_model_metrics, prefix = model_name, project_name = project_name)

    return base_model_metrics

def evaluate_ir_matryoshka(
        queries,
        relevant_docs,
        corpus,
        model,
        matryoshka_dims,
        project_name = 'dev',
        print_and_save = True,
        model_name = "base_model"):
    """
    Build a IR evaluator based on queries, relevant_docs and corpus, and use it to evaluate a matryoshka sentence transformer (model), whose dimensions are given by matryosha_dims. Sequentially evaluate the model with reduced dimensions.
    """

    evaluators = []
    for d in matryoshka_dims:
        ir_evaluator = InformationRetrievalEvaluator(
            queries = queries,
            corpus = corpus,
            relevant_docs = relevant_docs,
            truncate_dim = d,
            name = f"{project_name}_{d}"
        )
        evaluators = evaluators + [ir_evaluator]

    evaluator = SequentialEvaluator(evaluators)

    logging.info("\nEvaluating...")
    base_model_metrics = evaluator(model)
    print(base_model_metrics)

    if print_and_save:
        print_and_save_eval(
            base_model_metrics,
            prefix = model_name,
            project_name = project_name,
            matryoshka_dims = matryoshka_dims)

    return base_model_metrics

