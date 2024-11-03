"""
Take a pre-trained sentence transformer, and evaluate it with an IR task. The evaluation metrics are saved in a csv file.

The IR task is defined in data.py and evaluate.py.
"""
import logging

import utils
import evaluate
import data

from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level = logging.INFO,
    handlers = utils.logging_handlers('eval_base.log')
)

# I also tried BASE_MODEL = "all-mpnet-base-v2"

# Model card: https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
BASE_MODEL = "all-MiniLM-L6-v2"
PROJ_NAME = 'home_depot_dev'

# Evaluate truncated embedding?
EVAL_TRUNCATED = True
MATRYOSHKA_DIMS = [384, 192, 96, 48]

if __name__ == '__main__':
    
    test_queries, test_relevant_docs, test_corpus, _ = data.prep_data(use_score = False)

    logging.info(f"\nLoading {BASE_MODEL}")
    base_model = SentenceTransformer(BASE_MODEL)

    if EVAL_TRUNCATED:
        base_mode_eval = evaluate.evaluate_ir_matryoshka(
            test_queries,
            test_relevant_docs,
            test_corpus,
            base_model,
            MATRYOSHKA_DIMS,
            project_name = PROJ_NAME,
            print_and_save = True,
            model_name = BASE_MODEL + '_base')
    else:
        base_mode_eval = evaluate.evaluate_ir(
            test_queries,
            test_relevant_docs,
            test_corpus,
            base_model,
            project_name = PROJ_NAME,
            print_and_save = True,
            model_name = BASE_MODEL + '_base')
