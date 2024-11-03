"""
Fine-tune a pretrained sentence transformer
"""
import logging

from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.losses import MultipleNegativesRankingLoss

import utils
import data
import train
import evaluate

logging.basicConfig(
    level = logging.INFO,
    handlers = utils.logging_handlers('finetune.log')
)

# Model card: https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
# Embedding dim = 384
BASE_MODEL = "all-MiniLM-L6-v2"
MODEL_SUFFIX = 'MNR' # For saving metrics
PROJ_NAME = 'home_depot_dev' # For tracking

# If True, use Cosinesimilarityloss to train pairs of sentences with scores. Otherwise, use MNR loss to train pairs of sentences.
METRIC_LEARNING = False

# If LEARNING_RATE is None, use the default optimizer
#LEARNING_RATE = 5e-6
LEARNING_RATE = None
BATCH_SIZE = 16
N_EPOCHS = 30

# Evaluate truncated embedding?
EVAL_TRUNCATED = True
MATRYOSHKA_DIMS = [384, 192, 96, 48]

if __name__ == '__main__':
    logging.info(f"\nLoading and prepping data...")
    test_queries, test_relevant_docs, test_corpus, train_samples = data.prep_data(use_score = METRIC_LEARNING)

    logging.info(f"\nLoading {BASE_MODEL}...")
    base_model = SentenceTransformer(BASE_MODEL)

    if METRIC_LEARNING:
        # Use cosine similarity loss function, if there is a score in the training data to indicate relevancy
        # See https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss
        loss = CosineSimilarityLoss(base_model)
    else:
        # If the training data does not include similarity scores, use the MNR loss
        # https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss
        loss = MultipleNegativesRankingLoss(base_model)

    logging.info(f"METRIC_LEARNING: {METRIC_LEARNING}")
    logging.info(f"learning_rate: {LEARNING_RATE}")
    logging.info(f"batch_size: {BATCH_SIZE}")
    logging.info(f"n_epochs: {N_EPOCHS}")
    
    trainer = train.make_trainer(
        base_model,
        train_samples,
        loss,
        learning_rate = LEARNING_RATE,
        num_epochs = N_EPOCHS,
        batch_size = BATCH_SIZE,
        model_name = BASE_MODEL,
        project_name = PROJ_NAME
    )

    trainer.train()
    trainer.save_model()

    # After training, load the fine-tuned model and evaluate
    finetuned_model = train.load_finetuned_model(BASE_MODEL)

    if EVAL_TRUNCATED:
        fine_tuned_model_eval = evaluate.evaluate_ir_matryoshka(
            test_queries,
            test_relevant_docs,
            test_corpus,
            finetuned_model,
            MATRYOSHKA_DIMS,
            project_name = PROJ_NAME,
            print_and_save = True,
            model_name = BASE_MODEL + '_finetuned' + '_' + MODEL_SUFFIX)
    else:
        fine_tuned_model_eval = evaluate.evaluate_ir(
            test_queries,
            test_relevant_docs,
            test_corpus,
            finetuned_model,
            project_name = PROJ_NAME,
            print_and_save = True,
            model_name = BASE_MODEL + '_finetuned' + '_' + MODEL_SUFFIX)
    
