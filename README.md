# Fine-Tune a Sentence Transformer with and without Matryoshka Loss

### Contents
- `src/`: Scripts for fine-tuning and evaluation
- `logs/`: Logs generated during runs
- `evals/`: Metrics for evaluation
- `models/`: Saved models
- `report/`: Visualization of the results and summary of the experiments.

### How to run?
Under `src/`,
- `python data.py`: Print examples of query/retrieval pairs in the test set.
- `python eval_base.py`: Evaluate the pre-trained [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model with an IR (Information Retrieval) task
- `python finetune_base`: Finetune the base model without Matryoshka loss
- `python finetine_matryoshka`: Fine tune base model with Matryoshka loss

The training process can be monitored with [wandb](https://wandb.ai/site/).
