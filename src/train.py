"""
Fine-tune a pretrained model
"""

import math
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers.trainer_callback import TrainerCallback
from transformers import EarlyStoppingCallback
from sentence_transformers.training_args import BatchSamplers

import utils

##### Default parameters
N_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# Stop early if there is no improvements
# I decided not to use it, because it takes quite a bit of time
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience = 2,  # Number of epochs with no improvement after which to stop
    early_stopping_threshold = 0.0,  # Minimum improvement required to consider as improvement
)

# Print the global step
class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs = None, **kwargs):
        print(state.global_step, logs)

def mk_training_args(model_name, project_name, num_epochs, batch_size):
    """
    Return training arguments for the trainer.  Note that I am using W&B to track the training process.
    Note that I use the NO_DUPLICATES batch sampling, but I am not sure if it's helpful in this case
    """
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir = utils.parent_dir('models') / model_name,
        # Optional training parameters:
        num_train_epochs = num_epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        warmup_ratio = 0.1,
        fp16 = False,  # Set to False if GPU can't handle FP16
        bf16 = False,  # Set to True if GPU supports BF16
        batch_sampler = BatchSamplers.NO_DUPLICATES,
        # Optional tracking/debugging parameters:
        #eval_strategy = "steps",
        #eval_steps = 100,
        save_strategy = "steps",
        save_steps = 100,
        save_total_limit = 2,
        logging_steps = 50,
        logging_dir = utils.parent_dir('logs'),
        run_name = project_name,  # Used for w&b
        report_to = "wandb"
    )
    return args

def make_trainer(
        base_model,
        train_dataset,
        loss,
        learning_rate = LEARNING_RATE,
        num_epochs = N_EPOCHS,
        batch_size = BATCH_SIZE,
        model_name = 'base_model',
        project_name = "dev"):
    """
    Return a PyTorch trainer. I decided not to evaluate IR performance in training to reduce training time.
    If learning_rate is None, use the default optimizer.
    """
    args = mk_training_args(model_name, project_name, num_epochs, batch_size)

    if learning_rate is None:
        optimizer = None
    else:
        # Decided not to use the scheduler
        optimizer = AdamW(base_model.parameters(), lr = learning_rate)

    # First increase the learning rate, and then decrease it
    # Didn't seem to be useful, so it's disabled.
    #scheduler = OneCycleLR(
    #    optimizer,
    #    learning_rate,
    #    epochs = num_epochs,
    #    steps_per_epoch = math.ceil(len(train_dataset) / (batch_size)),
    #)
    scheduler = None

    trainer = SentenceTransformerTrainer(
        model = base_model,
        args = args,
        train_dataset = train_dataset,
        loss = loss,
        callbacks = [PrinterCallback],
        optimizers = (optimizer, scheduler)
    )

    return trainer

def load_finetuned_model(model_name):
    """
    Return a fine-tuned model given a model_name (str)
    """
    model_path_name = (utils.parent_dir('models') / model_name).as_posix()
    return SentenceTransformer(model_path_name)
    
