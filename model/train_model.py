from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk


def load_model(model_name="gpt2"):
    """
    Load and prepare the model and tokenizer.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        model (AutoModelForCausalLM): Loaded language model.
        tokenizer (AutoTokenizer): Loaded tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add pad_token if not present and update model embeddings
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def load_datasets(train_path="data/train_data", test_path="data/test_data",
                  train_size=50, eval_size=10):
    """
    Load and process the training and evaluation datasets.

    Args:
        train_path (str): Path to the training data.
        test_path (str): Path to the evaluation data.
        train_size (int): Number of training examples to use.
        eval_size (int): Number of evaluation examples to use.

    Returns:
        train_dataset, eval_dataset: Processed training and evaluation datasets.
    """
    train_dataset = load_from_disk(train_path).select(range(train_size))
    eval_dataset = load_from_disk(test_path).select(range(eval_size))

    # Add labels for causal language modeling
    def add_labels(batch):
        batch["labels"] = batch["input_ids"].clone()
        return batch

    train_dataset = train_dataset.map(add_labels, batched=True)
    eval_dataset = eval_dataset.map(add_labels, batched=True)
    return train_dataset, eval_dataset


def create_trainer(model, train_dataset, eval_dataset, output_dir="./results",
                   logging_dir="./logs", learning_rate=5e-5, train_batch_size=8,
                   num_train_epochs=3, weight_decay=0.01):
    """
    Create a Trainer instance with the specified parameters.

    Args:
        model: The model to train.
        train_dataset: The training dataset.
        eval_dataset: The evaluation dataset.
        output_dir (str): Directory to save the model outputs.
        logging_dir (str): Directory to save the training logs.
        learning_rate (float): The learning rate for training.
        train_batch_size (int): The batch size for training.
        num_train_epochs (int): The number of epochs for training.
        weight_decay (float): Weight decay parameter.

    Returns:
        trainer (Trainer): Configured Trainer instance.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        logging_dir=logging_dir,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    return trainer


def prompt_tune():
    """
    Main function to run the prompt tuning process.
    """
    print("Loading model and tokenizer...")
    model, tokenizer = load_model()

    print("Loading and processing datasets...")
    train_dataset, eval_dataset = load_datasets()

    print("Creating Trainer...")
    trainer = create_trainer(model, train_dataset, eval_dataset)

    print("Starting training...")
    trainer.train()

    print("Saving the trained model and tokenizer...")
    model.save_pretrained("./prompt-tuned-model")
    tokenizer.save_pretrained("./prompt-tuned-model")
    print("Prompt Tuning Complete!")


def main():
    """
    Entry point of the script.
    """
    prompt_tune()


if __name__ == "__main__":
    main()
