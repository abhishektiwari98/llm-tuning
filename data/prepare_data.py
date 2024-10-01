from datasets import load_dataset
from transformers import AutoTokenizer


def load_and_reduce_dataset(dataset_name="ag_news", train_size=50, test_size=10):
    """
    Load and optionally reduce the dataset size for quicker experiments.

    Args:
        dataset_name (str): Name of the dataset to load.
        train_size (int): Number of examples to use from the training dataset.
        test_size (int): Number of examples to use from the test dataset.

    Returns:
        datasets_dict (dict): Reduced training and test dataset.
    """
    dataset = load_dataset(dataset_name)

    dataset['train'] = dataset['train'].select(range(train_size))
    dataset['test'] = dataset['test'].select(range(test_size))

    return dataset


def load_and_prepare_tokenizer(model_name="gpt2"):
    """
    Load the tokenizer and add pad_token if not present.

    Args:
        model_name (str): Name of the model to use for loading the tokenizer.

    Returns:
        tokenizer (PreTrainedTokenizer): The loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return tokenizer


def preprocess_dataset(dataset, tokenizer, max_length=128):
    """
    Tokenize the dataset and format for PyTorch.

    Args:
        dataset (DatasetDict): The dataset to preprocess.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        max_length (int): Maximum length for tokenization.

    Returns:
        tokenized_dataset (DatasetDict): The tokenized and formatted dataset.
    """

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["label", "text"])
    tokenized_dataset.set_format("torch")

    return tokenized_dataset


def save_datasets(dataset, train_path="data/train_data", test_path="data/test_data"):
    """
    Save the processed datasets to disk.

    Args:
        dataset (DatasetDict): The dataset to save.
        train_path (str): Path to save the training dataset.
        test_path (str): Path to save the test dataset.
    """
    dataset["train"].save_to_disk(train_path)
    dataset["test"].save_to_disk(test_path)
    print("Data saved to disk at", train_path, "and", test_path)


def prepare_dataset(model_name="gpt2", dataset_name="ag_news"):
    """
    Full pipeline to prepare datasets for training and evaluation.

    Args:
        model_name (str): Model name for loading tokenizer.
        dataset_name (str): Name of the dataset to load.
    """
    print("Loading and reducing dataset...")
    dataset = load_and_reduce_dataset(dataset_name)

    print("Loading and preparing tokenizer...")
    tokenizer = load_and_prepare_tokenizer(model_name)

    print("Preprocessing dataset...")
    tokenized_dataset = preprocess_dataset(dataset, tokenizer)

    print("Saving preprocessed datasets...")
    save_datasets(tokenized_dataset)

    print("Data preparation complete!")


def main():
    """
    Entry point for the script.
    """
    prepare_dataset()


if __name__ == "__main__":
    main()
