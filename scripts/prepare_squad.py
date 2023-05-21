"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import sys
import random
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import json
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
random.seed(42)
DATA_TRAINING_FILE_NAME = "squad_data_train.jsonl"
DATA_TESTING_FILE_NAME = "squad_data_dev.jsonl"

IGNORE_INDEX = -1


def prepare(
    destination_path: Path = Path("data/squad2"), 
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    max_seq_length: int = 512,
    seed: int = 42,
    mask_inputs: bool = True,  
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.
    
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    
    destination_path.mkdir(parents=True, exist_ok=True)
    training_file_path = destination_path / DATA_TRAINING_FILE_NAME
    testing_file_path = destination_path / DATA_TESTING_FILE_NAME


    tokenizer = Tokenizer(tokenizer_path)
    
    # open the JSONL file for reading
    train_set = []
    test_set = []
    with open(training_file_path, "r") as file:
        # loop over each line in the file
        for line in file:
            # parse the line as a JSON object
            data_dict = json.loads(line)
            train_set.append(data_dict)
    with open(testing_file_path, "r") as file:
        # loop over each line in the file
        for line in file:
            # parse the line as a JSON object
            data_dict = json.loads(line)
            test_set.append(data_dict)

    print(f"original dataset of train has {len(train_set)} samples")
    print(f"original dataset of val has {len(test_set)} samples")

    train_sample_set = []
    test_sample_set = []
    print("Processing train split ...")
    for sample in tqdm(train_set):
        sample_set = prepare_sample(sample, tokenizer, max_seq_length, mask_inputs)
        if sample_set == None:
            continue
        else:
            train_sample_set.append(sample_set)
    torch.save(train_sample_set, training_file_path.parent / "train.pt")

    print("Processing test split ...")
    for sample in tqdm(test_set):
        sample_set = prepare_sample(sample, tokenizer, max_seq_length, mask_inputs)
        if sample_set == None:
            continue
        else:
            test_sample_set.append(sample_set)    
    torch.save(test_sample_set, training_file_path.parent / "test.pt")

    print(f"train dataset of max seq {max_seq_length} has {len(train_sample_set)} samples")
    print(f"val dataset of max seq {max_seq_length} has {len(test_sample_set)} samples")


    ## Save token count for analysis
    train_tokens = [len(train_data["input_ids"]) for train_data in train_sample_set]
    test_tokens = [len(test_data["input_ids"]) for test_data in test_sample_set]

    # create histogram
    plt.hist(train_tokens, bins=range(min(train_tokens), max(train_tokens) + 100), align='left')

    # set title and labels
    plt.title(f'Training dataset (Total {len(train_sample_set)} samples)')
    plt.xlabel('Number of tokens')
    plt.ylabel('Frequency')
    train_token_path = destination_path / 'train_tokens.png'
    plt.savefig(train_token_path)
    plt.clf()
    # repeat for test
    plt.hist(test_tokens, bins=range(min(test_tokens), max(test_tokens) + 100), align='mid')

    # set title and labels
    plt.title(f'Testing dataset (Total {len(test_sample_set)} samples)')
    plt.xlabel('Number of tokens')
    plt.ylabel('Frequency')
    test_token_path = destination_path / 'test_tokens.png'
    plt.savefig(test_token_path)




def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True):
    """Processes a single sample.
    
    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt_qa(example)
    full_prompt_and_response = full_prompt + example["answer"]
    encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length, eos=False)
    encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response, eos=True, max_length=max_length)
    if len(encoded_full_prompt_and_response)==max_length:
        return None
    # print(f"Length of tokens: {len(encoded_full_prompt_and_response)}")

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[:len(encoded_full_prompt)] = IGNORE_INDEX

    return {**example, "input_ids": encoded_full_prompt_and_response, "input_ids_no_response": encoded_full_prompt, "labels": labels}


def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


def generate_prompt_qa(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    return f"### Context:\n{example['context']}\n\n### Question:\n{example['question']}\n\n### Answer:\n"


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
