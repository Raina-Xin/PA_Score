import json
import os

import datasets
import numpy as np
import openai
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer

import models as models
from config import Config
from copy import deepcopy

from typing import Optional, Any

device = 'cuda'
# device = config.device
# openai.api_key = config.openai_api_key

def load_questions(config: Config):
    questions_dir = "questions"
    questions_filename = f'{config.dataset}.json'
    questions_filename = os.path.join(questions_dir, questions_filename)
    questions = load_json(questions_filename)
    return questions

def load_json(filename):
    with open(filename, 'r') as f:
        dic = json.load(f)
        dic = {int(k): v for k, v in dic.items()}
    return dic
    
def sample_from_perturbed_questions(idx, perturbed_questions, config: Config):
    # The first question in perturbed_questions is the original question
    n_perturb = config.n_perturb
    perturbed_questions = perturbed_questions[idx]
    assert len(list(set(perturbed_questions))) >= n_perturb, f"Number of perturbed questions ({len(perturbed_questions)}) is less than the number of perturbations requested ({n_perturb})."
    
    # We sample n_perturb questions from perturbed_questions[1:n_perturb + 1] because the first question is the original question
    perturbed_questions = np.random.choice(perturbed_questions[1:], n_perturb, replace=False)
    return perturbed_questions

def sample_to_prompt(question, config: Config, full_sentence_response=False, **kwargs):
    # We use the same prompts as the Meta Llama paper for NQ and TriviaQA https://arxiv.org/pdf/2302.13971
    if isinstance(question, list):
        return [sample_to_prompt(q, **kwargs) for q in question]
    if config.dataset == 'trivia_qa':
        return f"""Answer these questions: 
    Q: In Scotland, a bothy/bothie is a?
    A: House
    Q: {question}
    A:"""

    elif config.dataset == 'nq':
        return f"""Answer these questions:
    Q: Who sang who wants to be a millionaire in high society?
    A: Frank Sinatra
    Q: {question}
    A:"""

    else:
        raise NotImplementedError

def sample_to_prompt_zero_shot(question, **kwargs):
    if isinstance(question, list):
        return [sample_to_prompt_zero_shot(q, **kwargs) for q in question]
    return question

def perturb_sample(sample, config: Config): 
    return rephrase_question(sample, config)
    


def load_dataset(config: Config, shuffle=False) -> list[dict]:
    """
    Load dataset from JSON files in the 'data' directory.

    Parameters
    ----------
    config : Config
        Configuration object that includes the dataset name and other settings.
    shuffle : bool, optional
        Whether to shuffle the dataset (default is False).

    Returns
    -------
    List[Dict]
        A list of dictionaries, where each dictionary represents a data sample with:
        - "index": unique sample index
        - "input": question
        - "context": optional context for tasks that require it
        - "choices": multiple-choice answer options (if available)
        - "answer": correct answer
    """
    
    data_file = os.path.join("data", f"{config.dataset}.json")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset file '{data_file}' not found!")

    # Load the dataset from JSON
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = []
    for i, entry in enumerate(data):
        sample = {
            "id": i,
            "question": entry["question"],  # The primary question
            "context": entry.get("context", ""),  # Include context if available
            "choices": entry.get("choices", {}),  # Multiple-choice answers
            "answer": entry["answer"]  # Correct answer
        }
        formatted_data.append(sample)

    if shuffle:
        np.random.shuffle(formatted_data)

    # Limit dataset size based on configuration
    return formatted_data

def load_test_dataset(config: Config, shuffle=False) -> list[dict]:
    """
    Load dataset from JSON files in the 'data' directory.

    Parameters
    ----------
    config : Config
        Configuration object that includes the dataset name and other settings.
    shuffle : bool, optional
        Whether to shuffle the dataset (default is False).

    Returns
    -------
    List[Dict]
        A list of dictionaries, where each dictionary represents a data sample with:
        - "index": unique sample index
        - "input": question
        - "context": optional context for tasks that require it
        - "choices": multiple-choice answer options (if available)
        - "answer": correct answer
    """
    
    data_file = os.path.join("data", f"{config.dataset}.json")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset file '{data_file}' not found!")

    # Load the dataset from JSON
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = []
    for i, entry in enumerate(data):
        sample = {
            "id": i,
            "question": entry["question"],  # The primary question
            "context": entry.get("context", ""),  # Include context if available
            "choices": entry.get("choices", {}),  # Multiple-choice answers
            "answer": entry["answer"]  # Correct answer
        }
        formatted_data.append(sample)

    if shuffle:
        np.random.shuffle(formatted_data)

    # Limit dataset size based on configuration
    return formatted_data[:config.n_test]

def parse_response(response):
    response = response.strip()
    return response

def generate_response(prompt: str, config: Config, pipeline: Optional[Any] = None, tokenizer: Optional[Any] = None):
    if config.model.startswith('gpt'):
        completions = models.gpt(
            system_prompt=None,
            prompt=prompt,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            n=config.n_sample,
            stop=config.stop
        )
        return [parse_response(response) for response in completions]
    
    elif config.model.startswith('llama'):
        sequences = pipeline(
            prompt,
            num_return_sequences=config.n_sample,
            eos_token_id=tokenizer.eos_token_id,
            max_length=len(tokenizer(prompt)['input_ids'])+config.max_tokens,
            # truncation=True
        )
        return [parse_response(response['generated_text'][len(prompt)+1:].split('\n')[0]) for response in sequences]

def get_cls_embeddings(outputs: np.array, model, tokenizer, to_numpy=True) -> torch.Tensor:
    """
    Calculate the [CLS] token embeddings from the last layer of the model for multiple outputs.

    Args:
        outputs: Array of strings with shape (batch_size,)
        model: The pre-trained model
        tokenizer: The tokenizer associated with the model
        to_numpy: Whether to convert the embeddings to a NumPy array (default: True)

    Returns:
        sentence_embeddings: Tensor of shape (batch_size, embedding_dim)
    """
    outputs = outputs.tolist()

    # Move the model to the CPU
    model = model.to('cpu')

    encoded_inputs = tokenizer(outputs, padding=True, truncation=False, return_tensors='pt')

    # Move the encoded inputs to the CPU
    encoded_inputs = {k: v.to('cpu') for k, v in encoded_inputs.items()}

    with torch.no_grad():
        model_output = model(**encoded_inputs)

    # Get the [CLS] token embeddings from the last layer
    sentence_embeddings = model_output.last_hidden_state[:, 0, :]

    if to_numpy:
        sentence_embeddings = sentence_embeddings.cpu().numpy()

    return sentence_embeddings
