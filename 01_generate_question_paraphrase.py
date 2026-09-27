# 01_generate_question_paraphrase.py
import json
import os
import logging
import argparse
from config import load_config, Config
import data_utils
import random
from copy import deepcopy
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import LlamaForCausalLM

logger = logging.getLogger(__name__)


DEFAULT_PARAPHRASE_DIR = os.path.join(os.path.dirname(__file__), "paraphrased_questions")


def resolve_paraphrase_dir(override: str = None) -> str:
    return override if override else DEFAULT_PARAPHRASE_DIR


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

def set_seed(seed: int):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model(model_name: str):
    """
    Load tokenizer/model for a given model_name with safe pad/length settings.
    """
    # --- Tokenizer ---
    if "Qwen" in model_name or "internlm" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    elif "SUS" in model_name or "Yi-34B-Chat" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Safe tokenizer defaults
    tokenizer.truncation_side = "left"
    try:
        tokenizer.model_max_length = min(int(getattr(tokenizer, "model_max_length", 2048)) or 2048, 2048)
    except Exception:
        tokenizer.model_max_length = 2048

    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif getattr(tokenizer, "unk_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id

    if getattr(tokenizer, "padding_side", None) is not None:
        tokenizer.padding_side = "left"

    # --- Model ---
    name_l = model_name.lower()
    if "falcon" in name_l or "deepseek" in name_l:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        if "deepseek" in name_l:
            model.generation_config = GenerationConfig.from_pretrained(model_name)
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
    elif "Llama" in model_name:
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    elif "Mistral" in model_name or "mpt" in model_name.lower() or "Mixtral" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    elif "Yi" in model_name or "SUS" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    elif "Qwen" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    elif "internlm" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    elif "COKAL" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, return_dict=True)
    else:
        raise NotImplementedError(f"Not supported model: {model_name}")

    # Align pad id
    if getattr(model.config, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    return tokenizer, model

def _group_pipeline_outputs(outputs, num_inputs: int, n_return: int):
    """
    Robustly convert pipeline outputs into List[List[dict]] of length num_inputs.
    Handles both nested and flat variants across HF versions.
    """
    # Already nested: len(outputs)==num_inputs and each is a list
    if isinstance(outputs, list) and len(outputs) == num_inputs and all(isinstance(x, list) for x in outputs):
        return outputs

    # Flat: length == num_inputs * n_return
    if isinstance(outputs, list) and len(outputs) == num_inputs * n_return and all(isinstance(x, dict) for x in outputs):
        grouped = []
        for i in range(num_inputs):
            grouped.append(outputs[i * n_return:(i + 1) * n_return])
        return grouped

    # Fallback: wrap each item
    return [[o] if isinstance(o, dict) else o for o in outputs]

def rephrase_question_batch(samples: list, config: Config, pipeline=None, tokenizer=None, model=None):
    """
    Rephrase a batch of questions using the paraphrasing model.
    Generates `n_perturb` paraphrases per question in one call.
    """
    prompts = [
        f"Rephrase the following question in your own words (preserving its meaning):\n"
        f"Original question: {sample}\nRephrased question:"
        for sample in samples
    ]

    # Compute per-batch max length w/ guard
    prompt_lens = [len(tokenizer(p)["input_ids"]) for p in prompts]
    prompt_len = max(prompt_lens) if prompt_lens else 0
    add_tokens = int(getattr(config, "max_tokens", 128) or 128)

    # Model/context guard
    ctx_limit = None
    if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings") and model.config.max_position_embeddings:
        ctx_limit = int(model.config.max_position_embeddings)
    # prefer tokenizer limit if smaller
    tk_limit = int(getattr(tokenizer, "model_max_length", 2048) or 2048)
    if ctx_limit is None:
        ctx_limit = tk_limit
    max_length = min(prompt_len + add_tokens, ctx_limit)

    raw_outputs = pipeline(
        prompts,
        max_length=max_length,
        num_return_sequences=config.n_perturb,
        temperature=1.1,
        do_sample=True,
        return_full_text=False,
    )

    outputs = _group_pipeline_outputs(raw_outputs, num_inputs=len(prompts), n_return=config.n_perturb)

    paraphrased_batches = []
    for original_question, output_set in zip(samples, outputs):
        seen = set()
        cleaned_versions = []
        for output in output_set:
            text = (output.get("generated_text", "") or "").strip()
            paraphrase = text.split("\n")[0].strip()
            if paraphrase.lower().startswith("rephrased question:"):
                paraphrase = paraphrase.split(":", 1)[1].strip()
            if not paraphrase:
                continue
            if paraphrase.strip().lower() == str(original_question).strip().lower():
                continue
            key = paraphrase.strip()
            if key in seen:
                continue
            seen.add(key)
            cleaned_versions.append(paraphrase)
            if len(cleaned_versions) >= config.n_perturb:
                break
        paraphrased_batches.append(cleaned_versions)

    logger.debug("Finished rephrasing one batch")
    return paraphrased_batches

def _generate_batch_paraphrases(questions, config, pipeline, tokenizer, model):
    return rephrase_question_batch(questions, config, pipeline, tokenizer, model)

def save_paraphrased_questions(paraphrased_questions_dict, config, dry_run: bool = False, paraphrase_dir: str = None):
    """
    Save paraphrases to JSON (atomic write).
    """
    paraphrase_dir = resolve_paraphrase_dir(paraphrase_dir)
    os.makedirs(paraphrase_dir, exist_ok=True)
    output_path = os.path.join(paraphrase_dir, f"{config.dataset}.json")
    temp_path = output_path + ".tmp"

    if dry_run:
        logger.info(f"[DRY-RUN] Would save {len(paraphrased_questions_dict)} entries to {output_path}")
        return

    logger.info(f"Saving paraphrased questions to {output_path}")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(paraphrased_questions_dict, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, output_path)
        logger.info(f"Successfully saved JSON to {output_path}")
    except OSError as e:
        logger.error(f"Failed to write JSON file: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

def generate_paraphrased_questions(config: Config, model_name: str, debug=False, debug_limit=None, save_every=1, seed=None, dry_run=False, paraphrase_dir: str = None):
    logger.info(f"Loading dataset: {config.dataset}")
    dataset = data_utils.load_dataset(config)

    if debug and debug_limit is None:
        debug_limit = 100
    if debug_limit is not None:
        dataset = dataset[:debug_limit]

    # Output
    paraphrase_dir = resolve_paraphrase_dir(paraphrase_dir)
    os.makedirs(paraphrase_dir, exist_ok=True)
    output_path = os.path.join(paraphrase_dir, f"{config.dataset}.json")

    paraphrased_questions_dict = {}
    if os.path.exists(output_path):
        logger.info(f"Resuming from existing file: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            paraphrased_questions_dict = json.load(f)
        paraphrased_questions_dict = {int(k): v for k, v in paraphrased_questions_dict.items()}

    remaining_dataset = [sample for sample in dataset if sample["id"] not in paraphrased_questions_dict]
    logger.info(f"Resuming with {len(remaining_dataset)} unprocessed / {len(dataset)} total.")

    # Seed for reproducibility in debug
    set_seed(seed)

    # Load model
    tokenizer, model = load_model(model_name)
    pipe = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)

    batch_size = int(getattr(config, "batch_size", 4) or 4)
    save_every = max(1, int(save_every))
    batches_done = 0

    for i in range(0, len(remaining_dataset), batch_size):
        batch = remaining_dataset[i: i + batch_size]
        batch_questions = [entry["question"] for entry in batch]
        batch_ids = [entry["id"] for entry in batch]

        logger.info(f"Processing batch {i // batch_size + 1}: {len(batch_questions)} questions")

        paraphrased_versions = _generate_batch_paraphrases(
            batch_questions, config, pipe, tokenizer, model
        )

        for q_id, paraphrased_qs in zip(batch_ids, paraphrased_versions):
            paraphrased_questions_dict[q_id] = paraphrased_qs

        batches_done += 1
        if batches_done % save_every == 0:
            save_paraphrased_questions(paraphrased_questions_dict, config, dry_run=dry_run, paraphrase_dir=paraphrase_dir)

    # Final save
    save_paraphrased_questions(paraphrased_questions_dict, config, dry_run=dry_run, paraphrase_dir=paraphrase_dir)
    return paraphrased_questions_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paraphrased questions in batches.")
    parser.add_argument("--debug", action="store_true", help="Debug mode; default limit=100 unless --debug_limit is set.")
    parser.add_argument("--debug_limit", type=int, default=None, help="Hard limit on number of samples processed.")
    parser.add_argument("--save_every", type=int, default=1, help="Save every N batches (default 1).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--dry_run", action="store_true", help="Run without writing outputs.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument("--config_path", required=True, help="Path to config file.")
    parser.add_argument("--model", required=True, help="Model name, e.g., 'Qwen/Qwen3-8B', 'meta-llama/Llama-3.1-8B-Instruct'.")
    parser.add_argument("--paraphrase_dir", type=str, default=None,
                        help=f"Directory to write paraphrase JSONs (default: {DEFAULT_PARAPHRASE_DIR}).")
    args = parser.parse_args()

    setup_logging(args.verbose)
    config = load_config(args.config_path)

    # Allow CLI to override config.model cleanly
    config.model = args.model

    generate_paraphrased_questions(
        config=config,
        model_name=config.model,
        debug=args.debug,
        debug_limit=args.debug_limit,
        save_every=args.save_every,
        seed=args.seed,
        dry_run=args.dry_run,
        paraphrase_dir=args.paraphrase_dir,
    )
