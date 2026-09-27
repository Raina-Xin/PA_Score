# 02_generate_adversarial_paraphrase.py
import json
import os
import logging
import argparse
import random
import re
from typing import List, Dict, Any
from copy import deepcopy
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import LlamaForCausalLM

from config import load_config, Config
import data_utils

logger = logging.getLogger(__name__)

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

class AdversarialParaphraser:
    """
    Generates adversarial paraphrases using LLM prompts designed to perturb model outputs while preserving meaning.
    """
    
    def __init__(self, tokenizer, model, config: Config):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        
        # Balanced adversarial paraphrasing prompts designed to create natural but challenging variations
        self.adversarial_prompts = [
            """Rephrase the following question using different wording while keeping the same meaning:

{question}""",

            """Rewrite this question in a different way that asks for the same information:

{question}""",

            """Paraphrase this question using alternative phrasing:

{question}""",

            """Express this question using different words and sentence structure:

{question}""",

            """Restate this question in a different way:

{question}""",

            """Rephrase this question using varied vocabulary and phrasing:

{question}""",

            """Rewrite this question with different wording but same meaning:

{question}"""
        ]

    def generate_adversarial_paraphrase_with_llm(self, question: str, prompt_template: str) -> str:
        """Generate an adversarial paraphrase using LLM with the given prompt template."""
        prompt = prompt_template.format(question=question)
        
        # Compute max length with guard
        prompt_len = len(self.tokenizer(prompt)["input_ids"])
        add_tokens = int(getattr(self.config, "max_tokens", 128) or 128)
        
        # Model/context guard
        ctx_limit = None
        if hasattr(self.model, "config") and hasattr(self.model.config, "max_position_embeddings") and self.model.config.max_position_embeddings:
            ctx_limit = int(self.model.config.max_position_embeddings)
        # prefer tokenizer limit if smaller
        tk_limit = int(getattr(self.tokenizer, "model_max_length", 2048) or 2048)
        if ctx_limit is None:
            ctx_limit = tk_limit
        max_length = min(prompt_len + add_tokens, ctx_limit)
        
        # Generate using the model
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ctx_limit)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode the response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the paraphrase from the response
        paraphrase = generated_text[len(prompt):].strip()
        
        # Clean up the response
        if "\n" in paraphrase:
            paraphrase = paraphrase.split("\n")[0].strip()
        
        # Remove common prefixes that might be added
        prefixes_to_remove = [
            "Rephrase:",
            "Rewrite:",
            "Paraphrase:",
            "Express:",
            "Restate:",
            "Alternative version:",
            "Different wording:",
            "Here's a rephrased version:",
            "Here's a different way:",
            "Here's an alternative:"
        ]
        
        for prefix in prefixes_to_remove:
            if paraphrase.startswith(prefix):
                paraphrase = paraphrase[len(prefix):].strip()
                break
        
        return paraphrase

    def generate_adversarial_paraphrases(self, question: str, n_perturb: int) -> List[str]:
        """Generate multiple adversarial paraphrases of a question using LLM prompts."""
        paraphrases = []
        seen = set()
        
        # Try different adversarial prompts
        available_prompts = self.adversarial_prompts.copy()
        random.shuffle(available_prompts)
        
        for prompt_template in available_prompts:
            if len(paraphrases) >= n_perturb:
                break
                
            try:
                paraphrase = self.generate_adversarial_paraphrase_with_llm(question, prompt_template)
                
                # Clean and validate paraphrase
                paraphrase = paraphrase.strip()
                if (paraphrase and 
                    paraphrase.lower() != question.lower() and 
                    paraphrase not in seen and
                    len(paraphrase) > 5 and   # Ensure reasonable length
                    len(paraphrase) < 200 and  # Avoid overly long responses
                    not any(char in paraphrase for char in ['```', '```python', '```json']) and  # Remove code blocks
                    not paraphrase.startswith('To ') and  # Remove meta-commentary
                    '?' in paraphrase):  # Ensure it's still a question
                    paraphrases.append(paraphrase)
                    seen.add(paraphrase)
                    
            except Exception as e:
                logger.debug(f"Error generating paraphrase with prompt: {e}")
                continue
        
        # If we don't have enough, try with different random seeds or retry
        attempts = 0
        while len(paraphrases) < n_perturb and attempts < n_perturb * 2:
            attempts += 1
            prompt_template = random.choice(self.adversarial_prompts)
            
            try:
                paraphrase = self.generate_adversarial_paraphrase_with_llm(question, prompt_template)
                
                paraphrase = paraphrase.strip()
                if (paraphrase and 
                    paraphrase.lower() != question.lower() and 
                    paraphrase not in seen and
                    len(paraphrase) > 5 and
                    len(paraphrase) < 200 and
                    not any(char in paraphrase for char in ['```', '```python', '```json']) and
                    not paraphrase.startswith('To ') and
                    '?' in paraphrase):
                    paraphrases.append(paraphrase)
                    seen.add(paraphrase)
                    
            except Exception as e:
                logger.debug(f"Error in retry attempt {attempts}: {e}")
                continue
        
        return paraphrases[:n_perturb]

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

QUESTION_END_RE = re.compile(r'(.+?\?)')  # first sentence ending in '?'

def clean_one_question(text: str) -> str:
    # grab first question sentence
    m = QUESTION_END_RE.search(text.strip())
    if m:
        cand = m.group(1).strip()
    else:
        cand = text.strip()

    # strip common prefaces
    prefaces = [
        "Rephrase:", "Rewrite:", "Paraphrase:", "Express:", "Restate:",
        "Alternative version:", "Different wording:", "Here's a rephrased version:",
        "Here's a different way:", "Here's an alternative:", "Sure,", "Certainly,"
    ]
    for p in prefaces:
        if cand.startswith(p):
            cand = cand[len(p):].strip()

    # remove quotes/bullets
    cand = cand.lstrip("-•*").strip().strip('"').strip("'")

    # ensure ends with '?'
    if not cand.endswith("?"):
        # try to salvage first line
        first_line = cand.splitlines()[0].strip()
        if first_line.endswith("?"):
            cand = first_line
        else:
            # if it looks like a statement, append '?' cautiously
            if len(first_line) > 5 and not first_line.endswith("."):
                cand = first_line + "?"
            else:
                cand = first_line

    return cand

def trivial_paraphrase_fallback(q: str) -> str:
    # very light, safe tweaks that preserve meaning
    rep = [
        (r"\bWhat is\b", "What's"),
        (r"\bWhich of the following\b", "Which option"),
        (r"\bcan you\b", "could you"),
        (r"\bplease\b", ""),
        (r"\bIn which\b", "Where"),
    ]
    out = q
    for pat, repl in rep:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    if not out.endswith("?"):
        out = out.rstrip(".") + "?"
    if out.lower() == q.lower():
        # last resort: prefix “Could you tell me …”
        out = f"Could you tell me: {q.rstrip(' ?')}?"
    return out

def generate_adversarial_paraphrases_batch(
    samples: list, 
    config: Config, 
    paraphraser: AdversarialParaphraser
):
    """
    Batched generation: single forward pass for all samples, then a small retry set.
    Returns: List[List[str]] (each inner list has exactly 1 paraphrase).
    """
    model = paraphraser.model
    tok = paraphraser.tokenizer
    n = len(samples)

    # Choose one (random) strict template per question
    templates = random.choices(paraphraser.adversarial_prompts, k=n)
    prompts = [
        templates[i].format(question=samples[i]).strip() for i in range(n)
    ]

    # Speed knobs
    max_new = int(getattr(config, "max_tokens", 48) or 48)   # smaller than 128
    gen_kwargs = dict(
        max_new_tokens=max_new,
        do_sample=True,
        temperature=0.7,         # a bit cooler for concise outputs
        top_p=0.9,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        use_cache=True,
    )

    # Tokenize as a batch (left padding already set)
    inputs = tok(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    # Decode and clean
    decoded = tok.batch_decode(out, skip_special_tokens=True)
    paraphrases = []
    failed_idxs = []

    for i, full in enumerate(decoded):
        # remove prompt prefix
        # NOTE: with left-padding + generation on many models, the decoded text usually contains the full prompt; 
        # to be safe, find the paraphrase as the suffix after the prompt.
        suffix = full[len(prompts[i]):].strip() if full.startswith(prompts[i]) else full.strip()
        cand = clean_one_question(suffix)

        ok = (
            cand and
            cand.lower() != samples[i].lower() and
            5 < len(cand) < 200 and
            "```" not in cand and
            cand.endswith("?")
        )
        if ok:
            paraphrases.append([cand])
        else:
            failed_idxs.append(i)
            paraphrases.append([None])  # placeholder

    # Retry only the failures with an even stricter template
    if failed_idxs:
        strict_template = (
            "Rewrite the question below as ONE single question.\n"
            "- Output ONLY the question.\n"
            "- No preface, no list, no quotes.\n"
            "- ≤ 20 words.\n"
            "- MUST end with a question mark.\n\n{question}"
        )
        retry_prompts = [strict_template.format(question=samples[i]) for i in failed_idxs]
        retry_inputs = tok(retry_prompts, return_tensors="pt", truncation=True, padding=True)
        if torch.cuda.is_available():
            retry_inputs = {k: v.cuda() for k, v in retry_inputs.items()}
        with torch.no_grad():
            retry_out = model.generate(**retry_inputs, **gen_kwargs)
        retry_decoded = tok.batch_decode(retry_out, skip_special_tokens=True)

        for j, idx in enumerate(failed_idxs):
            full = retry_decoded[j]
            suf = full[len(retry_prompts[j]):].strip() if full.startswith(retry_prompts[j]) else full.strip()
            cand = clean_one_question(suf)

            ok = (
                cand and
                cand.lower() != samples[idx].lower() and
                5 < len(cand) < 200 and
                "```" not in cand and
                cand.endswith("?")
            )
            if ok:
                paraphrases[idx] = [cand]
            else:
                # final fallback: trivial paraphrase that still changes surface form
                paraphrases[idx] = [trivial_paraphrase_fallback(samples[idx])]

    return paraphrases

def save_adversarial_paraphrases(paraphrased_questions_dict, config, dry_run: bool = False):
    """
    Save adversarial paraphrases to JSON (atomic write).
    """
    paraphrase_dir = os.path.join(os.path.dirname(__file__), "adversarial_questions")
    os.makedirs(paraphrase_dir, exist_ok=True)
    output_path = os.path.join(paraphrase_dir, f"{config.dataset}.json")
    temp_path = output_path + ".tmp"

    if dry_run:
        logger.info(f"[DRY-RUN] Would save {len(paraphrased_questions_dict)} entries to {output_path}")
        return

    logger.info(f"Saving adversarial paraphrased questions to {output_path}")
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

def generate_adversarial_paraphrased_questions(config: Config, model_name: str, debug=False, debug_limit=None, save_every=1, seed=None, dry_run=False):
    logger.info(f"Loading dataset: {config.dataset}")
    dataset = data_utils.load_dataset(config)

    if debug and debug_limit is None:
        debug_limit = 100
    if debug_limit is not None:
        dataset = dataset[:debug_limit]

    # Output
    paraphrase_dir = os.path.join(os.path.dirname(__file__), "adversarial_paraphrased_questions")
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

    # Seed for reproducibility
    set_seed(seed)

    # Load model once and reuse
    logger.info(f"Loading model: {model_name}")
    tokenizer, model = load_model(model_name)
    paraphraser = AdversarialParaphraser(tokenizer, model, config)

    batch_size = int(getattr(config, "batch_size", 12) or 12)
    save_every = max(1, int(save_every))
    batches_done = 0

    for i in range(0, len(remaining_dataset), batch_size):
        batch = remaining_dataset[i: i + batch_size]
        batch_questions = [entry["question"] for entry in batch]
        batch_ids = [entry["id"] for entry in batch]

        logger.info(f"Processing batch {i // batch_size + 1}: {len(batch_questions)} questions")

        paraphrased_versions = generate_adversarial_paraphrases_batch(
            batch_questions, config, paraphraser
        )

        for q_id, paraphrased_qs in zip(batch_ids, paraphrased_versions):
            paraphrased_questions_dict[q_id] = paraphrased_qs

        batches_done += 1
        if batches_done % save_every == 0:
            save_adversarial_paraphrases(paraphrased_questions_dict, config, dry_run=dry_run)

    # Final save
    save_adversarial_paraphrases(paraphrased_questions_dict, config, dry_run=dry_run)
    return paraphrased_questions_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate adversarial paraphrased questions in batches.")
    parser.add_argument("--debug", action="store_true", help="Debug mode; default limit=100 unless --debug_limit is set.")
    parser.add_argument("--debug_limit", type=int, default=None, help="Hard limit on number of samples processed.")
    parser.add_argument("--save_every", type=int, default=1, help="Save every N batches (default 1).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--dry_run", action="store_true", help="Run without writing outputs.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument("--config_path", required=True, help="Path to config file.")
    parser.add_argument("--model", required=True, help="Model name, e.g., 'Yi-34B-Chat', 'Qwen-14B', 'DeepSeek-67B'.")
    args = parser.parse_args()

    setup_logging(args.verbose)
    config = load_config(args.config_path)

    # Allow CLI to override config.model cleanly
    config.model = args.model

    generate_adversarial_paraphrased_questions(
        config=config,
        model_name=config.model,
        debug=args.debug,
        debug_limit=args.debug_limit,
        save_every=args.save_every,
        seed=args.seed,
        dry_run=args.dry_run,
    )
