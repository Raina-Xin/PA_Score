#!/usr/bin/env python3
import os
import json
import torch
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import LlamaForCausalLM
import random

import prompt as pt  

# ---------------------------
# Utils
# ---------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prefer_bf16():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def dtype_for_model():
    return torch.bfloat16 if prefer_bf16() else torch.float16

# ---------------------------
# Model & Tokenizer
# ---------------------------

def load_model(args):
    # tokenizer
    if "Qwen" in args.model or "internlm" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    elif "Phi-3" in args.model or "phi-3" in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    elif "Yi" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = min(getattr(tokenizer, "model_max_length", 2048), 2048)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Force left-padding for batched inference: this script reads logits at
    # `last_hidden_state[:, -1, :]`, so right-padding would put the "last"
    # position on a pad token and produce meaningless A..F logits. Left-
    # padding keeps the actual answer position at index -1 regardless of
    # input length within the batch. Affects every model whose tokenizer
    # defaults to padding_side="right" (Llama-3.1, Qwen2.5, Qwen3); no-op
    # for tokenizers that already default to "left" (Phi-3-small).
    tokenizer.padding_side = "left"

    # model
    torch_dtype = dtype_for_model()
    common_kwargs = dict(device_map="auto", dtype=torch_dtype)

    if "falcon" in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model, **common_kwargs)
    elif "deepseek" in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model, **common_kwargs)
        model.generation_config = GenerationConfig.from_pretrained(args.model)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    elif "Llama" in args.model:
        model = LlamaForCausalLM.from_pretrained(args.model, **common_kwargs)
    elif "Yi" in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model, **common_kwargs)
    elif "Phi-3" in args.model or "phi-3" in args.model.lower():
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, **common_kwargs)
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.eos_token_id
    elif "Qwen" in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, **common_kwargs)
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.eos_token_id
    else:
        # Fallback
        model = AutoModelForCausalLM.from_pretrained(args.model, **common_kwargs)

    model.eval()
    return tokenizer, model

# ---------------------------
# Data loading
# ---------------------------

def load_data(data_file, debug=False, debug_cap=100):
    with open(data_file, "r") as f:
        data = json.load(f)
    if debug:
        data = data[:min(len(data), debug_cap)]
    return data

def load_paraphrased_questions(paraphrase_file):
    with open(paraphrase_file, 'r') as f:
        return json.load(f)

# ---------------------------
# Few-shot helpers
# ---------------------------

DEFAULT_FEWSHOT_BY_SOURCE = {
    "MMLU": 5,
    "HellaSwag": 5,
    "CosmosQA": 5,
    "Halu-OpenDialKG": 3,
    "Halu-CNN/DailyMail": 1,
}

def get_fewshot_exps(data, k=None, exclude_id=None):
    """
    Robust few-shot selection:
    - Keep same source as current data[0]['source'] (assumes dataset is homogeneous; if not, we filter by example['source'])
    - Exclude current example id.
    - Take first k examples (default from DEFAULT_FEWSHOT_BY_SOURCE)
    """
    if not data:
        return []
    src = data[0].get("source", None)
    if src is None:
        return []
    if k is None:
        k = DEFAULT_FEWSHOT_BY_SOURCE.get(src, 3)

    few = []
    for ex in data:
        if ex.get("source") == src and ex.get("id") != exclude_id:
            few.append(ex)
            if len(few) >= k:
                break
    return few

# ---------------------------
# Prompt formatting
# ---------------------------

def format_example(example, prompt, with_answer=False):
    src = example["source"]
    if src == "MMLU":
        prompt += "Question: " + example["question"] + "\nChoices:\n"
    elif src == "CosmosQA":
        prompt += "Context: " + example["context"] + "\n" + "Question: " + example["question"] + "\nChoices:\n"
    elif src == "HellaSwag":
        prompt += "Context: " + example["context"] + "\n" + "Question: " + example["question"] + "\nChoices:\n"
    elif src == "Halu-OpenDialKG":
        prompt += "Dialogue: " + example["context"] + "\n" + "Question: " + example["question"] + "\nChoices:\n"
    elif src == "Halu-CNN/DailyMail":
        prompt += "Document: " + example["context"] + "\n" + "Question: " + example["question"] + "\nChoices:\n"
    else:
        raise NotImplementedError(f"Unsupported dataset source: {src}")

    for k, v in example["choices"].items():
        prompt += f"{k}. {str(v)}\n"
    prompt += "Answer:"
    if with_answer:
        prompt += " " + example["answer"] + "\n"
    return prompt

def _apply_chat_template_safe(tokenizer, text: str) -> str:
    # Try tokenizer's chat template first; fallback to a simple single-turn prompt.
    try:
        msg = [{"role": "user", "content": text}]
        # Qwen3's default template leaves the assistant turn open after
        # `<|im_start|>assistant\n` so the model emits a `<think>...</think>`
        # reasoning block before answering. That makes the next-token logit at
        # A/B/C/D... pure noise (P(<think>) ≈ 1, P(letter) ≈ 1e-15). Force-
        # disable thinking so the template injects an empty
        # `<think>\n\n</think>\n\n` and the next tokens are the actual answer.
        # Harmless on non-Qwen3 templates.
        kwargs = {}
        name_or_path = (getattr(tokenizer, "name_or_path", "") or "").lower()
        if "qwen3" in name_or_path:
            kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, **kwargs)
    except Exception:
        return f"{text}\n"

def format_task_prompt(example, args, tokenizer, fewshot_exps=None):
    src = example["source"]
    if args.few_shot == 0 and not args.cot:
        pt_dict = json.loads(pt.task_zero_prompt, strict=False)
        prompt = pt_dict[src]
    elif args.few_shot > 0 and not args.cot:
        pt_dict = json.loads(pt.task_few_prompt, strict=False)
        prompt = pt_dict[src]
        fs = fewshot_exps or []
        for fs_exp in fs:
            prompt = format_example(fs_exp, prompt, with_answer=True)
        prompt += "\nNow make your best effort and select the correct answer for the following question. You only need to output the option.\n\n"
    elif args.few_shot == 0 and args.cot:
        pt_dict = json.loads(pt.task_cot_prompt, strict=False)
        prompt = pt_dict[src]
    else:
        raise NotImplementedError("Unsupported method (few_shot, cot).")

    prompt = format_example(example, prompt)

    # Falcon family expects explicit "User/Assistant" sometimes; keep your special-case if desired
    if "falcon" in args.model.lower():
        prompt = "User: " + prompt + "\nAssistant:"
    else:
        prompt = _apply_chat_template_safe(tokenizer, prompt)

    return {"id": example["id"], "prompt": prompt}

# ---------------------------
# Choice token id inference
# ---------------------------

def infer_option_token_ids(tokenizer, base_text: str, options=("A","B","C","D","E","F")):
    """
    Robustly infer the *next-token ids* for options after 'Answer:' by adding ' Answer: ' + option,
    then taking the token id of that option's first token.
    """
    option_ids = []
    prefix = base_text
    # ensure trailing space before option
    if not prefix.endswith(" "):
        prefix += " "
    for opt in options:
        sample = prefix + opt
        ids = tokenizer.encode(sample, add_special_tokens=False)
        # We need the last token id (corresponding to opt's first token)
        # Find the suffix token that appears when we append opt
        # Simplest: encode prefix and diff
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        if len(ids) <= len(prefix_ids):
            # Fallback: just take last id
            option_ids.append(ids[-1])
        else:
            option_ids.append(ids[len(prefix_ids)])
    return option_ids

# ---------------------------
# Main computation
# ---------------------------

def get_hidden_representations(model, tokenizer, data, paraphrased_questions, args):
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = (args.model).split('/')[-1].replace("/", "_")
    checkpoint_file = os.path.join(args.output_dir, f"{args.dataset_name}_{model_name}_results.pkl")

    # resume if present
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            all_results = pickle.load(f)
        start_idx = len(all_results)
        print(f"[Resume] Found checkpoint with {start_idx} processed examples.")
    else:
        all_results, start_idx = [], 0

    options = ["A", "B", "C", "D", "E", "F"]

    # Few-shot exemplars (dynamic per example due to exclusion)
    # Option ids need the exact 'Answer:' prefix the prompt will have.
    # We'll infer in-batch using the actual prompt text later to be 100% safe.

    pbar = tqdm(range(start_idx, len(data)), initial=start_idx, total=len(data))
    for i in pbar:
        datum = data[i]
        qid = str(datum["id"])

        # ---- Build 7 variants (original + up to 6 paraphrases), de-duplicated, preserve order
        paraphrases = paraphrased_questions.get(qid, [])
        # if the paraphrase file includes original at index 0, remove duplicates with the original
        variants = []
        seen = set()
        def _add_variant(text):
            key = text.strip()
            if key not in seen and len(key) > 0:
                seen.add(key)
                variants.append(text)

        _add_variant(datum["question"])
        for v in paraphrases:
            _add_variant(v)
            if len(variants) >= 7:
                break

        # Pad if fewer than 7 (repeat original)
        while len(variants) < 7:
            variants.append(datum["question"])

        # ---- Few-shot for this example
        fewshot_exps = get_fewshot_exps(data, k=(args.few_shot if args.few_shot > 0 else None), exclude_id=datum.get("id"))

        # ---- Build prompts
        prompts = []
        for qv in variants:
            example_variant = {**datum, "question": qv}
            exp = format_task_prompt(example_variant, args, tokenizer, fewshot_exps=fewshot_exps)
            prompts.append(exp["prompt"])

        # ---- Tokenize as a batch
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=tokenizer.model_max_length
        )
        # track truncation (heuristic)
        if inputs["input_ids"].shape[1] >= tokenizer.model_max_length:
            pbar.set_description(" truncation occurred")

        # move tensors to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # last token logits for next-token prediction
            full_logits = outputs.logits[:, -1, :]                   # (B, V)
            hidden_states = outputs.hidden_states[-1][:, -1, :]      # (B, H)

            # Infer option token ids using the concrete "Answer:" context of each prompt
            # Safer: do it once using the first prompt, since they share the same suffix structure.
            # We just need the 'Answer:' tail; use the raw prompt text.
            base_text = prompts[0].rstrip()
            # ensure we are at the position right after "Answer:"
            # (our prompts end with "...Answer:" and possibly a trailing space; append a space to be safe)
            base_for_ids = base_text + " "
            option_ids = infer_option_token_ids(tokenizer, base_for_ids, options)

            selected_logits = full_logits[:, option_ids]             # (B, 6)

            # to numpy (optional half precision to save space)
            if args.save_float16:
                last_hidden_np = hidden_states.detach().cpu().to(torch.float16).numpy()
                selected_logits_np = selected_logits.detach().cpu().to(torch.float16).numpy()
            else:
                last_hidden_np = hidden_states.detach().cpu().to(torch.float32).numpy()
                selected_logits_np = selected_logits.detach().cpu().to(torch.float32).numpy()

            predictions = []
            for row in selected_logits_np:
                idx = int(np.argmax(row))
                predictions.append(options[idx])

        all_results.append({
            "id": qid,
            "hidden_representations": last_hidden_np,   # shape: (7, H)
            "logits": selected_logits_np,               # shape: (7, 6)
            "predicted_answers": predictions            # length 7
        })

        # checkpoint
        if (i + 1) % args.ckpt_every == 0:
            with open(checkpoint_file, "wb") as f:
                pickle.dump(all_results, f)
            pbar.set_postfix_str(f"saved @ {i+1}")

    # final save
    with open(checkpoint_file, "wb") as f:
        pickle.dump(all_results, f)
    print(f"[Done] {len(data)} examples. Saved to {checkpoint_file}")

    return all_results

# ---------------------------
# CLI
# ---------------------------

def main(args):
    set_seed(args.seed)

    dataset_name = args.dataset_name
    paraphrase_file = os.path.join(args.paraphrase_dir, f"{dataset_name}.json")

    # Load paraphrases
    paraphrased_questions = load_paraphrased_questions(paraphrase_file)

    # Load model + tokenizer
    tokenizer, model = load_model(args)
    try:
        print("Device map:", getattr(model, "hf_device_map", None))
    except Exception:
        pass

    # Load data
    data = load_data(os.path.join(args.data_path, f"{dataset_name}.json"),
                     debug=args.debug, debug_cap=args.debug_cap)

    # Run
    get_hidden_representations(model, tokenizer, data, paraphrased_questions, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-8B",
                        help="HF model id or local path")
    parser.add_argument('--data_path', type=str, default="data",
                        help="Path to dataset directory")
    parser.add_argument('--dataset_name', type=str, default="cosmosqa_10k",
                        help="Dataset name (used to find data/<name>.json and paraphrases)")
    parser.add_argument('--paraphrase_dir', type=str, default="paraphrased_questions",
                        help="Directory with paraphrase JSONs")
    parser.add_argument('--output_dir', type=str, default='outputs_hidden_reps',
                        help="Output directory for .pkl")
    parser.add_argument('--few_shot', type=int, default=2,
                        help="Number of few-shot exemplars to include (0 = zero-shot)")
    parser.add_argument('--cot', action="store_true",
                        help="Use CoT prompt (few_shot must be 0)")
    parser.add_argument('--debug', action="store_true",
                        help="Debug mode: run on a small subset")
    parser.add_argument('--debug_cap', type=int, default=20,
                        help="How many examples to keep in debug mode")
    parser.add_argument('--ckpt_every', type=int, default=500,
                        help="Checkpoint interval")
    parser.add_argument('--save_float16', action="store_true",
                        help="Save arrays as float16 to shrink pickle")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.cot and args.few_shot != 0:
        raise SystemExit("CoT currently supported only with --few_shot 0")

    main(args)
