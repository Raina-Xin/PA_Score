import json
import logging
import os
import random
import re
from typing import Optional

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    pipeline,
)


META_LEAKAGE_PATTERNS = [
    re.compile(r"rephrase", re.IGNORECASE),
    re.compile(r"rephrased question", re.IGNORECASE),
    re.compile(r"original question", re.IGNORECASE),
    re.compile(r"preserving its meaning", re.IGNORECASE),
    re.compile(r"here'?s the rephrased", re.IGNORECASE),
    re.compile(r"the rephrased question", re.IGNORECASE),
    re.compile(r"maintains the original meaning", re.IGNORECASE),
    re.compile(r"this rephrased question", re.IGNORECASE),
    re.compile(r"to rephrase", re.IGNORECASE),
]

_KNOWN_PREFIX_PATTERNS = [
    re.compile(r"^\s*rephrased question\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*here is the rephrased question\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*here'?s the rephrased question\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*to rephrase the original question\s*,?\s*", re.IGNORECASE),
    re.compile(r"^\s*the rephrased question is\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*rewritten question\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*alternative version\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*sure\s*,?\s*", re.IGNORECASE),
    re.compile(r"^\s*certainly\s*,?\s*", re.IGNORECASE),
]
_LEADING_JUNK_RE = re.compile(r"^[\s\-–—•*>'\"`“”‘’\[\](){}:;,.!?]+")
_FIRST_QUESTION_RE = re.compile(r"(.+?\?)")
_QUESTION_WORD_RE = re.compile(r"\b(what|which|why|how|who|where|when)\b", re.IGNORECASE)

_COMPRESSION_PATTERNS = [
    (r"\bof wavelength\s+(\d+)\s*nanometers\b", r" at \1 nm"),
    (r"\b(\d+)\s*nanometers\b", r"\1 nm"),
    (r"\bindex of refraction\b", "refractive index"),
    (r"\bother than zero\b", "nonzero"),
    (r"\bwhen viewed near normal incidence\b", "at near-normal incidence"),
    (r"\bnear normal incidence\b", "near-normal incidence"),
    (r"\bAssuming that\b", "If"),
    (r"\bWhat is the minimum thickness of the\b", "What's the minimum"),
    (r"\bminimum thickness of the\b", "minimum"),
    (r"\bthin film of ([^?.,]+?) on a ([^?.,]+?)\b", r"\1 film on \2"),
    (r"\bmost strongly reflected off\b", "reflected by"),
    (r"\bWhat is my reason for\b", "Why am I"),
    (r"\bWhat is the reason for\b", "Why"),
    (r"\bWhat is our reason for\b", "Why are we"),
    (r"\bWhat is their reason for\b", "Why are they"),
    (r"\bWhat is his reason for\b", "Why is he"),
    (r"\bWhat is her reason for\b", "Why is she"),
    (r"\bresiding in\b", "living in"),
]


def setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    return logging.getLogger("clean_generation")


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config at {config_path} must parse to a dictionary.")
    return payload


def load_hf_model_and_tokenizer(model_name: str):
    model_name_lower = model_name.lower()
    has_cuda = torch.cuda.is_available()
    device_map = "auto" if has_cuda else None
    half_dtype = torch.float16 if has_cuda else torch.float32
    bf16_dtype = torch.bfloat16 if has_cuda else torch.float32

    if "qwen" in model_name_lower or "internlm" in model_name_lower:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    elif "sus" in model_name_lower or "yi-34b-chat" in model_name_lower:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    try:
        tokenizer.model_max_length = min(
            int(getattr(tokenizer, "model_max_length", 2048)) or 2048,
            2048,
        )
    except Exception:
        tokenizer.model_max_length = 2048

    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif getattr(tokenizer, "unk_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id

    if "falcon" in model_name_lower or "deepseek" in model_name_lower:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=bf16_dtype,
            device_map=device_map,
        )
        if "deepseek" in model_name_lower:
            model.generation_config = GenerationConfig.from_pretrained(model_name)
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
    elif "llama" in model_name_lower:
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=half_dtype,
            device_map=device_map,
        )
    elif "mistral" in model_name_lower or "mpt" in model_name_lower or "mixtral" in model_name_lower:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    elif "yi" in model_name_lower or "sus" in model_name_lower:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype="auto",
        )
    elif "qwen" in model_name_lower:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=half_dtype,
            trust_remote_code=True,
        )
    elif "internlm" in model_name_lower:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=half_dtype,
            trust_remote_code=True,
        )
    else:
        raise NotImplementedError(f"Unsupported model family for clean regeneration: {model_name}")

    if getattr(model.config, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    return model, tokenizer


def build_text_generation_pipeline(model, tokenizer):
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def has_meta_leakage(text: str) -> bool:
    return any(pattern.search(text) for pattern in META_LEAKAGE_PATTERNS)


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").split())


def strip_known_prompt_prefixes(text: str) -> str:
    cleaned = normalize_whitespace(text)
    changed = True
    while changed and cleaned:
        changed = False
        cleaned = _LEADING_JUNK_RE.sub("", cleaned).strip()
        for pattern in _KNOWN_PREFIX_PATTERNS:
            updated = pattern.sub("", cleaned).strip()
            if updated != cleaned:
                cleaned = updated
                changed = True
    return cleaned


def extract_first_question(text: str) -> str:
    for line in str(text or "").splitlines():
        line = normalize_whitespace(line)
        line = _LEADING_JUNK_RE.sub("", line).strip()
        if not line:
            continue
        match = _FIRST_QUESTION_RE.search(line)
        first = match.group(1) if match else line
        first = _LEADING_JUNK_RE.sub("", first).strip()
        return normalize_whitespace(first)
    return ""


def clean_generated_question(text: str) -> str:
    cleaned = normalize_whitespace(text)
    cleaned = strip_known_prompt_prefixes(cleaned)
    cleaned = extract_first_question(cleaned)
    return normalize_whitespace(cleaned)


def _compress_question_text(text: str) -> str:
    compact = normalize_whitespace(text)
    for pattern, replacement in _COMPRESSION_PATTERNS:
        compact = re.sub(pattern, replacement, compact, flags=re.IGNORECASE)
        compact = normalize_whitespace(compact)
    compact = re.sub(r"\s+([,.;:?!])", r"\1", compact)
    compact = re.sub(r"\bthe\b\s+", "", compact, flags=re.IGNORECASE)
    return normalize_whitespace(compact)


def _truncate_question(text: str, max_chars: int = 220) -> str:
    cleaned = normalize_whitespace(text).rstrip("?").strip()
    if not cleaned:
        return "What is the answer?"
    if len(cleaned) + 1 <= max_chars:
        return f"{cleaned}?"

    cut = cleaned[: max_chars - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    cut = cut.rstrip(" ,;:.")
    if not _QUESTION_WORD_RE.search(cut):
        cut = f"What is {cut}".strip()
    return normalize_whitespace(f"{cut}?")


def _ordered_unique(values):
    seen = set()
    ordered = []
    for value in values:
        normalized = normalize_whitespace(value)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


def validate_candidate_question(candidate: str, source_question: str) -> tuple[bool, str]:
    candidate = normalize_whitespace(candidate)
    source_question = normalize_whitespace(source_question)
    max_candidate_len = max(220, min(len(source_question) + 20, 400))

    if not candidate:
        return False, "empty"
    if len(candidate) < 6:
        return False, "too_short"
    if len(candidate) > max_candidate_len:
        return False, "too_long"
    if "```" in candidate:
        return False, "code_block"
    if has_meta_leakage(candidate):
        return False, "meta_leakage"
    if candidate.casefold() == source_question.casefold():
        return False, "identical"
    if not candidate.endswith("?"):
        return False, "not_question"
    return True, "ok"


def trivial_question_fallback(question: str) -> str:
    source = normalize_whitespace(question)
    out = source.rstrip("?").strip()
    replacements = [
        (r"\bWhat is\b", "What's"),
        (r"\bWhich of the following\b", "Which option"),
        (r"\bcan you\b", "could you"),
        (r"\bplease\b", ""),
        (r"\bIn which\b", "Where"),
    ]
    for pattern, replacement in replacements:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    out = normalize_whitespace(out).rstrip(".").rstrip("?").strip()
    if not out:
        return "Could you restate the question?"
    out = f"{out}?"
    if out.casefold() == normalize_whitespace(question).casefold():
        lowered = out[:-1].strip()
        if lowered:
            lowered = lowered[0].lower() + lowered[1:]
        out = f"Could you tell me {lowered}?"
    candidates = [
        out,
        _compress_question_text(out),
        _compress_question_text(source),
        re.sub(r"\bWhat is\b", "What's", source, flags=re.IGNORECASE),
        re.sub(r"\bWhat is the reason for\b", "Why", source, flags=re.IGNORECASE),
        re.sub(r"\bWhat is my reason for\b", "Why am I", source, flags=re.IGNORECASE),
    ]

    for candidate in _ordered_unique(candidates):
        candidate = clean_generated_question(candidate)
        candidate = _compress_question_text(candidate)
        if not candidate.endswith("?"):
            candidate = candidate.rstrip(".").rstrip("?").strip() + "?"
        candidate = _truncate_question(candidate, max_chars=220)
        candidate = clean_generated_question(candidate)
        if not candidate.endswith("?"):
            candidate = candidate.rstrip(".").rstrip("?").strip() + "?"
        if candidate and candidate.casefold() != source.casefold():
            return normalize_whitespace(candidate)

    shortened = _truncate_question(_compress_question_text(source), max_chars=220)
    shortened = clean_generated_question(shortened)
    if shortened.casefold() == source.casefold():
        shortened = re.sub(r"\bWhat is\b", "What's", shortened, flags=re.IGNORECASE)
        shortened = _truncate_question(shortened, max_chars=220)
    return normalize_whitespace(shortened)


def deterministic_question_variants(question: str):
    source = normalize_whitespace(question).rstrip("?").strip()
    compressed = _compress_question_text(source)
    variants = [
        compressed,
        re.sub(r"\bWhat is\b", "What's", compressed, flags=re.IGNORECASE),
        re.sub(r"\bWhat is the reason for\b", "Why", compressed, flags=re.IGNORECASE),
        re.sub(r"\bWhat is my reason for\b", "Why am I", compressed, flags=re.IGNORECASE),
        re.sub(r"\bWhat is our reason for\b", "Why are we", compressed, flags=re.IGNORECASE),
        re.sub(r"\bWhat is their reason for\b", "Why are they", compressed, flags=re.IGNORECASE),
        re.sub(r"\bWhat is his reason for\b", "Why is he", compressed, flags=re.IGNORECASE),
        re.sub(r"\bWhat is her reason for\b", "Why is she", compressed, flags=re.IGNORECASE),
        re.sub(r"\bresiding in\b", "living in", compressed, flags=re.IGNORECASE),
        re.sub(r"\bliving in\b", "staying in", compressed, flags=re.IGNORECASE),
        re.sub(r"\bresiding in\b", "being in", compressed, flags=re.IGNORECASE),
        re.sub(r"\bWhich of the following\b", "Which option", compressed, flags=re.IGNORECASE),
        re.sub(r"\bCan you\b", "Could you", compressed, flags=re.IGNORECASE),
        re.sub(r"\bCould you\b", "Can you", compressed, flags=re.IGNORECASE),
    ]

    if source.lower().startswith("why am i "):
        tail = source[9:].strip()
        variants.extend(
            [
                f"What's my reason for {tail}",
                f"For what reason am I {tail}",
                f"What explains why I am {tail}",
                f"Why am I currently {tail}",
            ]
        )
    elif source.lower().startswith("what is my reason for "):
        tail = source[22:].strip()
        variants.extend(
            [
                f"Why am I {tail}",
                f"For what reason am I {tail}",
                f"What is the reason I am {tail}",
                f"What explains my {tail}",
            ]
        )
    elif source.lower().startswith("what is the reason for "):
        tail = source[23:].strip()
        variants.extend(
            [
                f"Why {tail}",
                f"For what reason {tail}",
                f"What explains {tail}",
            ]
        )
    elif source.lower().startswith("what brought you to "):
        tail = source[20:].strip()
        variants.extend(
            [
                f"Why did you go to {tail}",
                f"What made you go to {tail}",
                f"What caused you to go to {tail}",
                f"What led you to {tail}",
                f"What was the reason you went to {tail}",
                f"Why were you at {tail}",
            ]
        )
    elif source.lower().startswith("what brought me to "):
        tail = source[19:].strip()
        variants.extend(
            [
                f"Why did I go to {tail}",
                f"What made me go to {tail}",
                f"What caused me to go to {tail}",
                f"What led me to {tail}",
                f"What was the reason I went to {tail}",
                f"Why was I at {tail}",
            ]
        )
    elif source.lower().startswith("what brought him to "):
        tail = source[20:].strip()
        variants.extend(
            [
                f"Why did he go to {tail}",
                f"What made him go to {tail}",
                f"What caused him to go to {tail}",
                f"What led him to {tail}",
            ]
        )
    elif source.lower().startswith("what brought her to "):
        tail = source[20:].strip()
        variants.extend(
            [
                f"Why did she go to {tail}",
                f"What made her go to {tail}",
                f"What caused her to go to {tail}",
                f"What led her to {tail}",
            ]
        )

    can_you_match = re.match(r"^Can you (.+)$", source, flags=re.IGNORECASE)
    if can_you_match:
        tail = can_you_match.group(1).strip()
        variants.extend(
            [
                f"Could you {tail}",
                f"Are you able to {tail}",
                f"Would you be able to {tail}",
                f"Can {tail} be done",
            ]
        )
        if tail.lower().endswith(" easily"):
            base = tail[:-7].strip()
            variants.extend(
                [
                    f"Is it easy to {base}",
                    f"Can you {base} with ease",
                    f"Can you {base} without difficulty",
                    f"Is {base} easy",
                    f"Would it be easy to {base}",
                ]
            )

    easy_match = re.match(r"^Is it easy to (.+)$", source, flags=re.IGNORECASE)
    if easy_match:
        tail = easy_match.group(1).strip()
        variants.extend(
            [
                f"Can you {tail} easily",
                f"Could you {tail} easily",
                f"Would it be easy to {tail}",
                f"Is {tail} easy",
                f"Can {tail} be done easily",
                f"Are you able to {tail} easily",
            ]
        )

    wrapped_base = _compress_question_text(source).rstrip("?").strip()
    variants.extend(
        [
            f"In other words, {wrapped_base}",
            f"Put differently, {wrapped_base}",
            f"Asked another way, {wrapped_base}",
            f"Said differently, {wrapped_base}",
            f"Stated differently, {wrapped_base}",
        ]
    )

    for candidate in _ordered_unique(variants):
        candidate = clean_generated_question(candidate)
        candidate = _compress_question_text(candidate)
        if not candidate.endswith("?"):
            candidate = candidate.rstrip(".").rstrip("?").strip() + "?"
        candidate = _truncate_question(candidate, max_chars=220)
        yield normalize_whitespace(candidate)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json_if_exists(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, found {type(payload).__name__}")
    return payload


def save_json_atomic(obj, path: str):
    directory = os.path.dirname(path)
    if directory:
        ensure_dir(directory)
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


def update_reason_counter(counter: dict, reason: str):
    counter[reason] = counter.get(reason, 0) + 1
