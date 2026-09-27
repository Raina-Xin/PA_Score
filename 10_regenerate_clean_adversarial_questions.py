#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from clean_generation_utils import (
    build_text_generation_pipeline,
    clean_generated_question,
    deterministic_question_variants,
    ensure_dir,
    has_meta_leakage,
    load_hf_model_and_tokenizer,
    load_json_if_exists,
    load_yaml_config,
    normalize_whitespace,
    save_json_atomic,
    set_seed,
    setup_logging,
    trivial_question_fallback,
    update_reason_counter,
    validate_candidate_question,
)


PROMPT_VARIANTS = [
    (
        "Rewrite the following question as ONE clean question with different wording while preserving meaning.\n"
        "Output ONLY the rewritten question.\n"
        "No explanation. No preface. No quotes. No bullet points.\n"
        "The output must end with a question mark.\n\n"
        "Original question: {question}\n"
        "Rewritten question:"
    ),
    (
        "Rewrite the question below as exactly one clean rewritten question.\n"
        "Preserve meaning.\n"
        "Output only the question.\n"
        "No preface. No explanation. No quotes. No list markers.\n"
        "It must end with a question mark.\n\n"
        "Question: {question}\n"
        "Rewritten question:"
    ),
    (
        "Produce one rewritten version of the question below.\n"
        "Keep the same meaning.\n"
        "Return only the rewritten question.\n"
        "No commentary, no quotes, no bullets.\n"
        "End with a question mark.\n\n"
        "Original question: {question}\n"
        "Rewritten question:"
    ),
    (
        "Write one alternative wording of the question below.\n"
        "Preserve the meaning.\n"
        "Output only the question text.\n"
        "No preface. No explanation. No quotes. No bullets.\n"
        "The result must end with a question mark.\n\n"
        "Question: {question}\n"
        "Rewritten question:"
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate cleaned adversarial questions into a standalone output root.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--max_retries_per_item", type=int, default=6)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--debug_limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _extract_generated_texts(outputs):
    if isinstance(outputs, list):
        extracted = []
        for item in outputs:
            if isinstance(item, list):
                item = item[0] if item else {}
            extracted.append((item or {}).get("generated_text", ""))
        return extracted
    return []


def _load_dataset_records(data_path: str, dataset: str, debug_limit: int | None):
    data_file = Path(data_path) / f"{dataset}.json"
    with open(data_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Dataset file must contain a list: {data_file}")
    if debug_limit is not None:
        payload = payload[:debug_limit]

    records = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(f"Dataset entry {index} in {data_file} is not an object.")
        question = normalize_whitespace(entry.get("question", ""))
        if not question:
            continue
        item_id = str(entry.get("id", index))
        records.append({"id": item_id, "question": question})
    return records


def _compute_final_stats(records, output_map):
    finals = []
    sample_final_examples = []
    for record in records:
        values = output_map.get(record["id"], [])
        if not values:
            continue
        final_question = normalize_whitespace(values[0])
        finals.append(final_question)
        if len(sample_final_examples) < 50:
            sample_final_examples.append(
                {
                    "id": record["id"],
                    "source_question": record["question"],
                    "final_question": final_question,
                }
            )

    count = len(finals)
    meta_count = sum(1 for text in finals if has_meta_leakage(text))
    question_count = sum(1 for text in finals if text.endswith("?"))
    avg_char_len = (sum(len(text) for text in finals) / count) if count else 0.0
    return {
        "completed_items": count,
        "meta_leakage_rate_final": (meta_count / count) if count else 0.0,
        "question_mark_rate_final": (question_count / count) if count else 0.0,
        "avg_char_len_final": avg_char_len,
        "sample_final_examples": sample_final_examples,
    }


def main():
    args = parse_args()
    logger = setup_logging(args.verbose)
    set_seed(args.seed)

    config = load_yaml_config(args.config_path)
    batch_size = int(args.batch_size or config.get("batch_size") or 8)
    max_new_tokens = int(args.max_new_tokens or config.get("max_tokens") or 64)
    temperature = float(config.get("temperature") or 1.0)

    records = _load_dataset_records(args.data_path, args.dataset, args.debug_limit)
    total_items = len(records)

    output_root = Path(args.output_root)
    output_dir = output_root / "adversarial_questions"
    report_dir = output_root / "reports"
    ensure_dir(str(output_dir))
    ensure_dir(str(report_dir))

    output_path = output_dir / f"{args.dataset}.json"
    report_path = report_dir / f"{args.dataset}_adv_question_report.json"

    output_map = {str(key): value for key, value in load_json_if_exists(str(output_path)).items()}
    rejection_reason_counts = {}
    sample_invalid_examples = []
    used_fallback_count = 0
    total_retry_events = 0
    processed_since_save = 0

    pending_records = []
    for record in records:
        existing = output_map.get(record["id"], [])
        candidate = ""
        if isinstance(existing, list) and existing:
            candidate = clean_generated_question(existing[0])
        valid, _ = validate_candidate_question(candidate, record["question"])
        if valid:
            output_map[record["id"]] = [candidate]
        else:
            pending_records.append(record)

    logger.info("Loaded %s total items; %s already valid, %s pending.", total_items, total_items - len(pending_records), len(pending_records))

    model, tokenizer = load_hf_model_and_tokenizer(args.model)
    generator = build_text_generation_pipeline(model, tokenizer)

    for batch_start in range(0, len(pending_records), batch_size):
        batch_records = pending_records[batch_start: batch_start + batch_size]
        pending_batch = list(batch_records)
        retry_counts = {record["id"]: 0 for record in batch_records}
        final_batch = {}

        for attempt in range(args.max_retries_per_item):
            if not pending_batch:
                break

            prompt_template = PROMPT_VARIANTS[attempt % len(PROMPT_VARIANTS)]
            prompts = [prompt_template.format(question=record["question"]) for record in pending_batch]
            raw_outputs = generator(
                prompts,
                batch_size=min(batch_size, len(prompts)),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                num_return_sequences=1,
                return_full_text=False,
            )
            raw_texts = _extract_generated_texts(raw_outputs)

            next_pending = []
            for record, raw_candidate in zip(pending_batch, raw_texts):
                cleaned_candidate = clean_generated_question(raw_candidate)
                is_valid, reason = validate_candidate_question(cleaned_candidate, record["question"])
                if is_valid:
                    final_batch[record["id"]] = [cleaned_candidate]
                else:
                    retry_counts[record["id"]] += 1
                    total_retry_events += 1
                    update_reason_counter(rejection_reason_counts, reason)
                    if len(sample_invalid_examples) < 50:
                        sample_invalid_examples.append(
                            {
                                "id": record["id"],
                                "source_question": record["question"],
                                "raw_candidate": normalize_whitespace(raw_candidate),
                                "cleaned_candidate": cleaned_candidate,
                                "reason": reason,
                            }
                        )
                    next_pending.append(record)
            pending_batch = next_pending

        for record in pending_batch:
            cleaned_fallback = ""
            last_reason = "empty"
            fallback_candidates = [trivial_question_fallback(record["question"])]
            fallback_candidates.extend(deterministic_question_variants(record["question"]))
            for fallback in fallback_candidates:
                candidate = clean_generated_question(fallback)
                is_valid, reason = validate_candidate_question(candidate, record["question"])
                if is_valid:
                    cleaned_fallback = candidate
                    break
                last_reason = reason
            if not cleaned_fallback:
                raise ValueError(f"Fallback failed for id={record['id']} with reason={last_reason}")
            final_batch[record["id"]] = [cleaned_fallback]
            used_fallback_count += 1

        for record in batch_records:
            output_map[record["id"]] = final_batch[record["id"]]
            processed_since_save += 1

        if processed_since_save >= args.save_every:
            save_json_atomic(output_map, str(output_path))
            processed_since_save = 0

    save_json_atomic(output_map, str(output_path))

    final_stats = _compute_final_stats(records, output_map)
    report = {
        "dataset": args.dataset,
        "model": args.model,
        "total_items": total_items,
        "completed_items": final_stats["completed_items"],
        "used_fallback_count": used_fallback_count,
        "avg_retries_per_item": (total_retry_events / total_items) if total_items else 0.0,
        "rejection_reason_counts": rejection_reason_counts,
        "meta_leakage_rate_final": final_stats["meta_leakage_rate_final"],
        "question_mark_rate_final": final_stats["question_mark_rate_final"],
        "avg_char_len_final": final_stats["avg_char_len_final"],
        "sample_invalid_examples": sample_invalid_examples,
        "sample_final_examples": final_stats["sample_final_examples"],
    }
    save_json_atomic(report, str(report_path))
    logger.info("Wrote cleaned adversarial questions to %s", output_path)
    logger.info("Wrote report to %s", report_path)


if __name__ == "__main__":
    main()
