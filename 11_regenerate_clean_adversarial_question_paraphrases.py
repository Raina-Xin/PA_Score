#!/usr/bin/env python3
import argparse
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
        "Rewrite the question below as ONE alternative clean question that preserves meaning.\n"
        "Output ONLY the rewritten question.\n"
        "No explanation. No preface. No quotes. No bullet points.\n"
        "The output must end with a question mark.\n\n"
        "Question: {question}\n"
        "Rewritten question:"
    ),
    (
        "Write one clean paraphrase of the question below.\n"
        "Preserve meaning.\n"
        "Return only the rewritten question.\n"
        "No preface. No explanation. No quotes. No bullets.\n"
        "It must end with a question mark.\n\n"
        "Question: {question}\n"
        "Rewritten question:"
    ),
    (
        "Produce one alternative wording of the question below.\n"
        "Keep the same meaning.\n"
        "Output only the question text.\n"
        "No commentary, no quotes, no bullets.\n"
        "End with a question mark.\n\n"
        "Question: {question}\n"
        "Rewritten question:"
    ),
    (
        "Return one clean rewritten version of the question below.\n"
        "Preserve the meaning.\n"
        "Output only the rewritten question.\n"
        "No preface. No explanation. No quotes. No bullets.\n"
        "The result must end with a question mark.\n\n"
        "Question: {question}\n"
        "Rewritten question:"
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate cleaned adversarial question paraphrases into a standalone output root.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--n_perturb", type=int, default=None, help="Override n_perturb from config (number of paraphrases per item).")
    parser.add_argument("--max_retries_per_item", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)
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


def _load_input_questions(input_root: str, dataset: str, debug_limit: int | None):
    input_path = Path(input_root) / "adversarial_questions" / f"{dataset}.json"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Regenerated adversarial question file not found: {input_path}. "
            "Run 10_regenerate_clean_adversarial_questions.py first or point --input_root at an existing cleaned artifact root."
        )
    payload = load_json_if_exists(str(input_path))
    items = sorted(payload.items(), key=lambda kv: int(kv[0]))
    if debug_limit is not None:
        items = items[:debug_limit]

    records = []
    for item_id, values in items:
        question = ""
        if isinstance(values, list) and values:
            question = clean_generated_question(values[0])
        if not question:
            continue
        records.append({"id": str(item_id), "question": question})
    if not records:
        raise ValueError(f"No usable regenerated adversarial questions found in {input_path}")
    return records


def _fallback_variants(question: str):
    variants = [trivial_question_fallback(question)]
    variants.extend(deterministic_question_variants(question))
    for candidate in variants:
        yield clean_generated_question(candidate)


def _compute_final_stats(records, output_map, target_n_perturb: int):
    finals = []
    sample_final_examples = []
    for record in records:
        values = output_map.get(record["id"], [])
        cleaned_values = [normalize_whitespace(value) for value in values if normalize_whitespace(value)]
        finals.extend(cleaned_values)
        if len(sample_final_examples) < 50 and cleaned_values:
            sample_final_examples.append(
                {
                    "id": record["id"],
                    "input_adv_question": record["question"],
                    "final_paraphrases": cleaned_values,
                }
            )

    count = len(finals)
    meta_count = sum(1 for text in finals if has_meta_leakage(text))
    question_count = sum(1 for text in finals if text.endswith("?"))
    avg_char_len = (sum(len(text) for text in finals) / count) if count else 0.0
    return {
        "completed_items": sum(
            1 for record in records if len(output_map.get(record["id"], [])) == target_n_perturb
        ),
        "avg_final_paraphrases_per_item": (
            sum(len(output_map.get(record["id"], [])) for record in records) / len(records)
            if records
            else 0.0
        ),
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
    target_n_perturb = int(args.n_perturb) if args.n_perturb is not None else int(config.get("n_perturb") or 6)

    records = _load_input_questions(args.input_root, args.dataset, args.debug_limit)
    total_items = len(records)

    output_root = Path(args.output_root)
    output_dir = output_root / "adversarial_question_paraphrases"
    report_dir = output_root / "reports"
    ensure_dir(str(output_dir))
    ensure_dir(str(report_dir))

    output_path = output_dir / f"{args.dataset}.json"
    report_path = report_dir / f"{args.dataset}_adv_question_paraphrase_report.json"

    existing_output = {str(key): value for key, value in load_json_if_exists(str(output_path)).items()}
    output_map = {}
    pending_records = []
    for record in records:
        seen = set()
        cleaned_values = []
        for value in existing_output.get(record["id"], []):
            cleaned = clean_generated_question(value)
            valid, _ = validate_candidate_question(cleaned, record["question"])
            if not valid:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned_values.append(cleaned)
            if len(cleaned_values) >= target_n_perturb:
                break
        output_map[record["id"]] = cleaned_values
        if len(cleaned_values) < target_n_perturb:
            pending_records.append(record)

    logger.info("Loaded %s total items; %s complete, %s pending.", total_items, total_items - len(pending_records), len(pending_records))

    rejection_reason_counts = {}
    sample_invalid_examples = []
    total_retry_events = 0
    fallback_fill_count = 0
    processed_since_save = 0

    model, tokenizer = load_hf_model_and_tokenizer(args.model)
    generator = build_text_generation_pipeline(model, tokenizer)

    for batch_start in range(0, len(pending_records), batch_size):
        batch_records = pending_records[batch_start: batch_start + batch_size]
        retry_counts = {record["id"]: 0 for record in batch_records}

        for attempt in range(args.max_retries_per_item):
            active_records = [
                record for record in batch_records if len(output_map.get(record["id"], [])) < target_n_perturb
            ]
            if not active_records:
                break

            prompt_template = PROMPT_VARIANTS[attempt % len(PROMPT_VARIANTS)]
            prompts = [prompt_template.format(question=record["question"]) for record in active_records]
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

            for record, raw_candidate in zip(active_records, raw_texts):
                cleaned_candidate = clean_generated_question(raw_candidate)
                is_valid, reason = validate_candidate_question(cleaned_candidate, record["question"])
                if is_valid:
                    seen = {value.casefold() for value in output_map.get(record["id"], [])}
                    if cleaned_candidate.casefold() in seen:
                        reason = "duplicate"
                        is_valid = False

                if is_valid:
                    output_map.setdefault(record["id"], []).append(cleaned_candidate)
                else:
                    retry_counts[record["id"]] += 1
                    total_retry_events += 1
                    update_reason_counter(rejection_reason_counts, reason)
                    if len(sample_invalid_examples) < 50:
                        sample_invalid_examples.append(
                            {
                                "id": record["id"],
                                "input_adv_question": record["question"],
                                "raw_candidate": normalize_whitespace(raw_candidate),
                                "cleaned_candidate": cleaned_candidate,
                                "reason": reason,
                            }
                        )

        for record in batch_records:
            existing = output_map.get(record["id"], [])
            seen = {value.casefold() for value in existing}
            if len(existing) < target_n_perturb:
                for fallback_candidate in _fallback_variants(record["question"]):
                    valid, reason = validate_candidate_question(fallback_candidate, record["question"])
                    if not valid:
                        continue
                    key = fallback_candidate.casefold()
                    if key in seen:
                        continue
                    existing.append(fallback_candidate)
                    seen.add(key)
                    fallback_fill_count += 1
                    if len(existing) >= target_n_perturb:
                        break

                if len(existing) < target_n_perturb:
                    raise ValueError(
                        f"Unable to fill {target_n_perturb} clean paraphrases for id={record['id']} after retries and deterministic fallback."
                    )
            output_map[record["id"]] = existing[:target_n_perturb]
            processed_since_save += 1

        if processed_since_save >= args.save_every:
            save_json_atomic(output_map, str(output_path))
            processed_since_save = 0

    save_json_atomic(output_map, str(output_path))

    final_stats = _compute_final_stats(records, output_map, target_n_perturb)
    report = {
        "dataset": args.dataset,
        "model": args.model,
        "total_items": total_items,
        "completed_items": final_stats["completed_items"],
        "target_n_perturb": target_n_perturb,
        "avg_final_paraphrases_per_item": final_stats["avg_final_paraphrases_per_item"],
        "avg_retries_per_item": (total_retry_events / total_items) if total_items else 0.0,
        "fallback_fill_count": fallback_fill_count,
        "rejection_reason_counts": rejection_reason_counts,
        "meta_leakage_rate_final": final_stats["meta_leakage_rate_final"],
        "question_mark_rate_final": final_stats["question_mark_rate_final"],
        "avg_char_len_final": final_stats["avg_char_len_final"],
        "sample_invalid_examples": sample_invalid_examples,
        "sample_final_examples": final_stats["sample_final_examples"],
    }
    save_json_atomic(report, str(report_path))
    logger.info("Wrote cleaned adversarial paraphrases to %s", output_path)
    logger.info("Wrote report to %s", report_path)


if __name__ == "__main__":
    main()
