import os
import json
import pickle
import argparse
import logging
import numpy as np

#############################################
# Utility Functions
#############################################
def build_options(num_options: int):
    """Build option list dynamically based on num_options."""
    assert 2 <= num_options <= 6, "num_options must be in [2,6]"
    return ["A", "B", "C", "D", "E", "F"][:num_options]

def letter_to_index(letter, options):
    """Map option letter to index using dynamic options list."""
    if letter not in options:
        raise ValueError(f"Option {letter} not in allowed {options}")
    return options.index(letter)

def softmax(x, axis=1):
    """Compute softmax in a numerically stable way."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def validate_result_item(res, expected_paraphrase_count, strict):
    """Validate a single result item for required fields and shapes."""
    if "id" not in res or "logits" not in res or "hidden_representations" not in res:
        msg = "Missing 'id', 'logits', or 'hidden_representations' in item."
        if strict:
            raise ValueError(msg)
        logging.warning(msg)
        return False
    
    logits = np.asarray(res["logits"])
    hidden_reps = np.asarray(res["hidden_representations"])
    
    if logits.ndim != 2 or hidden_reps.ndim != 2:
        msg = f"Bad logits/hidden_reps dimensions: {logits.shape}, {hidden_reps.shape}"
        if strict:
            raise ValueError(msg)
        logging.warning(msg)
        return False
    
    if logits.shape[0] != hidden_reps.shape[0]:
        msg = f"Mismatch between logits and hidden_reps first dimension: {logits.shape[0]} vs {hidden_reps.shape[0]}"
        if strict:
            raise ValueError(msg)
        logging.warning(msg)
        return False
    
    if logits.shape[0] < 2:
        msg = f"Need at least 2 rows (original + paraphrases), got {logits.shape[0]}"
        if strict:
            raise ValueError(msg)
        logging.warning(msg)
        return False
    
    if expected_paraphrase_count is not None and logits.shape[0] != 1 + expected_paraphrase_count:
        logging.warning(f"Item {res['id']} has {logits.shape[0]-1} paraphrases; expected {expected_paraphrase_count}. Continuing.")
    
    return True

#############################################
# Variability Score Computation
#############################################
def compute_S_mean(pred_probs, idx):
    """
    Compute the mean paraphrase-aware predictive variability score.
    For each paraphrased distribution: score = 1 - P(y)
    """
    scores = 1.0 - pred_probs[:, idx]
    return float(np.mean(scores))

def compute_S_weighted(pred_probs, hidden_reps, idx):
    """
    Compute the weighted paraphrase-aware predictive variability score.
    Weights are based on the L2 distance between the original hidden rep (row 0)
    and each paraphrase's hidden rep.
    """
    original = hidden_reps[0]  # original question representation
    paraphrase_hidden = hidden_reps[1:]  # shape (P, hidden_dim)
    
    if paraphrase_hidden.shape[0] == 0:
        return float("nan")
    
    distances = np.linalg.norm(paraphrase_hidden - original, axis=1)
    weights = 1.0 / (1.0 + distances)  # higher weight if more similar
    scores = 1.0 - pred_probs[:, idx]
    return float(np.sum(weights * scores) / np.sum(weights))

def compute_S_worst(pred_probs, idx):
    """
    Compute the worst-case paraphrase-aware predictive variability score.
    S_worst(x,y) = max_{paraphrases} (1 - P(y)).
    """
    if pred_probs.shape[0] == 0:
        return float("nan")
    scores = 1.0 - pred_probs[:, idx]
    return float(np.max(scores))

def calculate_paraphrase_variability_scores(results, options, expected_paraphrase_count, strict_validate, debug_n, save_debug_dir):
    """
    Given a list of result dicts (one per question) that include:
      - "id": question id,
      - "hidden_representations": NumPy array of shape (1+P, hidden_dim),
      - "logits": NumPy array of shape (1+P, num_options),
    compute for each question the mean and weighted variability scores for each MCQ option.
    """
    scores = {}
    processed_items = 0
    skipped_items = 0
    
    for i, res in enumerate(results):
        if debug_n > 0 and i >= debug_n:
            break
            
        try:
            # Validate the result item
            if not validate_result_item(res, expected_paraphrase_count, strict_validate):
                skipped_items += 1
                continue
                
            question_id = res["id"]
            logits = res["logits"]  # shape: (1+P, num_options)
            hidden_reps = res["hidden_representations"]  # shape: (1+P, hidden_dim)

            # Convert logits to probabilities using softmax.
            pred_probs_all = softmax(logits, axis=1)
            # Use only the paraphrased variants (rows 1:).
            pred_probs_paraphrase = pred_probs_all[1:]

            s_mean_dict = {}
            s_weighted_dict = {}
            s_worst_dict = {}
            
            for opt in options:
                idx = letter_to_index(opt, options)
                s_mean = compute_S_mean(pred_probs_paraphrase, idx)
                s_weighted = compute_S_weighted(pred_probs_paraphrase, hidden_reps, idx)
                s_worst = compute_S_worst(pred_probs_paraphrase, idx)
                
                # Validate score ranges
                if not (0 <= s_mean <= 1):
                    logging.error(f"Invalid S_mean {s_mean} for {question_id}, option {opt}")
                    if strict_validate:
                        raise ValueError(f"S_mean out of range: {s_mean}")
                    s_mean = float("nan")
                
                if not (0 <= s_weighted <= 1) and not np.isnan(s_weighted):
                    logging.error(f"Invalid S_weighted {s_weighted} for {question_id}, option {opt}")
                    if strict_validate:
                        raise ValueError(f"S_weighted out of range: {s_weighted}")
                    s_weighted = float("nan")
                
                if not (0 <= s_worst <= 1) and not np.isnan(s_worst):
                    logging.error(f"Invalid S_worst {s_worst} for {question_id}, option {opt}")
                    if strict_validate:
                        raise ValueError(f"S_worst out of range: {s_worst}")
                    s_worst = float("nan")
                
                s_mean_dict[opt] = s_mean
                s_weighted_dict[opt] = s_weighted
                s_worst_dict[opt] = s_worst
            
            scores[question_id] = {
                "S_mean": s_mean_dict,
                "S_weighted": s_weighted_dict,
                "S_worst": s_worst_dict
            }
            
            processed_items += 1
            
            # Debug logging
            if debug_n > 0:
                logging.info(f"Item {question_id}: shapes {hidden_reps.shape}, {logits.shape}")
                distances = np.linalg.norm(hidden_reps[1:] - hidden_reps[0], axis=1)
                logging.info(f"  Distances: min={distances.min():.3f}, median={np.median(distances):.3f}, max={distances.max():.3f}")
                logging.info(f"  Sample probs: {pred_probs_paraphrase[0] if len(pred_probs_paraphrase) > 0 else 'N/A'}")
                logging.info(f"  S_mean sample: {list(s_mean_dict.values())[:2]}")
                logging.info(f"  S_weighted sample: {list(s_weighted_dict.values())[:2]}")
                logging.info(f"  S_worst sample: {list(s_worst_dict.values())[:2]}")
            
            # Save debug data if requested
            if save_debug_dir:
                debug_file = os.path.join(save_debug_dir, f"{question_id}_debug.npz")
                np.savez(debug_file,
                        hidden_reps=hidden_reps,
                        logits=logits,
                        probs=pred_probs_all,
                        distances=distances if 'distances' in locals() else np.array([]),
                        scores_mean=np.array(list(s_mean_dict.values())),
                        scores_weighted=np.array(list(s_weighted_dict.values())),
                        scores_worst=np.array(list(s_worst_dict.values())))
                
        except Exception as e:
            logging.error(f"Error processing item {res.get('id', 'unknown')}: {e}")
            if strict_validate:
                raise
            skipped_items += 1
            continue
    
    return scores, processed_items, skipped_items


#############################################
# Main Function: Load LLM Logits, Compute Scores
#############################################
def main():
    parser = argparse.ArgumentParser(
        description="Calculate Paraphrase-aware Predictive Variability Scores using LLM logits directly (ablation study)."
    )
    parser.add_argument("--llm_logits_file", type=str, 
                        default="outputs_hidden_reps/cosmosqa_10k_Qwen3-8B_results.pkl",
                        help="Path to the LLM logits pickle file.")
    parser.add_argument("--output_dir", type=str, default="output_score_llm_ablation",
                        help="Directory to save the variability scores.")
    parser.add_argument("--num_options", type=int, default=6,
                        help="Number of MCQ options (2-6).")
    parser.add_argument("--paraphrase_count", type=int, default=6,
                        help="Expected number of paraphrases per question.")
    parser.add_argument("--debug_n", type=int, default=0,
                        help="If >0, only process the first N items and enable verbose logging.")
    parser.add_argument("--strict_validate", action="store_true",
                        help="Raise errors on validation failures instead of warning.")
    parser.add_argument("--save_debug_dir", type=str, default="",
                        help="Directory to save per-question debug files.")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug_n > 0 else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set random seeds for reproducibility in debug mode
    if args.debug_n > 0:
        np.random.seed(42)

    # Validate arguments
    if not (2 <= args.num_options <= 6):
        raise ValueError("num_options must be between 2 and 6")
    
    if args.paraphrase_count < 0:
        raise ValueError("paraphrase_count must be non-negative")

    # Build options list
    options = build_options(args.num_options)
    logging.info(f"Using options: {options}")

    # --- Load LLM logits and hidden representations ---
    with open(args.llm_logits_file, "rb") as f:
        results = pickle.load(f)
    logging.info(f"Loaded {len(results)} items from {args.llm_logits_file}")

    # --- Calculate variability scores using LLM logits directly ---
    variability_scores, processed_items, skipped_items = calculate_paraphrase_variability_scores(
        results, options, args.paraphrase_count, 
        args.strict_validate, args.debug_n, args.save_debug_dir
    )

    # --- Save scores to the output directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract model name from file path for output naming
    llm_file_name = os.path.basename(args.llm_logits_file)
    model_name = llm_file_name.replace('_results.pkl', '')
    output_file = os.path.join(args.output_dir, f"{model_name}_llm_ablation_variability_scores.json")
    
    with open(output_file, "w") as f:
        json.dump(variability_scores, f, indent=2)
    
    # Create summary file
    summary = {
        "counts": {
            "total_items": len(results),
            "processed_items": processed_items,
            "skipped_items": skipped_items
        },
        "settings": {
            "num_options": args.num_options,
            "paraphrase_count": args.paraphrase_count,
            "llm_model": model_name
        },
        "stats": {}
    }
    
    # Compute basic statistics
    if variability_scores:
        all_s_mean = []
        all_s_weighted = []
        all_s_worst = []
        for item_scores in variability_scores.values():
            all_s_mean.extend(item_scores["S_mean"].values())
            all_s_weighted.extend(item_scores["S_weighted"].values())
            all_s_worst.extend(item_scores["S_worst"].values())
        
        # Filter out NaN values for statistics
        s_mean_valid = [x for x in all_s_mean if not np.isnan(x)]
        s_weighted_valid = [x for x in all_s_weighted if not np.isnan(x)]
        
        if s_mean_valid:
            summary["stats"]["S_mean"] = {
                "min": float(np.min(s_mean_valid)),
                "median": float(np.median(s_mean_valid)),
                "max": float(np.max(s_mean_valid)),
                "mean": float(np.mean(s_mean_valid)),
                "std": float(np.std(s_mean_valid))
            }
        
        if s_weighted_valid:
            summary["stats"]["S_weighted"] = {
                "min": float(np.min(s_weighted_valid)),
                "median": float(np.median(s_weighted_valid)),
                "max": float(np.max(s_weighted_valid)),
                "mean": float(np.mean(s_weighted_valid)),
                "std": float(np.std(s_weighted_valid))
            }
        
        s_worst_valid = [x for x in all_s_worst if not np.isnan(x)]
        if s_worst_valid:
            summary["stats"]["S_worst"] = {
                "min": float(np.min(s_worst_valid)),
                "median": float(np.median(s_worst_valid)),
                "max": float(np.max(s_worst_valid)),
                "mean": float(np.mean(s_worst_valid)),
                "std": float(np.std(s_worst_valid))
            }
    
    summary_file = os.path.join(args.output_dir, f"{model_name}_llm_ablation_variability_scores_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"LLM ablation variability scores saved to {output_file}")
    logging.info(f"Summary saved to {summary_file}")
    logging.info(f"Processed {processed_items} items, skipped {skipped_items} items")

if __name__ == "__main__":
    main()
