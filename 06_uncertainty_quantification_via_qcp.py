#!/usr/bin/env python3
import pickle
import json
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

from conditionalconformal import CondConf
from qcp.embeddings_and_cluster import (
    load_or_compute_embeddings,
    extract_question_text,
    fit_kmeans,
    auto_select_k,
    assign_clusters,
    build_phi_features,
)

import argparse

# =========================
# Constants & Small Utils
# =========================

options = ["A", "B", "C", "D", "E", "F"]
#ids_to_remove = [1, 3, 5, 7, 9]  # remove data points used as demonstrations, by record ID
ids_to_remove = []


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for 1D arrays."""
    x = np.asarray(x, dtype=float).reshape(-1)
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


# ----- Phi functions for non-cluster modes -----

def phi_intercept(x: np.ndarray) -> np.ndarray:
    """Phi(x) = [1]."""
    return np.ones((x.shape[0], 1))

# Think of the phi function as a way to cluster questions together for CCP
def phi_identity(x: np.ndarray) -> np.ndarray:
    """Phi(x) = x."""
    return x


def phi_stats(x: np.ndarray) -> np.ndarray:
    """Phi(x) = [1, max_p, entropy, margin]."""
    x = np.asarray(x, dtype=float)
    x_clipped = np.clip(x, 1e-12, 1.0)
    max_p = np.max(x_clipped, axis=1, keepdims=True)
    sortd = np.sort(x_clipped, axis=1)[:, ::-1]
    margin = (sortd[:, [0]] - sortd[:, [1]])
    entropy = -(x_clipped * np.log(x_clipped)).sum(axis=1, keepdims=True)
    return np.concatenate([np.ones_like(max_p), max_p, entropy, margin], axis=1)


def get_phi(mode: str):
    if mode == "identity":
        return phi_identity
    elif mode == "stats":
        return phi_stats
    else:
        return phi_intercept


# =========================
# Paraphrase-aware score utils
# =========================

def load_paraphrase_scores(pa_score_file: str, metric_key: str) -> Dict[str, Dict[str, float]]:
    """Load paraphrase-aware scores JSON and return mapping id -> option->score for the chosen metric.

    Parameters
    ----------
    pa_score_file: path to JSON produced by 05_calculate_paraphrase_aware_score.py
    metric_key: "S_mean" or "S_weighted"
    """
    if not os.path.exists(pa_score_file):
        raise FileNotFoundError(f"Paraphrase-aware score file not found: {pa_score_file}")
    with open(pa_score_file, "r") as f:
        data = json.load(f)

    out: Dict[str, Dict[str, float]] = {}
    for qid, entry in data.items():
        # ids could be int in json; normalize to str
        sid = str(qid)
        if metric_key not in entry:
            raise ValueError(f"Metric {metric_key} missing for id {sid} in {pa_score_file}")
        # Ensure option keys are uppercase letters
        opt_to_score = {str(k): float(v) for k, v in entry[metric_key].items()}
        out[sid] = opt_to_score
    return out


# =========================
# SBERT-clustered Phi
# =========================

def phi_cluster_sbert(
    dataset_name: str,
    raw_data: List[Dict[str, Any]],
    cal_ids: Set[str],
    test_ids: Set[str],
    question_key: str,
    context_key: str,
    sbert_model: str,
    embed_cache_dir: str,
    clusters: int,
    cluster_selection: str,
    phi_rep: str,
    normalize_embeddings: bool,
    kmeans_minibatch: bool,
    seed: int,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Build cluster-based phi features using SBERT embeddings.
    Returns (phi_cal, phi_test, cluster_metadata).
    """
    # Split raw data to consistent order
    cal_raw_data = [r for r in raw_data if str(r["id"]) in cal_ids]
    test_raw_data = [r for r in raw_data if str(r["id"]) in test_ids]

    # Build texts by combining context and question when available
    def build_texts(rows: List[Dict[str, Any]]) -> List[str]:
        texts: List[str] = []
        for r in rows:
            q = str(r.get(question_key, ""))
            ctx = str(r.get(context_key, ""))
            if ctx:
                texts.append(f"{ctx} [SEP] {q}")
            else:
                texts.append(q)
        return texts

    cal_texts = build_texts(cal_raw_data)
    test_texts = build_texts(test_raw_data)

    cal_ids_list = [str(r["id"]) for r in cal_raw_data]
    test_ids_list = [str(r["id"]) for r in test_raw_data]
    all_ids = cal_ids_list + test_ids_list
    all_texts = cal_texts + test_texts

    # Compute or load embeddings (single cache keyed by dataset + model)
    embeddings, meta = load_or_compute_embeddings(
        f"{dataset_name}",
        all_ids,
        all_texts,
        sbert_model,
        embed_cache_dir,
        normalize_embeddings,
        force_recompute=False,
    )

    # Split embeddings
    n_cal = len(cal_texts)
    emb_cal = embeddings[:n_cal]
    emb_test = embeddings[n_cal:]

    if verbose:
        print(f"[SBERT] emb_cal: {emb_cal.shape}, emb_test: {emb_test.shape}")

    # Choose K
    if cluster_selection == "auto":
        candidates = [8, 12, 16, 20, 24, 32]
        K = auto_select_k(emb_cal, candidates, seed, kmeans_minibatch)
    else:
        K = clusters

    # Fit KMeans on calibration only
    centroids, kmeans_model = fit_kmeans(emb_cal, K, seed, kmeans_minibatch)

    # Assign clusters
    cal_labels, cal_distances = assign_clusters(emb_cal, centroids)
    test_labels, test_distances = assign_clusters(emb_test, centroids)

    # Build Φ features
    phi_cal = build_phi_features(cal_labels, cal_distances, phi_rep, K)
    phi_test = build_phi_features(test_labels, test_distances, phi_rep, K)

    # Metadata for analysis
    cluster_metadata: Dict[str, Any] = {
        "K": int(K),
        "cluster_selection": cluster_selection,
        "phi_rep": phi_rep,
        "sbert_model": sbert_model,
        "normalize_embeddings": bool(normalize_embeddings),
        "cal_cluster_sizes": [int(np.sum(cal_labels == i)) for i in range(K)],
        "test_cluster_sizes": [int(np.sum(test_labels == i)) for i in range(K)],
        "cal_ids": cal_ids_list,
        "test_ids": test_ids_list,
        "centroids_shape": list(centroids.shape),
    }
    # include centroids & labels so caller can save them if desired
    cluster_metadata["centroids"] = centroids.tolist()
    cluster_metadata["cal_labels"] = cal_labels.astype(int).tolist()
    cluster_metadata["test_labels"] = test_labels.astype(int).tolist()

    if verbose:
        print(f"[SBERT] Chosen K={K}")
        print(f"[SBERT] cal sizes: {cluster_metadata['cal_cluster_sizes']}")
        print(f"[SBERT] test sizes: {cluster_metadata['test_cluster_sizes']}")

    return phi_cal, phi_test, cluster_metadata


# =========================
# Data Loading & Alignment
# =========================

def split_ids_by_raw(raw_data: List[Dict[str, Any]], cal_ratio: float, seed: int) -> Tuple[Set[str], Set[str]]:
    ids = [str(r["id"]) for r in raw_data]
    cal_ids, test_ids = train_test_split(ids, train_size=cal_ratio, random_state=seed, shuffle=True)
    return set(cal_ids), set(test_ids)


def index_logits_by_id(logits_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    d: Dict[str, Dict[str, Any]] = {}
    for row in logits_list:
        rid = str(row["id"])
        if rid in d:
            raise ValueError(f"Duplicate logit id {rid}")
        d[rid] = row
    return d


def get_raw_data(
    raw_data_dir: str,
    data_name: str,
    cal_ratio: float,
    debug: bool = False,
    seed: int = 42,
    calib_ids_override: Optional[Set[str]] = None,
    test_ids_override: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Set[str], Set[str]]:
    """Load raw JSON, remove demo IDs by record ID. If overrides are provided, use them;
    otherwise split by cal_ratio. Return cal/test lists and ID sets.
    """
    raw_data_path = os.path.join(raw_data_dir, data_name + ".json")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")

    with open(raw_data_path, "r") as f:
        raw_data = json.load(f)

    # remove by record ID
    raw_data = [item for item in raw_data if item.get("id") not in ids_to_remove]

    if debug:
        raw_data = raw_data[:200]

    if calib_ids_override is not None and test_ids_override is not None:
        cal_ids = set(str(x) for x in calib_ids_override)
        test_ids = set(str(x) for x in test_ids_override)
    else:
        cal_ids, test_ids = split_ids_by_raw(raw_data, cal_ratio, seed)

    cal_raw_data = [r for r in raw_data if str(r["id"]) in cal_ids]
    test_raw_data = [r for r in raw_data if str(r["id"]) in test_ids]

    if debug:
        cal_raw_data = cal_raw_data[:50]
        test_raw_data = test_raw_data[:50]
        cal_ids = {str(r["id"]) for r in cal_raw_data}
        test_ids = {str(r["id"]) for r in test_raw_data}

    print(f"Dataset: {data_name} | Total: {len(raw_data)} | Calib: {len(cal_raw_data)} | Test: {len(test_raw_data)}")
    return cal_raw_data, test_raw_data, cal_ids, test_ids


def load_splits_from_manifest(manifest_path: str) -> Tuple[Set[str], Set[str]]:
    """Load calibration and test ID sets from a split manifest JSON produced by 03_generate_calib_test_split.py."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    split_files = manifest.get("split_files", {})
    calib_path = split_files.get("calib")
    test_path = split_files.get("test")
    if not calib_path or not test_path:
        raise KeyError("Manifest missing 'calib' or 'test' entries in 'split_files'")
    with open(calib_path, "r") as f:
        calib_ids_list = json.load(f)
    with open(test_path, "r") as f:
        test_ids_list = json.load(f)
    calib_ids = {str(x) for x in calib_ids_list}
    test_ids = {str(x) for x in test_ids_list}
    return calib_ids, test_ids


def get_logits_data(
    model_name: str,
    data_name: str,
    cal_raw_data: List[Dict[str, Any]],
    test_raw_data: List[Dict[str, Any]],
    cal_ids: Set[str],
    test_ids: Set[str],
    hidden_reps_dir: str,
    prompt_methods: List[str],
    icl_methods: List[str],
    debug: bool = False,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Load logits .pkl and align to raw cal/test by ID; assert exact alignment."""
    logits_data_all: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for m in prompt_methods:
        for fs in icl_methods:
            logits_file = os.path.join(hidden_reps_dir, f"{data_name}_{model_name}_results.pkl")
            if not os.path.exists(logits_file):
                raise FileNotFoundError(f"Logits file not found: {logits_file}")

            with open(logits_file, "rb") as f:
                logits_data = pickle.load(f)

            logits_data = [item for item in logits_data if int(item.get("id")) not in ids_to_remove]
            if debug:
                logits_data = logits_data[:200]

            logits_by_id = index_logits_by_id(logits_data)

            cal_logits_data: List[Dict[str, Any]] = []
            test_logits_data: List[Dict[str, Any]] = []

            for raw_item in cal_raw_data:
                rid = str(raw_item["id"])
                if rid not in logits_by_id:
                    raise ValueError(f"Missing logits for calibration ID {rid}")
                cal_logits_data.append(logits_by_id[rid])

            for raw_item in test_raw_data:
                rid = str(raw_item["id"])
                if rid not in logits_by_id:
                    raise ValueError(f"Missing logits for test ID {rid}")
                test_logits_data.append(logits_by_id[rid])

            if debug:
                cal_logits_data = cal_logits_data[:50]
                test_logits_data = test_logits_data[:50]

            # strict alignment
            assert len(cal_logits_data) == len(cal_raw_data)
            assert len(test_logits_data) == len(test_raw_data)
            for rr, ll in zip(cal_raw_data, cal_logits_data):
                assert str(rr["id"]) == str(ll["id"])
            for rr, ll in zip(test_raw_data, test_logits_data):
                assert str(rr["id"]) == str(ll["id"])

            logits_data_all[f"{m}_{fs}"] = {"cal": cal_logits_data, "test": test_logits_data}

    return logits_data_all


# =========================
# Baseline CP (LAC/APS)
# =========================

def LAC_CP(
    logits_data_all: Dict[str, Dict[str, List[Dict[str, Any]]]],
    cal_raw_data: List[Dict[str, Any]],
    prompt_methods: List[str],
    icl_methods: List[str],
    alpha: float = 0.1,
) -> Dict[str, Dict[str, List[str]]]:
    pred_sets_all: Dict[str, Dict[str, List[str]]] = {}

    for m in prompt_methods:
        for fs in icl_methods:
            key = f"{m}_{fs}"
            pred_sets_all[key] = {}
            cal_scores: List[float] = []
            cal_logits_data = logits_data_all[key]["cal"]

            for idx, row in enumerate(cal_logits_data):
                probs = softmax(row['logits'][0, :])
                truth_answer = cal_raw_data[idx]["answer"]
                assert int(cal_raw_data[idx]["id"]) == int(row["id"])
                cal_scores.append(1 - probs[options.index(truth_answer)])

            n = len(cal_logits_data)
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            qhat = np.quantile(np.array(cal_scores), q_level, method="higher")

            pred_sets: Dict[str, List[str]] = {}
            for row in logits_data_all[key]["test"]:
                probs = softmax(row['logits'][0, :])
                ps = [options[ii] for ii, p in enumerate(probs) if p >= 1 - qhat]
                if not ps:
                    ps.append(options[np.argmax(probs)])
                pred_sets[str(row["id"])] = ps

            pred_sets_all[key] = pred_sets

    return pred_sets_all


def APS_CP(
    logits_data_all: Dict[str, Dict[str, List[Dict[str, Any]]]],
    cal_raw_data: List[Dict[str, Any]],
    prompt_methods: List[str],
    icl_methods: List[str],
    alpha: float = 0.1,
) -> Dict[str, Dict[str, List[str]]]:
    pred_sets_all: Dict[str, Dict[str, List[str]]] = {}

    for m in prompt_methods:
        for fs in icl_methods:
            key = f"{m}_{fs}"
            pred_sets_all[key] = {}
            cal_scores: List[float] = []
            cal_logits_data = logits_data_all[key]["cal"]

            for idx, row in enumerate(cal_logits_data):
                probs = softmax(row['logits'][0, :])
                truth_answer = cal_raw_data[idx]["answer"]
                assert int(cal_raw_data[idx]["id"]) == int(row["id"])
                cal_pi = np.argsort(probs)[::-1]
                cal_cumsum = np.take_along_axis(probs, cal_pi, axis=0).cumsum()
                cal_sum_r = np.take_along_axis(cal_cumsum, cal_pi.argsort(), axis=0)
                cal_scores.append(float(cal_sum_r[options.index(truth_answer)]))

            n = len(cal_logits_data)
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            qhat = np.quantile(np.array(cal_scores), q_level, method="higher")

            pred_sets: Dict[str, List[str]] = {}
            for row in logits_data_all[key]["test"]:
                probs = softmax(row['logits'][0, :])
                cal_pi = np.argsort(probs)[::-1]
                cal_cumsum = np.take_along_axis(probs, cal_pi, axis=0).cumsum()

                ps: List[str] = []
                ii = 0
                while ii < len(cal_cumsum) and cal_cumsum[ii] <= qhat:
                    ps.append(options[cal_pi[ii]])
                    ii += 1
                if not ps:
                    ps.append(options[cal_pi[0]])
                pred_sets[str(row["id"])] = ps

            pred_sets_all[key] = pred_sets

    return pred_sets_all


# =========================
# Paraphrase-aware CP/QCP
# =========================

def PA_CP(
    pa_scores: Dict[str, Dict[str, float]],
    cal_raw_data: List[Dict[str, Any]],
    test_raw_data: List[Dict[str, Any]],
    alpha: float = 0.1,
) -> Dict[str, List[str]]:
    """Standard CP using paraphrase-aware nonconformity scores.

    - Calibration score for item i: s_i = PA[id_i][y_true]
    - Test-time prediction set for item j: { o in options: PA[id_j][o] <= qhat }
      with fallback to the option with minimal score if empty.
    Returns mapping test_id -> prediction set.
    """
    # Build calibration scores
    cal_scores: List[float] = []
    for row in cal_raw_data:
        sid = str(row["id"])
        y_true = row["answer"]
        if sid not in pa_scores or y_true not in pa_scores[sid]:
            raise KeyError(f"Missing PA score for cal id={sid}, option={y_true}")
        cal_scores.append(float(pa_scores[sid][y_true]))

    n = len(cal_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(np.array(cal_scores, dtype=float), q_level, method="higher")

    # Build prediction sets for test
    pred_sets: Dict[str, List[str]] = {}
    for row in test_raw_data:
        sid = str(row["id"])
        if sid not in pa_scores:
            raise KeyError(f"Missing PA scores for test id={sid}")
        scores_map = pa_scores[sid]
        ps = [opt for opt in options if opt in scores_map and float(scores_map[opt]) <= qhat]
        if not ps:
            # fallback: include the option with smallest score
            available = [(opt, float(scores_map[opt])) for opt in options if opt in scores_map]
            if not available:
                raise KeyError(f"No PA scores available for any option for id={sid}")
            ps = [min(available, key=lambda t: t[1])[0]]
        pred_sets[sid] = ps

    return pred_sets


def PA_QCP(
    pa_scores: Dict[str, Dict[str, float]],
    logits_data_all: Dict[str, Dict[str, List[Dict[str, Any]]]],
    cal_raw_data: List[Dict[str, Any]],
    prompt_methods: List[str],
    icl_methods: List[str],
    alpha: float = 0.1,
    phi_mode: str = "intercept",
    phi_cal: Optional[np.ndarray] = None,
    phi_test: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """Conditional CP using paraphrase-aware nonconformity scores with the same Phi options.

    Conditioning features follow existing behavior:
    - If phi_mode == cluster_sbert: use provided phi_cal/phi_test
    - Else: use probs as features (consistent with LAC/APS implementations)
    """
    pred_sets_all: Dict[str, Dict[str, List[str]]] = {}

    use_precomputed_phi = (phi_mode == "cluster_sbert")
    phi_fn = (lambda x: x) if use_precomputed_phi else get_phi(phi_mode)

    for m in prompt_methods:
        for fs in icl_methods:
            key = f"{m}_{fs}"
            pred_sets_all[key] = {}

            # Calibration nonconformity scores from PA
            cal_scores: List[float] = []
            cal_logits_data = logits_data_all[key]["cal"]
            cal_features_list: List[np.ndarray] = []

            for idx, row in enumerate(cal_logits_data):
                sid = str(row["id"])
                y_true = cal_raw_data[idx]["answer"]
                assert int(cal_raw_data[idx]["id"]) == int(row["id"])
                if sid not in pa_scores or y_true not in pa_scores[sid]:
                    raise KeyError(f"Missing PA score for cal id={sid}, option={y_true}")
                cal_scores.append(float(pa_scores[sid][y_true]))

                if not use_precomputed_phi:
                    probs = softmax(row['logits'][0, :])
                    cal_features_list.append(probs)

            cal_scores = np.array(cal_scores, dtype=float)
            if use_precomputed_phi:
                cal_features = phi_cal
            else:
                cal_features = np.array(cal_features_list)

            condconf = CondConf(score_fn=lambda x, y: y, Phi_fn=phi_fn)
            condconf.setup_problem(cal_features, cal_scores)

            # Predict sets for test
            test_logits_data = logits_data_all[key]["test"]
            pred_sets: Dict[str, List[str]] = {}

            for idx, row in enumerate(test_logits_data):
                sid = str(row["id"])
                if sid not in pa_scores:
                    raise KeyError(f"Missing PA scores for test id={sid}")
                if use_precomputed_phi:
                    feature_vec = phi_test[idx:idx + 1]
                else:
                    probs = softmax(row['logits'][0, :])
                    feature_vec = probs.reshape(1, -1)

                qhat = condconf.predict(
                    quantile=(1 - alpha),
                    x_test=feature_vec,
                    score_inv_fn=lambda c, x: c,
                    randomize=True,
                )

                scores_map = pa_scores[sid]
                ps = [opt for opt in options if opt in scores_map and float(scores_map[opt]) <= qhat]
                if not ps:
                    available = [(opt, float(scores_map[opt])) for opt in options if opt in scores_map]
                    if not available:
                        raise KeyError(f"No PA scores available for any option for id={sid}")
                    ps = [min(available, key=lambda t: t[1])[0]]
                pred_sets[sid] = ps

            pred_sets_all[key] = pred_sets

    return pred_sets_all
# =========================
# QCP (with Phi)
# =========================

def LAC_QCP(
    logits_data_all: Dict[str, Dict[str, List[Dict[str, Any]]]],
    cal_raw_data: List[Dict[str, Any]],
    prompt_methods: List[str],
    icl_methods: List[str],
    alpha: float = 0.1,
    phi_mode: str = "intercept",
    phi_cal: Optional[np.ndarray] = None,
    phi_test: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """QCP with LAC score."""
    pred_sets_all: Dict[str, Dict[str, List[str]]] = {}

    # Decide Phi function
    use_precomputed_phi = (phi_mode == "cluster_sbert")
    phi_fn = (lambda x: x) if use_precomputed_phi else get_phi(phi_mode)

    for m in prompt_methods:
        for fs in icl_methods:
            key = f"{m}_{fs}"
            pred_sets_all[key] = {}

            # Calibration scores
            cal_scores: List[float] = []
            cal_logits_data = logits_data_all[key]["cal"]
            cal_features_list: List[np.ndarray] = []

            for idx, row in enumerate(cal_logits_data):
                probs = softmax(row['logits'][0, :])
                truth_answer = cal_raw_data[idx]["answer"]
                assert int(cal_raw_data[idx]["id"]) == int(row["id"])
                cal_scores.append(1 - probs[options.index(truth_answer)])

                if not use_precomputed_phi:
                    cal_features_list.append(probs)

            cal_scores = np.array(cal_scores)

            if use_precomputed_phi:
                cal_features = phi_cal  # precomputed Φ
            else:
                cal_features = np.array(cal_features_list)

            condconf = CondConf(score_fn=lambda x, y: y, Phi_fn=phi_fn)
            condconf.setup_problem(cal_features, cal_scores)

            # Prediction
            pred_sets: Dict[str, List[str]] = {}
            test_logits_data = logits_data_all[key]["test"]

            for idx, row in enumerate(test_logits_data):
                probs = softmax(row['logits'][0, :])
                if use_precomputed_phi:
                    feature_vec = phi_test[idx:idx + 1]
                else:
                    feature_vec = probs.reshape(1, -1)

                qhat = condconf.predict(
                    quantile=(1 - alpha),
                    x_test=feature_vec,
                    score_inv_fn=lambda c, x: c,
                    randomize=True,
                )

                ps = [options[ii] for ii, p in enumerate(probs) if p >= 1 - qhat]
                if not ps:
                    ps.append(options[np.argmax(probs)])
                pred_sets[str(row["id"])] = ps

            pred_sets_all[key] = pred_sets

    return pred_sets_all


def APS_QCP(
    logits_data_all: Dict[str, Dict[str, List[Dict[str, Any]]]],
    cal_raw_data: List[Dict[str, Any]],
    prompt_methods: List[str],
    icl_methods: List[str],
    alpha: float = 0.1,
    phi_mode: str = "intercept",
    phi_cal: Optional[np.ndarray] = None,
    phi_test: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """QCP with APS score."""
    ada_pred_sets_all: Dict[str, Dict[str, List[str]]] = {}

    use_precomputed_phi = (phi_mode == "cluster_sbert")
    phi_fn = (lambda x: x) if use_precomputed_phi else get_phi(phi_mode)

    for m in prompt_methods:
        for fs in icl_methods:
            key = f"{m}_{fs}"
            ada_pred_sets_all[key] = {}

            cal_scores: List[float] = []
            cal_logits_data = logits_data_all[key]["cal"]
            cal_features_list: List[np.ndarray] = []

            for idx, row in enumerate(cal_logits_data):
                probs = softmax(row['logits'][0, :])
                truth_answer = cal_raw_data[idx]["answer"]
                assert int(cal_raw_data[idx]["id"]) == int(row["id"])

                cal_pi = np.argsort(probs)[::-1]
                cal_cumsum = np.take_along_axis(probs, cal_pi, axis=0).cumsum()
                cal_sum_r = np.take_along_axis(cal_cumsum, cal_pi.argsort(), axis=0)
                cal_scores.append(float(cal_sum_r[options.index(truth_answer)]))

                if not use_precomputed_phi:
                    cal_features_list.append(probs)

            cal_scores = np.array(cal_scores)

            if use_precomputed_phi:
                cal_features = phi_cal
            else:
                cal_features = np.array(cal_features_list)

            condconf = CondConf(score_fn=lambda x, y: y, Phi_fn=phi_fn)
            condconf.setup_problem(cal_features, cal_scores)

            pred_sets: Dict[str, List[str]] = {}
            test_logits_data = logits_data_all[key]["test"]

            for idx, row in enumerate(test_logits_data):
                probs = softmax(row['logits'][0, :])
                if use_precomputed_phi:
                    feature_vec = phi_test[idx:idx + 1]
                else:
                    feature_vec = probs.reshape(1, -1)

                qhat = condconf.predict(
                    quantile=(1 - alpha),
                    x_test=feature_vec,
                    score_inv_fn=lambda c, x: c,
                    randomize=True,
                )

                cal_pi = np.argsort(probs)[::-1]
                cal_cumsum = np.take_along_axis(probs, cal_pi, axis=0).cumsum()

                ps: List[str] = []
                ii = 0
                while ii < len(cal_cumsum) and cal_cumsum[ii] <= qhat:
                    ps.append(options[cal_pi[ii]])
                    ii += 1
                if not ps:
                    ps.append(options[cal_pi[0]])

                pred_sets[str(row["id"])] = ps

            ada_pred_sets_all[key] = pred_sets

    return ada_pred_sets_all


# =========================
# Metrics & Reporting
# =========================

def get_accuracy(logits_data: List[Dict[str, Any]], raw_data: List[Dict[str, Any]]) -> Tuple[float, List[str]]:
    res = []
    preds: List[str] = []
    for idx, row in enumerate(raw_data):
        truth_answer = row["answer"]
        pred = logits_data[idx]
        assert int(pred["id"]) == int(row["id"])
        pred_answer = options[np.argmax(pred['logits'][0, :])]
        preds.append(pred_answer)
        res.append(int(pred_answer == truth_answer))
    return float(sum(res)) / len(res), preds


def cal_acc(
    logits_data_all: Dict[str, Dict[str, List[Dict[str, Any]]]],
    test_raw_data: List[Dict[str, Any]],
    prompt_methods: List[str],
    icl_methods: List[str],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    results_acc: Dict[str, float] = {}
    E_ratios: Dict[str, float] = {}
    F_ratios: Dict[str, float] = {}
    for m in prompt_methods:
        for fs in icl_methods:
            key = f"{m}_{fs}"
            test_logits_data = logits_data_all[key]["test"]
            acc, preds = get_accuracy(test_logits_data, test_raw_data)
            results_acc[key] = acc
            counts = Counter(preds)
            E_ratios[key] = counts.get("E", 0) / len(preds)
            F_ratios[key] = counts.get("F", 0) / len(preds)
    return results_acc, E_ratios, F_ratios


def convert_id_to_ans(test_raw_data: List[Dict[str, Any]]) -> Dict[str, str]:
    return {str(row["id"]): row["answer"] for row in test_raw_data}


def cal_coverage(
    pred_sets_all: Dict[str, Dict[str, List[str]]],
    test_id_to_answer: Dict[str, str],
    prompt_methods: List[str],
    icl_methods: List[str],
) -> Dict[str, float]:
    """Calculate empirical coverage of prediction sets."""
    coverage_all: Dict[str, float] = {}
    for m in prompt_methods:
        for fs in icl_methods:
            key = f"{m}_{fs}"
            cover = []
            pred_sets = pred_sets_all[key]
            for k, v in pred_sets.items():
                cover.append(int(test_id_to_answer[k] in v))
            coverage_all[key] = float(sum(cover)) / len(cover) if cover else 0.0
    return coverage_all


def cal_set_size(
    pred_sets_all: Dict[str, Dict[str, List[str]]],
    prompt_methods: List[str],
    icl_methods: List[str],
) -> Dict[str, float]:
    set_sizes: Dict[str, float] = {}
    for m in prompt_methods:
        for fs in icl_methods:
            key = f"{m}_{fs}"
            sizes = [len(v) for _, v in pred_sets_all[key].items()]
            set_sizes[key] = float(sum(sizes)) / len(sizes) if sizes else 0.0
    return set_sizes


def cal_uacc(results_acc: Dict[str, float], set_sizes: Dict[str, float]) -> Dict[str, float]:
    """Utility-adjusted accuracy (simple)."""
    K = len(options)
    out: Dict[str, float] = {}
    for k, v in results_acc.items():
        denom = set_sizes.get(k, 1.0)
        out[k] = float(v) * np.sqrt(K) / max(denom, 1e-12)
    return out


# =========================
# Orchestrator
# =========================

def apply_conformal_prediction(args: argparse.Namespace) -> Dict[str, Any]:
    all_data_results: Dict[str, Any] = {}

    for data_name in args.data_names:
        # Determine calibration/test IDs either from manifest or by split
        calib_ids_override: Optional[Set[str]] = None
        test_ids_override: Optional[Set[str]] = None
        if args.split_manifest_template:
            manifest_path = args.split_manifest_template.format(data_name=data_name)
            print(f"Loading split IDs from manifest: {manifest_path}")
            calib_ids_override, test_ids_override = load_splits_from_manifest(manifest_path)

        # raw & logits using explicit IDs when available
        cal_raw_data, test_raw_data, cal_ids, test_ids = get_raw_data(
            args.raw_data_dir,
            data_name,
            args.cal_ratio,
            debug=args.debug,
            seed=args.seed,
            calib_ids_override=calib_ids_override,
            test_ids_override=test_ids_override,
        )
        logits_data_all = get_logits_data(
            args.model,
            data_name,
            cal_raw_data,
            test_raw_data,
            cal_ids,
            test_ids,
            args.hidden_reps_dir,
            args.prompt_methods,
            args.icl_methods,
            debug=args.debug,
        )

        results_acc, E_ratios, F_ratios = cal_acc(
            logits_data_all, test_raw_data, args.prompt_methods, args.icl_methods
        )
        test_id_to_answer = convert_id_to_ans(test_raw_data)

        # cluster_sbert Φ (optional)
        phi_cal = None
        phi_test = None
        cluster_metadata: Optional[Dict[str, Any]] = None
        all_raw_for_cluster: Optional[List[Dict[str, Any]]] = None

        if args.phi_mode == "cluster_sbert":
            raw_data_path = os.path.join(args.raw_data_dir, data_name + ".json")
            with open(raw_data_path, "r") as f:
                all_raw_for_cluster = json.load(f)
            all_raw_for_cluster = [it for it in all_raw_for_cluster if it.get("id") not in ids_to_remove]
            if args.debug:
                all_raw_for_cluster = all_raw_for_cluster[:200]

            # Semi-reworded fix: when --cluster_paraphrase_dir is provided,
            # train the SBERT cluster (Phi) on LM-reworded paraphrases of each
            # question instead of the original text. Required for the
            # semi-reworded QCCP setting, where calibration is drawn from the
            # original distribution but the test input is a reworded variant —
            # evaluating Phi on the original text on calib and reworded text on
            # test breaks the (Phi, score) exchangeability QCCP relies on.
            # Pass --cluster_paraphrase_index to choose which paraphrase to use
            # (default 0, i.e. the first reworded variant).
            cluster_dataset_name = data_name
            if getattr(args, "cluster_paraphrase_dir", "") and args.cluster_paraphrase_dir:
                para_path = os.path.join(args.cluster_paraphrase_dir, data_name + ".json")
                if not os.path.exists(para_path):
                    raise FileNotFoundError(
                        f"--cluster_paraphrase_dir given but paraphrase JSON not found: {para_path}"
                    )
                with open(para_path, "r") as f:
                    cluster_paraphrases = json.load(f)
                idx = int(args.cluster_paraphrase_index)
                replaced = 0
                missing = 0
                for it in all_raw_for_cluster:
                    qid = str(it["id"])
                    cands = cluster_paraphrases.get(qid)
                    if cands is None:
                        try:
                            cands = cluster_paraphrases.get(int(qid))
                        except (TypeError, ValueError):
                            cands = None
                    if cands and len(cands) > idx and str(cands[idx]).strip():
                        it[args.question_key] = str(cands[idx])
                        replaced += 1
                    else:
                        missing += 1
                # Distinct cache key so SBERT embeddings are recomputed for the
                # reworded text rather than served from the original-text cache.
                cluster_dataset_name = f"{data_name}_para{idx}"
                print(
                    f"[cluster] using reworded paraphrase[{idx}] from {para_path} "
                    f"for cluster phi: replaced={replaced}/{len(all_raw_for_cluster)}, "
                    f"missing={missing}"
                )

            phi_cal, phi_test, cluster_metadata = phi_cluster_sbert(
                dataset_name=cluster_dataset_name,
                raw_data=all_raw_for_cluster,
                cal_ids=cal_ids,
                test_ids=test_ids,
                question_key=args.question_key,
                context_key=args.context_key,
                sbert_model=args.sbert_model,
                embed_cache_dir=args.embed_cache_dir,
                clusters=args.clusters,
                cluster_selection=args.cluster_selection,
                phi_rep=args.phi_rep,
                normalize_embeddings=args.normalize_embeddings,
                kmeans_minibatch=args.kmeans_minibatch,
                seed=args.seed,
                verbose=args.verbose,
            )

            # Save metadata & arrays (optional)
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(args.out_dir, f"cluster_meta_{data_name}_{args.model}.json"), "w") as f:
                json.dump(cluster_metadata, f, indent=2)

            # Save centroids/labels (already in metadata; still write .npy for convenience)
            centroids_arr = np.array(cluster_metadata["centroids"], dtype=float)
            np.save(os.path.join(args.out_dir, f"centroids_{data_name}_{args.model}.npy"), centroids_arr)
            np.save(os.path.join(args.out_dir, f"labels_cal_{data_name}_{args.model}.npy"), np.array(cluster_metadata["cal_labels"], dtype=int))
            np.save(os.path.join(args.out_dir, f"labels_test_{data_name}_{args.model}.npy"), np.array(cluster_metadata["test_labels"], dtype=int))

        # If paraphrase-aware paths requested, load the file for this dataset
        pa_scores: Optional[Dict[str, Dict[str, float]]] = None
        if args.use_pa:
            if not args.pa_metric or not args.pa_score_path_template:
                raise ValueError("--use_pa requires --pa_metric and --pa_score_path_template")
            pa_path = args.pa_score_path_template.format(data_name=data_name, model=args.model)
            print(f"Loading paraphrase scores from {pa_path}")
            pa_scores = load_paraphrase_scores(pa_path, args.pa_metric)

        # QCP LAC
        pred_sets_all_LAC = LAC_QCP(
            logits_data_all,
            cal_raw_data,
            args.prompt_methods,
            args.icl_methods,
            alpha=args.alpha,
            phi_mode=args.phi_mode,
            phi_cal=phi_cal,
            phi_test=phi_test,
        )
        coverage_all_LAC = cal_coverage(
            pred_sets_all_LAC, test_id_to_answer, args.prompt_methods, args.icl_methods
        )
        set_sizes_LAC = cal_set_size(pred_sets_all_LAC, args.prompt_methods, args.icl_methods)
        results_uacc_LAC = cal_uacc(results_acc, set_sizes_LAC)

        # QCP APS
        pred_sets_all_APS = APS_QCP(
            logits_data_all,
            cal_raw_data,
            args.prompt_methods,
            args.icl_methods,
            alpha=args.alpha,
            phi_mode=args.phi_mode,
            phi_cal=phi_cal,
            phi_test=phi_test,
        )
        coverage_all_APS = cal_coverage(
            pred_sets_all_APS, test_id_to_answer, args.prompt_methods, args.icl_methods
        )
        set_sizes_APS = cal_set_size(pred_sets_all_APS, args.prompt_methods, args.icl_methods)
        results_uacc_APS = cal_uacc(results_acc, set_sizes_APS)

        # Optional: Paraphrase-aware CP/QCP
        pa_pred_sets_cp: Optional[Dict[str, List[str]]] = None
        pa_pred_sets_qcp: Optional[Dict[str, Dict[str, List[str]]]] = None
        if pa_scores is not None:
            pa_pred_sets_cp = PA_CP(pa_scores, cal_raw_data, test_raw_data, alpha=args.alpha)
            pa_pred_sets_qcp = PA_QCP(
                pa_scores,
                logits_data_all,
                cal_raw_data,
                args.prompt_methods,
                args.icl_methods,
                alpha=args.alpha,
                phi_mode=args.phi_mode,
                phi_cal=phi_cal,
                phi_test=phi_test,
            )

        if args.save_pred_sets:
            pred_dir = Path(f"{args.out_dir}/pred_sets/{data_name}_{args.model}_{args.phi_mode}")
            pred_dir.mkdir(parents=True, exist_ok=True)
            for key, dct in pred_sets_all_LAC.items():
                with open(pred_dir / f"{args.phi_mode}_{key}_LAC.json", "w") as f:
                    json.dump(dct, f, indent=2)
            for key, dct in pred_sets_all_APS.items():
                with open(pred_dir / f"{args.phi_mode}_{key}_APS.json", "w") as f:
                    json.dump(dct, f, indent=2)
            if pa_pred_sets_cp is not None:
                with open(pred_dir / f"{args.phi_mode}_PA_CP.json", "w") as f:
                    json.dump(pa_pred_sets_cp, f, indent=2)
            if pa_pred_sets_qcp is not None:
                for key, dct in pa_pred_sets_qcp.items():
                    with open(pred_dir / f"{args.phi_mode}_{key}_PA_QCP.json", "w") as f:
                        json.dump(dct, f, indent=2)

        # Collect results
        all_data_results[data_name] = {
            "Acc": results_acc,
            "E_rate": E_ratios,
            "F_rate": F_ratios,
            "LAC_set_size": set_sizes_LAC,
            "APS_set_size": set_sizes_APS,
            "LAC_coverage": coverage_all_LAC,
            "APS_coverage": coverage_all_APS,
            "UAcc_LAC": results_uacc_LAC,
            "UAcc_APS": results_uacc_APS,
            "n_cal": len(cal_raw_data),
            "n_test": len(test_raw_data),
        }
        if cluster_metadata is not None:
            all_data_results[data_name]["cluster_metadata"] = cluster_metadata

        # Add PA metrics if computed
        if pa_scores is not None and pa_pred_sets_cp is not None and pa_pred_sets_qcp is not None:
            # coverage and set size for PA_CP
            pa_cov_cp = np.mean([int(test_id_to_answer[k] in v) for k, v in pa_pred_sets_cp.items()])
            pa_size_cp = float(np.mean([len(v) for v in pa_pred_sets_cp.values()])) if pa_pred_sets_cp else 0.0
            # coverage and set size for PA_QCP averaged over prompt/icls
            pa_cov_q_list = []
            pa_size_q_list = []
            for m in args.prompt_methods:
                for fs in args.icl_methods:
                    key = f"{m}_{fs}"
                    dct = pa_pred_sets_qcp.get(key, {})
                    if dct:
                        pa_cov_q_list.append(np.mean([int(test_id_to_answer[k] in v) for k, v in dct.items()]))
                        pa_size_q_list.append(float(np.mean([len(v) for v in dct.values()])))
            all_data_results[data_name]["PA_CP_coverage"] = float(pa_cov_cp)
            all_data_results[data_name]["PA_CP_set_size"] = float(pa_size_cp)
            all_data_results[data_name]["PA_QCP_coverage"] = float(np.mean(pa_cov_q_list)) if pa_cov_q_list else 0.0
            all_data_results[data_name]["PA_QCP_set_size"] = float(np.mean(pa_size_q_list)) if pa_size_q_list else 0.0

    return all_data_results


def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    all_data_results = apply_conformal_prediction(args)

    # Console summary
    acc_means = []
    LAC_set_mean, APS_set_mean = [], []
    LAC_cov_mean, APS_cov_mean = [], []
    UAcc_LAC_mean, UAcc_APS_mean = [], []

    for data_name in args.data_names:
        acc_m = 100 * np.mean(list(all_data_results[data_name]["Acc"].values()))
        acc_means.append(acc_m)
        print(f"{data_name}_Acc: {acc_m:.2f}")

    print(f"Average acc: {np.mean(acc_means):.2f}")

    for data_name in args.data_names:
        LAC_set_mean.append(np.mean(list(all_data_results[data_name]["LAC_set_size"].values())))
        APS_set_mean.append(np.mean(list(all_data_results[data_name]["APS_set_size"].values())))
        LAC_cov_mean.append(100 * np.mean(list(all_data_results[data_name]["LAC_coverage"].values())))
        APS_cov_mean.append(100 * np.mean(list(all_data_results[data_name]["APS_coverage"].values())))
        UAcc_LAC_mean.append(100 * np.mean(list(all_data_results[data_name]["UAcc_LAC"].values())))
        UAcc_APS_mean.append(100 * np.mean(list(all_data_results[data_name]["UAcc_APS"].values())))
        # Optional PA metrics
        if "PA_QCP_coverage" in all_data_results[data_name]:
            print(f"{data_name}_PA_QCP_Coverage: {100*all_data_results[data_name]['PA_QCP_coverage']:.2f}")
        if "PA_QCP_set_size" in all_data_results[data_name]:
            print(f"{data_name}_PA_QCP_SS: {all_data_results[data_name]['PA_QCP_set_size']:.2f}")
        if "PA_CP_coverage" in all_data_results[data_name]:
            print(f"{data_name}_PA_CP_Coverage: {100*all_data_results[data_name]['PA_CP_coverage']:.2f}")
        if "PA_CP_set_size" in all_data_results[data_name]:
            print(f"{data_name}_PA_CP_SS: {all_data_results[data_name]['PA_CP_set_size']:.2f}")

    pred_set_size = [np.mean([s1, s2]) for s1, s2 in zip(LAC_set_mean, APS_set_mean)]
    for data_name, sz in zip(args.data_names, pred_set_size):
        print(f"{data_name}_SS: {sz:.2f}")
    print(f"Average SS: {np.mean(pred_set_size):.2f}")

    pred_coverage = [np.mean([c1, c2]) for c1, c2 in zip(LAC_cov_mean, APS_cov_mean)]
    for data_name, cr in zip(args.data_names, pred_coverage):
        print(f"{data_name}_Coverage Rate: {cr:.2f}")
    print(f"Average Coverage Rate: {np.mean(pred_coverage):.2f}")

    pred_uacc = [np.mean([u1, u2]) for u1, u2 in zip(UAcc_LAC_mean, UAcc_APS_mean)]
    for data_name, ua in zip(args.data_names, pred_uacc):
        print(f"{data_name}_UAcc: {ua:.2f}")
    print(f"Average UAcc: {np.mean(pred_uacc):.2f}")

    # Machine-readable summary
    summary: Dict[str, Any] = {
        "model": args.model,
        "alpha": args.alpha,
        "phi_mode": args.phi_mode,
        "seed": args.seed,
        "cal_ratio": args.cal_ratio,
        "datasets": {},
    }
    if args.phi_mode == "cluster_sbert":
        summary["phi_rep"] = args.phi_rep
        summary["sbert_model"] = args.sbert_model
        summary["normalize_embeddings"] = bool(args.normalize_embeddings)
        summary["cluster_selection"] = args.cluster_selection
        summary["clusters"] = int(args.clusters)

    for data_name in args.data_names:
        ds = all_data_results[data_name]
        entry: Dict[str, Any] = {
            "acc": float(np.mean(list(ds["Acc"].values()))),
            "coverage_LAC": float(np.mean(list(ds["LAC_coverage"].values()))),
            "coverage_APS": float(np.mean(list(ds["APS_coverage"].values()))),
            "set_size_LAC": float(np.mean(list(ds["LAC_set_size"].values()))),
            "set_size_APS": float(np.mean(list(ds["APS_set_size"].values()))),
            "uacc_LAC": float(np.mean(list(ds["UAcc_LAC"].values()))),
            "uacc_APS": float(np.mean(list(ds["UAcc_APS"].values()))),
            "E_rate": float(np.mean(list(ds["E_rate"].values()))),
            "F_rate": float(np.mean(list(ds["F_rate"].values()))),
            "n_cal": int(ds["n_cal"]),
            "n_test": int(ds["n_test"]),
        }
        if "cluster_metadata" in ds:
            cm = ds["cluster_metadata"]
            entry["K"] = int(cm.get("K", 0))
            entry["cal_cluster_sizes"] = cm.get("cal_cluster_sizes", [])
            entry["test_cluster_sizes"] = cm.get("test_cluster_sizes", [])
        # Include PA metrics when available
        if "PA_CP_coverage" in ds:
            entry["coverage_PA_CP"] = float(ds["PA_CP_coverage"])
        if "PA_CP_set_size" in ds:
            entry["set_size_PA_CP"] = float(ds["PA_CP_set_size"])
        if "PA_QCP_coverage" in ds:
            entry["coverage_PA_QCP"] = float(ds["PA_QCP_coverage"])
        if "PA_QCP_set_size" in ds:
            entry["set_size_PA_QCP"] = float(ds["PA_QCP_set_size"])

        summary["datasets"][data_name] = entry

    summary_file = os.path.join(args.out_dir, f"summary_{args.model}_{args.phi_mode}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

    # Optional CI-like check
    if getattr(args, "fail_on_cov_tol", None) is not None:
        tol = float(args.fail_on_cov_tol)
        target = 1.0 - float(args.alpha)
        for data_name in args.data_names:
            avg_cov = float(np.mean(list(all_data_results[data_name]["LAC_coverage"].values())))
            if abs(avg_cov - target) > tol:
                print(
                    f"ERROR: Coverage {avg_cov:.4f} deviates from target {target:.4f} by more than {tol}"
                )
                sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--raw_data_dir", type=str, default="data", help="Directory where raw data are stored.")
    parser.add_argument("--hidden_reps_dir", type=str, default="outputs_hidden_reps", help="Directory where hidden reps data are stored.")
    parser.add_argument(
        "--data_names",
        nargs="*",
        default=["mmlu_10k", "cosmosqa_10k", "hellaswag_10k", "halu_dialogue", "halu_summarization"],
        #default=["mmlu_10k", "cosmosqa_10k", "hellaswag_10k", "halu_dialogue"],
        #default=["mmlu_10k"],
        help="List of datasets to be evaluated.",
    )
    parser.add_argument(
        "--prompt_methods",
        nargs="*",
        default=["base", "shared", "task"],
        help="List of prompting methods.",
    )
    parser.add_argument(
        "--icl_methods",
        nargs="*",
        default=["icl1"],
        help="Select from icl1, icl0, icl0_cot.",
    )
    parser.add_argument("--cal_ratio", type=float, default=0.5, help="Ratio of calibration split.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Error rate parameter.")
    parser.add_argument(
        "--alpha_grid",
        nargs="*",
        type=float,
        default=None,
        help="List of alpha values to sweep over (not used yet; reserved).",
    )
    parser.add_argument(
        "--phi_mode",
        type=str,
        default="intercept",
        choices=["intercept", "identity", "stats", "cluster_sbert"],
        help="Phi function mode.",
    )
    parser.add_argument("--question_key", type=str, default="question", help="Field name with question text.")
    parser.add_argument("--context_key", type=str, default="context", help="Field name with context text.")
    parser.add_argument(
        "--sbert_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SBERT model name.",
    )
    parser.add_argument("--embed_cache_dir", type=str, default="cache/sbert", help="Cache dir for SBERT embeddings.")
    parser.add_argument("--clusters", type=int, default=20, help="K for KMeans when cluster_selection=fixed.")
    parser.add_argument(
        "--cluster_selection",
        type=str,
        default="fixed",
        choices=["fixed", "auto"],
        help="fixed K or auto by silhouette on calibration embeddings.",
    )
    parser.add_argument(
        "--phi_rep",
        type=str,
        default="onehot",
        choices=["onehot", "dist3"],
        help="phi representation: one-hot cluster or top-3 distances.",
    )
    parser.add_argument("--normalize_embeddings", action="store_true", help="L2-normalize SBERT embeddings.")
    parser.add_argument("--kmeans_minibatch", action="store_true", help="Use MiniBatchKMeans for large data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out_dir", type=str, default="outputs_qcp", help="Output directory.")
    parser.add_argument("--save_pred_sets", action="store_true", help="Save prediction set JSONs.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument("--debug", action="store_true", help="Debug mode with tiny splits.")
    parser.add_argument(
        "--fail_on_cov_tol",
        type=float,
        default=None,
        help="If set, exit(1) when |cov - (1-alpha)| > tol for LAC.",
    )
    parser.add_argument("--use_pa", action="store_true", help="Enable PA_CP and PA_QCP alongside LAC/APS.")
    parser.add_argument(
        "--pa_metric",
        type=str,
        default="S_mean",
        choices=["S_mean", "S_weighted", "S_worst"],
        help="Metric key in paraphrase-aware score JSON.",
    )
    parser.add_argument(
        "--pa_score_path_template",
        type=str,
        default="",
        help="Path template for PA score JSON per dataset; use {data_name}.",
    )
    parser.add_argument(
        "--split_manifest_template",
        type=str,
        default="",
        help="Optional path template to split manifest JSON per dataset (use {data_name}); when provided, calib/test IDs are loaded from it and override cal_ratio.",
    )
    parser.add_argument(
        "--cluster_paraphrase_dir",
        type=str,
        default="",
        help=("If set together with --phi_mode cluster_sbert, override each "
              "row's question text with paraphrases[id][cluster_paraphrase_index] "
              "before computing SBERT embeddings. Used for the semi-reworded "
              "QCCP setting so that calibration and test inputs are both "
              "evaluated on reworded text. Path should be a directory of "
              "<dataset>.json files mapping id -> list[str]."),
    )
    parser.add_argument(
        "--cluster_paraphrase_index",
        type=int,
        default=0,
        help="Which paraphrase to use as the cluster question text when --cluster_paraphrase_dir is set.",
    )

    args = parser.parse_args()
    main(args)