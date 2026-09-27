#!/usr/bin/env python3
"""
Compare calibration performance between proxy models and raw LLM logits.

This script evaluates both approaches on the calibration set and computes
various calibration metrics including ECE, Brier score, and reliability diagrams.
"""

import os
import json
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

# Import from existing codebase
from ece_losses import compute_squared_error_label_binning_pytorch

# Import functions from the training script
import importlib.util
spec = importlib.util.spec_from_file_location("train_proxy", "04_train_proxy_model.py")
train_proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_proxy)

# Extract needed functions and classes
ProxyModel = train_proxy.ProxyModel
to_str_id = train_proxy.to_str_id
load_ground_truth_labels = train_proxy.load_ground_truth_labels
load_hidden_representations = train_proxy.load_hidden_representations

# Set up device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def compute_calibration_metrics(logits: torch.Tensor, labels: torch.Tensor, 
                               num_bins: int = 15) -> Dict[str, float]:
    """
    Compute comprehensive calibration metrics.
    
    Args:
        logits: Model logits of shape (N, num_classes)
        labels: True labels of shape (N,)
        num_bins: Number of bins for ECE calculation
        
    Returns:
        Dictionary containing various calibration metrics
    """
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        confidences = torch.max(probs, dim=-1)[0]
        predictions = torch.argmax(probs, dim=-1)
        correct = (predictions == labels).float()
        
        # Convert to numpy for sklearn compatibility
        conf_np = confidences.cpu().numpy()
        correct_np = correct.cpu().numpy()
        probs_np = probs.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # ECE (Expected Calibration Error) - using existing implementation
        ece = float(compute_squared_error_label_binning_pytorch(logits, labels, m=num_bins, temperature=0.1).item())
        
        # MCE (Maximum Calibration Error)
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                mce = max(mce, torch.abs(avg_confidence_in_bin - accuracy_in_bin).item())
        
        # Brier Score (multi-class)
        brier_score = 0.0
        for i in range(probs.shape[1]):
            true_labels = (labels_np == i).astype(float)
            brier_score += brier_score_loss(true_labels, probs_np[:, i])
        brier_score /= probs.shape[1]
        
        # Accuracy
        accuracy = correct.float().mean().item()
        
        # NLL (Negative Log Likelihood)
        nll = -torch.log(probs[torch.arange(len(labels)), labels] + 1e-8).mean().item()
        
        # Reliability diagram data
        reliability_data = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            if in_bin.sum() > 0:
                accuracy_in_bin = correct[in_bin].float().mean().item()
                avg_confidence_in_bin = confidences[in_bin].mean().item()
                count_in_bin = in_bin.sum().item()
                reliability_data.append({
                    'bin_lower': bin_lower.item(),
                    'bin_upper': bin_upper.item(),
                    'accuracy': accuracy_in_bin,
                    'confidence': avg_confidence_in_bin,
                    'count': count_in_bin
                })
        
        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier_score,
            'accuracy': accuracy,
            'nll': nll,
            'reliability_data': reliability_data
        }

def load_proxy_model(model_path: str, input_dim: int, num_classes: int = 6) -> nn.Module:
    """Load a trained proxy model."""
    model = ProxyModel(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def evaluate_proxy_model(model: nn.Module, hidden_reps: np.ndarray, 
                        labels: np.ndarray) -> Dict[str, float]:
    """Evaluate proxy model on given data."""
    with torch.no_grad():
        X = torch.tensor(hidden_reps, dtype=torch.float32, device=DEVICE)
        y = torch.tensor(labels, dtype=torch.long, device=DEVICE)
        logits = model(X)
        return compute_calibration_metrics(logits, y)

def evaluate_llm_logits(llm_logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Evaluate raw LLM logits on given data."""
    with torch.no_grad():
        logits = torch.tensor(llm_logits, dtype=torch.float32, device=DEVICE)
        y = torch.tensor(labels, dtype=torch.long, device=DEVICE)
        return compute_calibration_metrics(logits, y)

def create_reliability_diagram(proxy_data: List[Dict], llm_data: List[Dict], 
                             save_path: str, title: str = "Reliability Diagram"):
    """Create a reliability diagram comparing proxy model vs LLM logits."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Proxy model reliability diagram
    proxy_conf = [d['confidence'] for d in proxy_data]
    proxy_acc = [d['accuracy'] for d in proxy_data]
    proxy_counts = [d['count'] for d in proxy_data]
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    ax1.plot(proxy_conf, proxy_acc, 'o-', linewidth=2, markersize=6, label='Proxy Model')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Proxy Model Reliability')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add sample counts as text
    for i, (conf, acc, count) in enumerate(zip(proxy_conf, proxy_acc, proxy_counts)):
        if count > 0:
            ax1.annotate(f'n={count}', (conf, acc), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8, alpha=0.7)
    
    # LLM logits reliability diagram
    llm_conf = [d['confidence'] for d in llm_data]
    llm_acc = [d['accuracy'] for d in llm_data]
    llm_counts = [d['count'] for d in llm_data]
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    ax2.plot(llm_conf, llm_acc, 'o-', linewidth=2, markersize=6, color='red', label='LLM Logits')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('LLM Logits Reliability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add sample counts as text
    for i, (conf, acc, count) in enumerate(zip(llm_conf, llm_acc, llm_counts)):
        if count > 0:
            ax2.annotate(f'n={count}', (conf, acc), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8, alpha=0.7)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_metrics_comparison(proxy_metrics: Dict[str, float], llm_metrics: Dict[str, float], 
                            save_path: str, dataset_name: str):
    """Create a bar chart comparing metrics between proxy model and LLM logits."""
    metrics_to_plot = ['ece', 'mce', 'brier_score', 'nll']
    metric_labels = ['ECE', 'MCE', 'Brier Score', 'NLL']
    
    proxy_values = [proxy_metrics[m] for m in metrics_to_plot]
    llm_values = [llm_metrics[m] for m in metrics_to_plot]
    
    x = np.arange(len(metric_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width/2, proxy_values, width, label='Proxy Model', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x + width/2, llm_values, width, label='LLM Logits', alpha=0.8, color='lightcoral')
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Calibration Metrics Comparison - {dataset_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare proxy model vs LLM logits calibration")
    parser.add_argument("--dataset", type=str, required=True, 
                       help="Dataset name (e.g., cosmosqa_10k)")
    parser.add_argument("--train_ids", type=str, required=True,
                       help="Path to training IDs JSON file")
    parser.add_argument("--calib_ids", type=str, required=True,
                       help="Path to calibration IDs JSON file")
    parser.add_argument("--hidden_reps", type=str, required=True,
                       help="Path to hidden representations pickle file")
    parser.add_argument("--ground_truth_json", type=str, required=True,
                       help="Path to ground truth JSON file")
    parser.add_argument("--proxy_model_path", type=str, required=True,
                       help="Path to trained proxy model")
    parser.add_argument("--output_dir", type=str, default="calibration_comparison",
                       help="Output directory for results and plots")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logging.info(f"Starting calibration comparison for dataset: {args.dataset}")
    
    # Load ground truth labels
    ground_truth_labels = load_ground_truth_labels(args.ground_truth_json, debug=args.debug)
    
    # Load hidden representations and logits
    hidden_reps, llm_logits, predicted_answers, question_ids, n_samples_per_q = load_hidden_representations(
        args.hidden_reps, debug=args.debug)
    
    # Load calibration IDs
    with open(args.calib_ids, "r") as f:
        calib_ids = {to_str_id(x) for x in json.load(f)}
    
    # Filter data to calibration set
    filtered_hidden_reps_list = []
    filtered_llm_logits_list = []
    filtered_question_ids = []
    
    for i, q_id in enumerate(question_ids):
        if q_id in calib_ids:
            start = i * n_samples_per_q
            end = (i + 1) * n_samples_per_q
            filtered_hidden_reps_list.append(hidden_reps[start:end])
            filtered_llm_logits_list.append(llm_logits[start:end])
            filtered_question_ids.append(q_id)
    
    if not filtered_hidden_reps_list:
        logging.error("No matching question IDs found in the calibration IDs file.")
        return 1
    
    # Concatenate filtered data
    calib_hidden_reps = np.concatenate(filtered_hidden_reps_list, axis=0)
    calib_llm_logits = np.concatenate(filtered_llm_logits_list, axis=0)
    
    # Create labels for calibration set
    calib_labels = []
    missing_label_ids = []
    for q_id in filtered_question_ids:
        if q_id not in ground_truth_labels:
            missing_label_ids.append(q_id)
            continue
        calib_labels.extend([ground_truth_labels[q_id]] * n_samples_per_q)
    
    if missing_label_ids:
        logging.warning(f"Missing ground-truth labels for {len(missing_label_ids)} IDs")
    
    calib_labels = np.array(calib_labels)
    
    logging.info(f"Calibration set: {len(filtered_question_ids)} questions, {len(calib_labels)} samples")
    logging.info(f"Hidden reps shape: {calib_hidden_reps.shape}")
    logging.info(f"LLM logits shape: {calib_llm_logits.shape}")
    
    # Load proxy model
    input_dim = int(calib_hidden_reps.shape[1])
    proxy_model = load_proxy_model(args.proxy_model_path, input_dim)
    
    # Evaluate proxy model
    logging.info("Evaluating proxy model...")
    proxy_metrics = evaluate_proxy_model(proxy_model, calib_hidden_reps, calib_labels)
    
    # Evaluate LLM logits
    logging.info("Evaluating LLM logits...")
    llm_metrics = evaluate_llm_logits(calib_llm_logits, calib_labels)
    
    # Print results
    logging.info("=" * 60)
    logging.info("CALIBRATION COMPARISON RESULTS")
    logging.info("=" * 60)
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Calibration set size: {len(calib_labels)} samples")
    logging.info("")
    
    logging.info("PROXY MODEL METRICS:")
    for metric, value in proxy_metrics.items():
        if metric != 'reliability_data':
            logging.info(f"  {metric.upper()}: {value:.6f}")
    
    logging.info("")
    logging.info("LLM LOGITS METRICS:")
    for metric, value in llm_metrics.items():
        if metric != 'reliability_data':
            logging.info(f"  {metric.upper()}: {value:.6f}")
    
    logging.info("")
    logging.info("IMPROVEMENT (Proxy vs LLM):")
    for metric in ['ece', 'mce', 'brier_score', 'nll']:
        if metric in proxy_metrics and metric in llm_metrics:
            improvement = ((llm_metrics[metric] - proxy_metrics[metric]) / llm_metrics[metric]) * 100
            logging.info(f"  {metric.upper()}: {improvement:+.2f}% improvement")
    
    # Save detailed results
    results = {
        'dataset': args.dataset,
        'calibration_set_size': len(calib_labels),
        'proxy_metrics': {k: v for k, v in proxy_metrics.items() if k != 'reliability_data'},
        'llm_metrics': {k: v for k, v in llm_metrics.items() if k != 'reliability_data'},
        'improvements': {}
    }
    
    for metric in ['ece', 'mce', 'brier_score', 'nll']:
        if metric in proxy_metrics and metric in llm_metrics:
            improvement = ((llm_metrics[metric] - proxy_metrics[metric]) / llm_metrics[metric]) * 100
            results['improvements'][metric] = improvement
    
    with open(output_dir / f"{args.dataset}_calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    logging.info("Creating visualizations...")
    
    # Reliability diagrams
    create_reliability_diagram(
        proxy_metrics['reliability_data'], 
        llm_metrics['reliability_data'],
        output_dir / f"{args.dataset}_reliability_diagram.png",
        f"Reliability Diagram - {args.dataset}"
    )
    
    # Metrics comparison
    create_metrics_comparison(
        proxy_metrics, 
        llm_metrics,
        output_dir / f"{args.dataset}_metrics_comparison.png",
        args.dataset
    )
    
    logging.info(f"Results saved to {output_dir}")
    logging.info("Calibration comparison completed successfully!")
    
    return 0

if __name__ == "__main__":
    exit(main())
