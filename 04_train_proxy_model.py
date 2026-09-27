import os
import json
import pickle
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from ece_losses import compute_squared_error_label_binning_pytorch  # Import Soft-Binned ECE Loss

# Device and determinism
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_str_id(x: Any) -> str:
    # robust stringify (handles numpy ints and digit-strings)
    if isinstance(x, (np.integer,)):
        return str(int(x))
    try:
        # normalize pure digit strings like "001" to int then str
        if isinstance(x, str) and x.strip().isdigit():
            return str(int(x))
    except Exception:
        pass
    return str(x)


def load_ground_truth_labels(json_file: str, debug: bool = False) -> Dict[str, int]:
    """Load ground truth labels from dataset JSON file. Normalize IDs to str."""
    with open(json_file, "r") as f:
        data = json.load(f)
    if debug:
        data = data[:100]
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
    ground_truth = {to_str_id(item["id"]): label_map[item["answer"]] for item in data}
    #ground_truth = {to_str_id(item["id"]): random.randint(0, 3) for item in data}
    return ground_truth


def load_hidden_representations(pickle_file: str, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], int]:
    """
    Load hidden representations, logits, and predicted answers from pickle.
    Returns arrays and normalized question_ids (str), and n_samples_per_q.
    """
    with open(pickle_file, "rb") as f:
        results = pickle.load(f)
    if debug:
        results = results[:100]

    hidden_reps_list: List[np.ndarray] = []
    logits_list: List[np.ndarray] = []
    predictions_list: List[np.ndarray] = []
    question_ids: List[str] = []

    n_samples_per_q: int = -1
    for item in results:
        qid = to_str_id(item["id"])
        question_ids.append(qid)
        h = np.array(item["hidden_representations"])  # (n, hidden_dim)
        l = np.array(item["logits"])  # (n, num_classes)
        p = np.array(item.get("predicted_answers", []))  # (n,)
        if n_samples_per_q == -1:
            n_samples_per_q = int(h.shape[0])
        else:
            if int(h.shape[0]) != n_samples_per_q:
                raise ValueError(f"Inconsistent samples per question: {h.shape[0]} vs {n_samples_per_q}")
        hidden_reps_list.append(h)
        logits_list.append(l)
        predictions_list.append(p)

    hidden_reps = np.vstack(hidden_reps_list) if hidden_reps_list else np.empty((0,))
    logits = np.vstack(logits_list) if logits_list else np.empty((0,))
    predictions = np.concatenate(predictions_list) if predictions_list else np.empty((0,))
    return hidden_reps, logits, predictions, question_ids, n_samples_per_q


class ProxyModel(nn.Module):
    """A 2-layer MLP proxy model for LLM MCQ prediction."""
    def __init__(self, input_dim: int, num_classes: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == y).float().mean().item()


def ece_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float(compute_squared_error_label_binning_pytorch(logits, y, m=15, temperature=0.1).item())


def kl_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    t_prob = torch.softmax(teacher_logits / T, dim=-1)
    s_logprob = torch.log_softmax(student_logits / T, dim=-1)
    return (T * T) * torch.nn.functional.kl_div(s_logprob, t_prob, reduction="batchmean")


def train_proxy_model(
    hidden_reps: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    input_dim: int,
    *,
    debug: bool = False,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    epochs: int = 50,
    patience: int = 5,
    lambda_ece: float = 1.0,
    seed: int = 42,
    use_kl_distill: bool = False,
    kl_weight: float = 0.0,
    distill_temperature: float = 1.0,
    model_save_path: str = "proxy_model.pth",
) -> Dict[str, Any]:
    """
    Train the proxy model using cross-entropy plus optional soft-binned ECE and KL distillation.
    Saves artifacts and returns metrics.
    """
    set_all_seeds(seed)

    # Get training and validation splits by question
    n_samples_per_q = 7
    num_questions = len(hidden_reps) // n_samples_per_q
    train_size = int(0.8 * num_questions)
    val_size = num_questions - train_size
    question_ids = list(range(num_questions))
    random.shuffle(question_ids)
    train_ids = question_ids[:train_size]
    val_ids = question_ids[train_size:]

    X_train = [hidden_reps[question_ids * n_samples_per_q:(question_ids + 1) * n_samples_per_q] for question_ids in train_ids]
    Y_train = [labels[question_ids * n_samples_per_q:(question_ids + 1) * n_samples_per_q] for question_ids in train_ids]
    X_train = torch.tensor(np.vstack(X_train), dtype=torch.float32, device=DEVICE)
    Y_train = torch.tensor(np.hstack(Y_train), dtype=torch.long, device=DEVICE)

    X_val = [hidden_reps[question_ids * n_samples_per_q:(question_ids + 1) * n_samples_per_q] for question_ids in val_ids]
    Y_val = [labels[question_ids * n_samples_per_q:(question_ids + 1) * n_samples_per_q] for question_ids in val_ids]
    X_val = torch.tensor(np.vstack(X_val), dtype=torch.float32, device=DEVICE)
    Y_val = torch.tensor(np.hstack(Y_val), dtype=torch.long, device=DEVICE)

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, loss, and optimizer
    model = ProxyModel(input_dim, num_classes=6).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_metrics = {}
    patience_counter = 0

    # Debug overrides
    if debug and epochs > 1:
        epochs = 1

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_kl = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            if use_kl_distill:
                batch_x, batch_y, batch_teacher = batch
            else:
                batch_x, batch_y = batch
                batch_teacher = None

            pred_logits = model(batch_x)
            loss_primary = criterion(pred_logits, batch_y)
            loss_secondary = compute_squared_error_label_binning_pytorch(pred_logits, batch_y, m=15, temperature=0.1)
            loss = loss_primary + lambda_ece * loss_secondary
            if use_kl_distill and kl_weight > 0.0:
                loss_kl = kl_divergence(pred_logits, batch_teacher, distill_temperature)
                train_kl += float(loss_kl.item()) * batch_x.size(0)
                loss = loss + kl_weight * loss_kl

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            train_correct += (pred_logits.argmax(dim=-1) == batch_y).sum().item()
            train_total += batch_x.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_ece_accum = 0.0
        val_kl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if use_kl_distill:
                    batch_x, batch_y, batch_teacher = batch
                else:
                    batch_x, batch_y = batch
                    batch_teacher = None
                pred_logits = model(batch_x)
                loss_primary = criterion(pred_logits, batch_y)
                loss_secondary = compute_squared_error_label_binning_pytorch(pred_logits, batch_y, m=15, temperature=0.1)
                loss = loss_primary + lambda_ece * loss_secondary
                if use_kl_distill and kl_weight > 0.0:
                    loss_kl = kl_divergence(pred_logits, batch_teacher, distill_temperature)
                    val_kl += float(loss_kl.item()) * batch_x.size(0)
                    loss = loss + kl_weight * loss_kl

                val_loss += loss.item() * batch_x.size(0)
                val_correct += (pred_logits.argmax(dim=-1) == batch_y).sum().item()
                val_total += batch_x.size(0)
                val_ece_accum += ece_from_logits(pred_logits, batch_y) * batch_x.size(0)

        # Aggregate
        train_loss /= max(1, train_total)
        val_loss /= max(1, val_total)
        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)
        val_ece = val_ece_accum / max(1, val_total)
        if use_kl_distill:
            train_kl /= max(1, train_total)
            val_kl /= max(1, val_total)

        if use_kl_distill:
            logging.info(
                f"[Epoch {epoch+1}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_ece={val_ece:.4f} "
                f"train_kl={train_kl:.4f} val_kl={val_kl:.4f}"
            )
        else:
            logging.info(
                f"[Epoch {epoch+1}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_ece={val_ece:.4f}"
            )

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_ece": val_ece,
            }
            if use_kl_distill:
                best_metrics.update({"train_kl": train_kl, "val_kl": val_kl})
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info("Early stopping triggered.")
            break

    # Save artifacts
    save_dir = Path(os.path.dirname(model_save_path) or "proxy_models")
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / (os.path.basename(model_save_path) or "proxy_model.pth")
    torch.save(best_state, model_path)
    logging.info(f"Best model saved at {model_path}")

    # Save metrics and run_info
    metrics = best_metrics
    with (save_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    run_info = {
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "patience": patience,
        "lambda_ece": lambda_ece,
        "seed": seed,
        "use_kl_distill": use_kl_distill,
        "kl_weight": kl_weight,
        "distill_temperature": distill_temperature,
        "device": str(DEVICE),
        "input_dim": int(input_dim),
        "num_samples": int(len(hidden_reps)),
    }
    with (save_dir / "run_info.json").open("w") as f:
        json.dump(run_info, f, indent=2)

    return {"model_path": str(model_path), "metrics_path": str(save_dir / "metrics.json"), "run_info_path": str(save_dir / "run_info.json")}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode: truncate and run 1 epoch.")
    parser.add_argument("--train_ids", type=str, default='calib_test_splits/train_ids_cosmosqa_10k_tr0p5_cf0p3_tf0p2.json',
                        help="Path to the training question IDs JSON file (from 03_generate_calib_test_split.py).")
    parser.add_argument("--hidden_reps", type=str, default='outputs_hidden_reps/cosmosqa_10k_results.pkl',
                        help="Path to the hidden reps & logits pickle file.")
    parser.add_argument("--ground_truth_json", type=str, default='data/cosmosqa_10k.json',
                        help="Path to ground-truth dataset JSON.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lambda_ece", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_kl_distill", action="store_true")
    parser.add_argument("--kl_weight", type=float, default=0.0)
    parser.add_argument("--distill_temperature", type=float, default=1.0)
    parser.add_argument("--model_save_path", type=str, default="")
    args = parser.parse_args()

    logging.basicConfig(level=(logging.DEBUG if args.debug else logging.INFO), format="%(levelname)s: %(message)s")

    set_all_seeds(args.seed)

    # Load ground truth labels
    ground_truth_labels = load_ground_truth_labels(args.ground_truth_json, debug=args.debug)

    # Load hidden representations and logits
    hidden_reps, logits, predicted_answers, question_ids, n_samples_per_q = load_hidden_representations(args.hidden_reps, debug=args.debug)

    # Load training question IDs and normalize to str
    with open(args.train_ids, "r") as f:
        train_ids = {to_str_id(x) for x in json.load(f)}

    # Filter by train_ids using robust indexing (only use train questions here)
    filtered_hidden_reps_list: List[np.ndarray] = []
    filtered_logits_list: List[np.ndarray] = []
    filtered_question_ids: List[str] = []

    for i, q_id in enumerate(question_ids):
        if q_id in train_ids:
            start = i * n_samples_per_q
            end = (i + 1) * n_samples_per_q
            filtered_hidden_reps_list.append(hidden_reps[start:end])
            filtered_logits_list.append(logits[start:end])
            filtered_question_ids.append(q_id)

    if filtered_hidden_reps_list:
        hidden_reps = np.concatenate(filtered_hidden_reps_list, axis=0)
        logits = np.concatenate(filtered_logits_list, axis=0)
        question_ids = filtered_question_ids
    else:
        logging.error("No matching question IDs found in the training IDs file.")
        exit(1)

    # Labels per sample
    labels: List[int] = []
    missing_label_ids: List[str] = []
    for q_id in question_ids:
        if q_id not in ground_truth_labels:
            missing_label_ids.append(q_id)
            continue
        labels.extend([ground_truth_labels[q_id]] * n_samples_per_q)

    if missing_label_ids:
        logging.error(f"Missing ground-truth labels for {len(missing_label_ids)} IDs (e.g., {missing_label_ids[:5]}).")
        exit(1)

    labels = np.array(labels)

    # Debug info
    if args.debug:
        class_counts = np.bincount(labels, minlength=6)
        logging.debug(f"Questions: {len(question_ids)}, n_samples_per_q: {n_samples_per_q}, total rows: {hidden_reps.shape[0]}")
        logging.debug(f"Hidden reps shape: {hidden_reps.shape}, Logits shape: {logits.shape}")
        logging.debug(f"Label distribution: {class_counts.tolist()}")

    # Validate input_dim
    input_dim = int(hidden_reps.shape[1])

    # Train
    artifacts = train_proxy_model(
        hidden_reps=hidden_reps,
        logits=logits,
        labels=labels,
        input_dim=input_dim,
        debug=args.debug,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        lambda_ece=args.lambda_ece,
        seed=args.seed,
        use_kl_distill=args.use_kl_distill,
        kl_weight=args.kl_weight,
        distill_temperature=args.distill_temperature,
        model_save_path=(args.model_save_path or os.path.join("proxy_models", f"proxy_model_{Path(args.train_ids).stem}.pth")),
    )

    print(json.dumps(artifacts, indent=2))