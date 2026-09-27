# Adopted from https://github.com/google/uncertainty-baselines/blob/main/experimental/caltrain/secondary_losses.py

import math
import torch
import numpy as np

EPS = 1e-5

def compute_squared_error_label_binning_pytorch(logits, y, m, temperature, precision=32):
    """
    Computes the squared soft-binned ECE (label-binned) loss.

    Args:
      logits: Tensor of predicted logits of shape [N, num_classes].
      y: Tensor of integer labels of shape [N].
      m: Number of bins.
      temperature: Soft binning temperature.
      precision: Either 32 or 64.

    Returns:
      A scalar tensor representing the squared soft-binned ECE.
    """
    if precision == 32:
        float_dtype = torch.float32
        int_dtype = torch.int32
    elif precision == 64:
        float_dtype = torch.float64
        int_dtype = torch.int64
    else:
        raise NotImplementedError

    # Cast tensors to desired types.
    logits = logits.type(float_dtype)
    y = y.type(int_dtype)

    # Create bins and compute bin centers.
    num_classes = logits.shape[1]
    bins = torch.linspace(1.0 / num_classes, 1.0, m + 1, dtype=float_dtype, device=logits.device)
    b = 0.5 * bins[1:] + 0.5 * bins[:-1]  # shape: [m]

    # Softmax over logits.
    q_hat = torch.softmax(logits, dim=1)
    # Predicted labels and corresponding confidence.
    y_hat = torch.argmax(q_hat, dim=1)
    p_hat = torch.max(q_hat, dim=1).values
    # Indicator for correct predictions.
    a = (y_hat == y).to(float_dtype)  # shape: [N]

    # Compute soft binning coefficients.
    # Expand dimensions: p_hat [N] -> [N, 1] and b [m] -> [1, m]
    c_numerator = torch.exp(-((b.unsqueeze(0) - p_hat.unsqueeze(1)) ** 2) / temperature)
    c_denominator = torch.sum(c_numerator, dim=1, keepdim=True)
    c_pre = c_numerator / c_denominator
    c = (1 - EPS) * c_pre + EPS * (1 / m)

    # Compute bin-average accuracy.
    # For each bin j: a_bar[j] = sum_i(c[i,j]*a[i]) / sum_i c[i,j]
    a_bar_numerator = torch.sum(c * a.unsqueeze(1), dim=0)
    a_bar_denominator = torch.sum(c, dim=0)
    a_bar = a_bar_numerator / a_bar_denominator  # shape: [m]

    # Squared error between bin-average accuracy and sample confidence.
    # Difference: for each sample i and bin j: (a_bar[j] - p_hat[i])^2.
    squared_error = torch.sum(c * ((a_bar.unsqueeze(0) - p_hat.unsqueeze(1)) ** 2))
    squared_error_scaled = squared_error / logits.shape[0]
    return squared_error_scaled


def get_soft_binning_ece_pytorch(predictions, labels, soft_binning_bins,
                                 soft_binning_use_decay, soft_binning_decay_factor,
                                 soft_binning_temp):
    """
    Computes the soft-binned ECE loss.

    Args:
      predictions: Tensor of predicted confidences of shape [N] (values in [0,1]).
      labels: Tensor of binary labels (0 for incorrect, 1 for correct) of shape [N].
      soft_binning_bins: Number of bins.
      soft_binning_use_decay: Boolean; if True, determine temperature using decay factor.
      soft_binning_decay_factor: Approximate decay factor between successive bins.
      soft_binning_temp: Soft binning temperature.

    Returns:
      A scalar tensor representing the soft-binned ECE.
    """
    device = predictions.device
    dtype = predictions.dtype
    # Create bin anchors: from 1/(2*m) to 1 with step 1/m.
    anchors = torch.arange(1.0 / (2 * soft_binning_bins),
                           1.0, 1.0 / soft_binning_bins,
                           device=device, dtype=dtype)
    if soft_binning_use_decay:
        soft_binning_temp = 1 / (math.log(soft_binning_decay_factor) *
                                 soft_binning_bins * soft_binning_bins)

    # Compute differences between prediction and each bin anchor.
    # differences shape: [N, num_bins]
    differences = -((predictions.unsqueeze(1) - anchors.unsqueeze(0)) ** 2) / soft_binning_temp

    # Soft binning coefficients via softmax over bins.
    predictions_soft_binning_coeffs = torch.softmax(differences, dim=1)  # [N, num_bins]

    # Sum of coefficients for each bin (over the batch).
    sum_coeffs_for_bin = predictions_soft_binning_coeffs.sum(dim=0)  # [num_bins]

    # Repeat predictions and labels along the bin dimension.
    intermediate_predictions = predictions.unsqueeze(1).expand(-1, anchors.shape[0])
    net_bin_confidence = (intermediate_predictions * predictions_soft_binning_coeffs).sum(dim=0) \
                         / torch.maximum(sum_coeffs_for_bin, torch.tensor(EPS, device=device, dtype=dtype))

    intermediate_labels = labels.unsqueeze(1).expand(-1, anchors.shape[0])
    net_bin_accuracy = (intermediate_labels * predictions_soft_binning_coeffs).sum(dim=0) \
                       / torch.maximum(sum_coeffs_for_bin, torch.tensor(EPS, device=device, dtype=dtype))

    # Normalize bin weights (L1 normalization).
    bin_weights = sum_coeffs_for_bin / (sum_coeffs_for_bin.sum() + EPS)

    soft_binning_ece = torch.sqrt(torch.sum((net_bin_confidence - net_bin_accuracy) ** 2 * bin_weights))
    return soft_binning_ece


def get_avuc_loss_pytorch(probabilities, labels, avuc_stop_prob_gradients,
                          avuc_entropy_threshold):
    """
    Computes the AvUC loss.

    Args:
      probabilities: Tensor of predicted probabilities of shape [N, num_classes].
      labels: Tensor of integer labels of shape [N].
      avuc_stop_prob_gradients: If True, use gradient stopping on probabilities.
      avuc_entropy_threshold: Entropy threshold.

    Returns:
      A scalar tensor representing the AvUC loss.
    """
    device = probabilities.device
    dtype = probabilities.dtype
    # Optionally stop gradient on confidences.
    if avuc_stop_prob_gradients:
        confidences = probabilities.max(dim=1).values.detach()
    else:
        confidences = probabilities.max(dim=1).values

    accuracies = (probabilities.argmax(dim=1) == labels).float()  # [N]

    num_classes = probabilities.shape[1]
    uniform_probabilities = torch.full((num_classes,), 1.0 / num_classes,
                                       dtype=dtype, device=device)
    log_safe_probabilities = (1 - EPS) * probabilities + EPS * uniform_probabilities
    log_probabilities = torch.log(log_safe_probabilities)
    entropies = -(log_safe_probabilities * log_probabilities).sum(dim=1)

    # Create masks for different conditions.
    ac_mask = (entropies < avuc_entropy_threshold) & (accuracies > 0.5)
    au_mask = (entropies >= avuc_entropy_threshold) & (accuracies > 0.5)
    ic_mask = (entropies < avuc_entropy_threshold) & (accuracies < 0.5)
    iu_mask = (entropies >= avuc_entropy_threshold) & (accuracies < 0.5)

    # Compute differences as in the paper.
    nac_diff = (confidences * (1 - torch.tanh(entropies)) * ac_mask.float()).sum()
    nau_diff = (confidences * torch.tanh(entropies) * au_mask.float()).sum()
    nic_diff = ((1 - confidences) * (1 - torch.tanh(entropies)) * ic_mask.float()).sum()
    niu_diff = ((1 - confidences) * torch.tanh(entropies) * iu_mask.float()).sum()

    denominator = torch.maximum(nac_diff + niu_diff,
                                torch.tensor(EPS, device=device, dtype=dtype))
    avuc_loss = torch.log(1 + ((nau_diff + nic_diff) / denominator))
    return avuc_loss


def get_soft_avuc_loss_pytorch(probabilities, labels, soft_avuc_use_deprecated_v0,
                               soft_avuc_temp, soft_avuc_theta):
    """
    Computes the soft AvUC loss.

    Args:
      probabilities: Tensor of predicted probabilities of shape [N, num_classes].
      labels: Tensor of integer labels of shape [N].
      soft_avuc_use_deprecated_v0: Boolean flag for deprecated formulation.
      soft_avuc_temp: Temperature parameter (> 0).
      soft_avuc_theta: Threshold parameter in (0, 1).

    Returns:
      A scalar tensor representing the soft AvUC loss.
    """
    device = probabilities.device
    dtype = probabilities.dtype

    accuracies = (probabilities.argmax(dim=1) == labels).float()
    num_classes = probabilities.shape[1]
    uniform_probabilities = torch.full((num_classes,), 1.0 / num_classes,
                                       dtype=dtype, device=device)
    log_safe_probabilities = (1 - EPS) * probabilities + EPS * uniform_probabilities
    log_probabilities = torch.log(log_safe_probabilities)
    entropies = -(log_safe_probabilities * log_probabilities).sum(dim=1)
    entmax = math.log(num_classes)

    def soft_uncertainty(e, temp=1, theta=0.5):
        # Avoid division by zero by clamping e.
        e = torch.clamp(e, min=EPS, max=1 - EPS)
        return torch.sigmoid((1 / temp) * torch.log(e * (1 - theta) / ((1 - e) * theta)))

    if soft_avuc_use_deprecated_v0:
        xus = -((entropies - entmax) ** 2)
        xcs = -(entropies ** 2)
        # Stack along a new dimension and compute softmax.
        qucs = torch.softmax(torch.stack([xus, xcs], dim=1), dim=1)
        qus = qucs[:, 0]
        qcs = qucs[:, 1]
    else:
        # Compute soft uncertainty elementwise.
        qus = torch.stack([soft_uncertainty(ent / entmax, temp=soft_avuc_temp, theta=soft_avuc_theta)
                           for ent in entropies])
        qcs = 1 - qus

    # Masks based on accuracy.
    mask_correct = (accuracies > 0.5)
    mask_incorrect = (accuracies < 0.5)
    # Compute differences.
    nac_diff = (qcs * (1 - torch.tanh(entropies)) * mask_correct.float()).sum()
    nau_diff = (qus * torch.tanh(entropies) * mask_correct.float()).sum()
    nic_diff = (qcs * (1 - torch.tanh(entropies)) * mask_incorrect.float()).sum()
    niu_diff = (qus * torch.tanh(entropies) * mask_incorrect.float()).sum()

    denominator = torch.maximum(nac_diff + niu_diff,
                                torch.tensor(EPS, device=device, dtype=dtype))
    avuc_loss = torch.log(1 + ((nau_diff + nic_diff) / denominator))
    return avuc_loss
