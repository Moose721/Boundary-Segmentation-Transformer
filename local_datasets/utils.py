import torch

def f1_score(preds, labels):
    #preds = torch.tensor([0, 1, 1, 0, 1])
    #labels = torch.tensor([0, 1, 0, 0, 1])

    # Calculate True Positives, False Positives, False Negatives
    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()

    # Calculate Precision and Recall with epsilon to avoid division by zero
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    # Calculate F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1, precision, recall, tp, fp, fn
