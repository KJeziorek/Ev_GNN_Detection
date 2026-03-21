import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from torch_geometric.loader import DataLoader


class Evaluator:
    """Evaluate a trained model on a test dataset."""

    def __init__(self, model, test_dataset, device="cpu", batch_size=32):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.loader = DataLoader(test_dataset, batch_size=batch_size)

    @torch.no_grad()
    def predict(self):
        all_preds = []
        all_labels = []
        for batch in self.loader:
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index, batch.batch)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())
        return all_labels, all_preds

    def evaluate(self):
        labels, preds = self.predict()
        results = {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average="weighted"),
            "recall": recall_score(labels, preds, average="weighted"),
            "f1": f1_score(labels, preds, average="weighted"),
            "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        }
        print(classification_report(labels, preds))
        return results
