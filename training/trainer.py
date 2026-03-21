import os
import torch
from torch_geometric.loader import DataLoader


class Trainer:
    """Handles the training loop, validation, checkpointing, and early stopping."""

    def __init__(self, model, train_dataset, val_dataset, config):
        self.device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        tc = config["training"]
        self.epochs = tc["epochs"]
        self.patience = tc["patience"]

        self.train_loader = DataLoader(train_dataset, batch_size=tc["batch_size"], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=tc["batch_size"])

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"]
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=tc["scheduler"]["factor"],
            patience=tc["scheduler"]["patience"],
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        self.checkpoint_dir = config["logging"]["checkpoint_dir"]
        self.save_every = config["logging"]["save_every"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(out, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        for batch in self.val_loader:
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.num_graphs
        return total_loss / total, correct / total

    def save_checkpoint(self, epoch, val_loss):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }, path)

    def fit(self):
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step(val_loss)

            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1

            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch, val_loss)

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        return best_val_loss
