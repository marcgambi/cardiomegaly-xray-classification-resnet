import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve, average_precision_score


def train(
    model,
    train_loader,
    val_loader,
    num_epochs,
    device,
    learning_rate,
    save_path,
    patience=15,
    patience_accuracy=15
):
    """
    Train and validate the binary classification model.
    """

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3
    )

    train_loss_history = []
    val_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []
    val_precision_history = []
    val_recall_history = []

    best_val_loss = float("inf")
    best_val_accuracy = 0

    patience_counter = 0
    accuracy_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        model.train()

        epoch_train_losses = []
        epoch_train_accuracies = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()

            outputs = model(batch_x).view(-1)
            loss = criterion(outputs, batch_y)

            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()

            accuracy = (predictions == batch_y).float().mean().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())
            epoch_train_accuracies.append(accuracy)

        mean_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        mean_train_accuracy = sum(epoch_train_accuracies) / len(epoch_train_accuracies)

        train_loss_history.append(mean_train_loss)
        train_accuracy_history.append(mean_train_accuracy)

        model.eval()

        epoch_val_losses = []
        epoch_val_accuracies = []
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float()

                outputs = model(batch_x).view(-1)
                loss = criterion(outputs, batch_y)

                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()

                accuracy = (predictions == batch_y).float().mean().item()

                epoch_val_losses.append(loss.item())
                epoch_val_accuracies.append(accuracy)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        mean_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        mean_val_accuracy = sum(epoch_val_accuracies) / len(epoch_val_accuracies)

        precision = precision_score(all_targets, all_predictions, zero_division=0)
        recall = recall_score(all_targets, all_predictions)

        val_loss_history.append(mean_val_loss)
        val_accuracy_history.append(mean_val_accuracy)
        val_precision_history.append(precision)
        val_recall_history.append(recall)

        scheduler.step(mean_val_loss)

        print(
            f"Train Loss: {mean_train_loss:.4f} | "
            f"Train Accuracy: {mean_train_accuracy:.4f}"
        )

        print(
            f"Validation Loss: {mean_val_loss:.4f} | "
            f"Validation Accuracy: {mean_val_accuracy:.4f} | "
            f"Precision: {precision:.4f} | Recall: {recall:.4f}"
        )

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{save_path}/best_model.pth")
            print(f"Best validation loss updated: {best_val_loss:.4f}")
        else:
            patience_counter += 1

        if mean_val_accuracy > best_val_accuracy:
            best_val_accuracy = mean_val_accuracy
            accuracy_counter = 0
            torch.save(model.state_dict(), f"{save_path}/best_model_accuracy.pth")
            print(f"Best validation accuracy updated: {best_val_accuracy:.4f}")
        else:
            accuracy_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered based on validation loss.")
            break

        if accuracy_counter >= patience_accuracy:
            print("Early stopping triggered based on validation accuracy.")
            break

    metrics = pd.DataFrame({
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "train_accuracy": train_accuracy_history,
        "val_accuracy": val_accuracy_history,
        "val_precision": val_precision_history,
        "val_recall": val_recall_history
    })

    metrics.to_csv(f"{save_path}/training_metrics.csv", index=False)

    plot_training_curves(
        train_loss_history,
        val_loss_history,
        train_accuracy_history,
        val_accuracy_history,
        save_path
    )

    return metrics


def plot_training_curves(
    train_loss,
    val_loss,
    train_accuracy,
    val_accuracy,
    save_path
):
    """
    Plot training and validation loss/accuracy.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/loss_plot.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracy, label="Training Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/accuracy_plot.png")
    plt.show()
