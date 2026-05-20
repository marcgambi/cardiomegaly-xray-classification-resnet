import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score


def evaluate_model(model, test_loader, device, model_path, save_path):
    """
    Evaluate a trained binary classification model on the test set.
    """

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()

    test_losses = []
    test_accuracies = []

    all_predictions = []
    all_labels = []
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()

            outputs = model(batch_x).view(-1)
            loss = criterion(outputs, batch_y)

            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()

            accuracy = (predictions == batch_y).float().mean().item()

            test_losses.append(loss.item())
            test_accuracies.append(accuracy)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_outputs.extend(probabilities.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    mean_test_loss = sum(test_losses) / len(test_losses)
    mean_test_accuracy = sum(test_accuracies) / len(test_accuracies)

    test_precision = precision_score(all_labels, all_predictions)
    test_recall = recall_score(all_labels, all_predictions)
    test_f1 = f1_score(all_labels, all_predictions)

    print(f"Test Loss: {mean_test_loss:.4f}")
    print(f"Test Accuracy: {mean_test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

    metrics = pd.DataFrame({
        "test_loss": [mean_test_loss],
        "test_accuracy": [mean_test_accuracy],
        "test_precision": [test_precision],
        "test_recall": [test_recall],
        "test_f1": [test_f1]
    })

    metrics.to_csv(f"{save_path}/test_metrics.csv", index=False)

    plot_confusion_matrix(all_labels, all_predictions, save_path)
    plot_test_curves(
        test_losses,
        test_accuracies,
        all_targets,
        all_outputs,
        save_path
    )

    return metrics


def plot_confusion_matrix(labels, predictions, save_path):
    """
    Plot and save the confusion matrix.
    """

    cm = confusion_matrix(labels, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)

    display.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{save_path}/confusion_matrix.png")
    plt.show()


def plot_test_curves(test_losses, test_accuracies, targets, outputs, save_path):
    """
    Plot test loss, accuracy, ROC curve and precision-recall curve.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(test_losses, label="Test Loss per Batch")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Test Loss per Batch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/test_loss_plot.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(test_accuracies, label="Test Accuracy per Batch")
    plt.xlabel("Batch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy per Batch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/test_accuracy_plot.png")
    plt.show()

    fpr, tpr, _ = roc_curve(targets, outputs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{save_path}/ROC.png")
    plt.show()

    precision, recall, _ = precision_recall_curve(targets, outputs)
    avg_precision = average_precision_score(targets, outputs)

    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, label=f"AP = {avg_precision:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{save_path}/PR.png")
    plt.show()
