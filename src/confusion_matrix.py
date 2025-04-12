from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def draw_confusion_matrix(y_test, y_pred, activity_labels):
    # Confusion Matrix
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create a labeled heatmap using activity names
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=activity_labels["label"].values,
                yticklabels=activity_labels["label"].values)

    plt.title("Confusion Matrix - SVM on UCI HAR Dataset")
    plt.xlabel("Predicted Activity")
    plt.ylabel("True Activity")
    plt.xticks(rotation=45)  # prevent overlapping labels
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()