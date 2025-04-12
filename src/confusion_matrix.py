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

    return cm

def matrix_accuracy(cm, activity_labels):
    # Compute accuracy from the confusion matrix
    labels = activity_labels["label"].values

    print("*************************** Confusion Matrix Accuracy **********************")
    # Display class-wise accuracy
    print("Per-Class Accuracy (Recall): \n")
    for i in range(len(labels)):
        total = sum(cm[i])  # total actual samples for this class
        correct = cm[i][i]  # correct predictions (diagonal element)
        accuracy = correct / total * 100  # accuracy % for the class
        print(labels[i], ":", correct, "out of", total, "correct -->", round(accuracy, 2), "%")