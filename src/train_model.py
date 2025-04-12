from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def train_svc_model(x_train, y_train):
    # Train the SVM model on the normalized training data
    clf = svm.SVC()  # initialize classifier with default parameters
    clf.fit(x_train, y_train.values.ravel())  # flatten y to avoid shape issues
    return clf

def predict_and_evaluate(clf, x_test, y_test, activity_labels):
    # Use the trained model to predict activity labels on the test set
    y_pred = clf.predict(x_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=activity_labels["label"].values)

    classifier_name = type(clf).__name__
    # Print Accuracy and Classification report
    print(f'\n****************** Training {classifier_name} Model ***************')
    print("\nAccuracy on test set: {:.2f}%".format(accuracy * 100))
    print("\nClassification Report: ")
    print(report)

    return y_pred, accuracy,report

def train_random_forest_model(x_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x_train, y_train.values.ravel())
    return clf

def compare_models(svc_accuracy, rf_accuracy):
    # Accuracy values
    model_names = ['SVM', 'Random Forest']
    accuracies = [svc_accuracy * 100, rf_accuracy * 100]  # note: append *100 because it's a float value

    # create bar chart
    plt.figure(figsize=(6, 4))
    bars = plt.bar(model_names, accuracies, color=['royalblue', 'forestgreen'])

    # add accuracy labels above bars
    for bar, acc in zip(bars, accuracies):
        y_val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, y_val + 0.5, f'{acc:.2f}%', ha='center', va='bottom')

    # labels and title
    plt.ylim(90, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison: SVM vs Random Forest ")
    plt.tight_layout()
    plt.show()