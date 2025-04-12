from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report

def train_svc_model(x_train, y_train):
    # Train the SVM model on the normalized training data
    clf = svm.SVC()  # initialize classifier with default parameters
    clf.fit(x_train, y_train.values.ravel())  # flatten y to avoid shape issues
    return clf

def predict_and_evaluate(clf, x_test, y_test, activity_labels):
    # Use the trained SVM model to predict activity labels on the test set
    y_pred = clf.predict(x_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=activity_labels["label"].values)

    return y_pred, accuracy,report
