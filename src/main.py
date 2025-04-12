from load_data import load_wine_data_from_local
from understanding_data import understanding_data
from normilaze_data import normalize_data
from draw_boxplot import draw_boxplot
from check_duplicates import check_duplicates_and_missing_values
from inspect_row_data import inspect_row_data
from train_model import train_svc_model, predict_and_evaluate
from confusion_matrix import draw_confusion_matrix

import pandas as pd

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)

x_train, y_train, x_test, y_test, features, activity_labels = load_wine_data_from_local()
understanding_data(x_train, y_train, x_test, y_test, features, activity_labels)
x_train_scaled, x_test_scaled = normalize_data(x_train, x_test)
draw_boxplot(x_train_scaled, features)
check_duplicates_and_missing_values(x_train, x_test)
inspect_row_data(activity_labels)

svc_clf = train_svc_model(x_train_scaled, y_train)
y_pred, accuracy,report = predict_and_evaluate(svc_clf, x_test_scaled, y_test, activity_labels)

# Print Accuracy and Classification report
print("****************** Training SVC Model ***************")
print("\nAccuracy on test set: {:.2f}%".format(accuracy * 100))
print("\nClassification Report: ")
print(report)

draw_confusion_matrix(y_test, y_pred, activity_labels)