from load_data import load_wine_data_from_local
from understanding_data import understanding_data
from normilaze_data import normalize_data
from draw_boxplot import draw_boxplot
from check_duplicates import check_duplicates_and_missing_values
from inspect_row_data import inspect_row_data
from train_model import train_svc_model, predict_and_evaluate, train_random_forest_model, compare_models
from confusion_matrix import draw_confusion_matrix, matrix_accuracy
from five_fold_cross_validation import cross_validation_scores, draw_cv_results

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
svc_y_pred, svc_accuracy, svc_report = predict_and_evaluate(svc_clf, x_test_scaled, y_test, activity_labels)

cm = draw_confusion_matrix(y_test, svc_y_pred, activity_labels)
matrix_accuracy(cm, activity_labels)
cv_scores = cross_validation_scores(x_train_scaled, y_train)
draw_cv_results(cv_scores)

rf_clf = train_random_forest_model(x_train_scaled, y_train)
rf_y_pred, rf_accuracy, rf_report = predict_and_evaluate(rf_clf, x_test_scaled, y_test, activity_labels)

compare_models(svc_accuracy, rf_accuracy)