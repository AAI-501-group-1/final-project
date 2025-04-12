from sklearn.model_selection import cross_val_score
from sklearn import svm
import matplotlib.pyplot as plt

def cross_validation_scores(x_train, y_train):
    cv_scores = cross_val_score(svm.SVC(kernel='linear'), x_train,
                                y_train.values.ravel(), cv=5)

    print("\n***************************** 5-Fold Cross Validation Results **********************")
    # Check for mean accuracy and standard deviation from cross-validation
    print("Cross-Validation Accuracy Scores: \n", cv_scores)
    print("Mean CV Accuracy: ", round(cv_scores.mean() * 100, 2), "%")
    print("Standard Deviation: ", round(cv_scores.std() * 100, 2), "%")

    return cv_scores

def draw_cv_results(cv_scores):
    # Bar plot of the 5 accuracy scores
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, 6), cv_scores * 100, color='skyblue', edgecolor='black')

    # Plot the mean accuracy as a horizontal line
    plt.axhline(y=cv_scores.mean() * 100, color='red', linestyle='--',
                label=f'Mean = {round(cv_scores.mean() * 100, 2)}%')

    # Bar plot labels/formatting
    plt.title("5-Fold Cross-Validation Accuracy (SVM)")
    plt.xlabel("Fold Number")
    plt.ylabel("Accuracy (%)")
    plt.ylim(85, 100)  # focus on 85%-100% for better visual
    plt.xticks([1, 2, 3, 4, 5])  # label x-axis tick marks 1 to 5
    plt.legend()
    plt.tight_layout()
    plt.show()