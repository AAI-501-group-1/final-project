import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def draw_boxplot(x_train_scaled, features):
    # Inspect for outliers
    # Box plot to visually check for outliers
    # First 10 features (just a representative sample)
    # If outliers exist in the data

    X_train_box = pd.DataFrame(x_train_scaled[:, :10],
                               columns=features["feature_name"][:10])

    # Box plot for first 10 features
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=X_train_box, orient='h')
    plt.title("Box Plot of First 10 Scaled Features (Training Data)")
    plt.xlabel("Scaled Feature Value")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    plt.show()