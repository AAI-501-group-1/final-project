def check_duplicates_and_missing_values(x_train, x_test):
    # Check for duplicates in training and test features
    print("Duplicate rows in X_train: ", x_train.duplicated().sum())
    print("Duplicate rows in X_test: ", x_test.duplicated().sum())

    # Check for missing values in training and test sets
    print("Missing values in X_train: ", x_train.isnull().sum().sum())
    print("Missing values in X_test: ", x_test.isnull().sum().sum())