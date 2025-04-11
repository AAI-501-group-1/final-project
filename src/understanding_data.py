

def understanding_data(x_train, y_train, x_test, y_test, features, activity_labels):
    print("Number of features in X_train:", x_train.shape[1])
    print("Number of features in y_train:", y_train.shape[1])
    print("Total number of features: ", features.shape[0])
    # Preview the first few rows of the training feature set
    print("\nX-train set preview:")
    print(x_train.head())
    print("\nFeatures set preview:")
    print(features.head())
    print("\nTraining feature set shape (X_train):", x_train.shape)
    print("Training label set shape (y_train):", y_train.shape)
    print("Test feature set shape (X_test):", x_test.shape)
    print("Test label set shape (y_test):", y_test.shape)
    # Display the activity label mapping
    # representing different human activities
    print("\nActivity labels:")
    print(activity_labels)
    # Check for missing values in the training and test sets
    print("\nMissing values in X_train: ", x_train.isnull().sum().sum())
    print("Missing values in X_test: ", x_test.isnull().sum().sum())