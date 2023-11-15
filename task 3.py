import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE

# Load the data
data = pd.read_csv('customer_churn.csv')

# Separate features and target variable
X = data.drop('Churn', axis=1)
y = data['Churn']

# Handle imbalanced dataset using SMOTE
oversampler = SMOTE(random_state=0)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Standardize numerical features
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train and evaluate Logistic Regression model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred_logistic = logistic_regression.predict(X_test)

print('Logistic Regression:')
print('Accuracy:', accuracy_score(y_test, y_pred_logistic))
print('Recall:', recall_score(y_test, y_pred_logistic))
print('Precision:', precision_score(y_test, y_pred_logistic))
print('F1-score:', f1_score(y_test, y_pred_logistic))

# Train and evaluate Random Forests model
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred_random_forest = random_forest.predict(X_test)

print('\nRandom Forests:')
print('Accuracy:', accuracy_score(y_test, y_pred_random_forest))
print('Recall:', recall_score(y_test, y_pred_random_forest))
print('Precision:', precision_score(y_test, y_pred_random_forest))
print('F1-score:', f1_score(y_test, y_pred_random_forest))

# Train and evaluate Gradient Boosting model
gradient_boosting = GradientBoostingClassifier(n_estimators=100)
gradient_boosting.fit(X_train, y_train)
y_pred_gradient_boosting = gradient_boosting.predict(X_test)

print('\nGradient Boosting:')
print('Accuracy:', accuracy_score(y_test, y_pred_gradient_boosting))
print('Recall:', recall_score(y_test, y_pred_gradient_boosting))
print('Precision:', precision_score(y_test, y_pred_gradient_boosting))
print('F1-score:', f1_score(y_test, y_pred_gradient_boosting))
