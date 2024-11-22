from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

print(X,y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a base decision tree classifier
base_tree = DecisionTreeClassifier()

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

base_tree = DecisionTreeClassifier()
bagging_clf = BaggingClassifier(estimator=base_tree, n_estimators=10, random_state=42)
# Train the bagging classifier
bagging_clf.fit(X_train, y_train)

# Make predictions
y_pred = bagging_clf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of Bagging with Decision Tree: {accuracy:.2f}')

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3, criterion='entropy')  # Example parameters
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
print("Accuracy:", accuracy_score(y_test, y_pred))