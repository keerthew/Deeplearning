import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = iris.target
X = df.iloc[:, 0:4]
Y = df['Species']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=4)

rf = RandomForestClassifier(n_estimators=120, random_state=4)
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
print("Accuracy:", acc)
cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='magma',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix using Random Forest")
plt.show()
