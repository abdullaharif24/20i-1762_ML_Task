from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset and train model
iris = load_iris()
model = RandomForestClassifier()
model.fit(iris.data, iris.target)

# Save the model
joblib.dump(model, 'model.pkl')
