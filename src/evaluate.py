import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/tourism.csv")

X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = joblib.load("model/model.pkl")
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
