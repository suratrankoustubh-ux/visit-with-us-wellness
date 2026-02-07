import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

os.makedirs("model", exist_ok=True)

df = pd.read_csv("data/tourism.csv")

X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "model/model.pkl")

print("Model training completed and saved")
