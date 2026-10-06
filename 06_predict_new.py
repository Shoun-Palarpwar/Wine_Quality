"""
Step 8: Predict wine quality on new/unseen samples
"""
import pandas as pd
import joblib

model = joblib.load("./rf_tuned_model.joblib")
X_train = pd.read_csv("./X_train.csv")
feature_names = list(X_train.columns)

# A few illustrative "new" wine samples (hand-picked to represent a poor,
# an average, and a premium wine profile based on the EDA findings)
new_samples = pd.DataFrame([
    # low alcohol, high volatile acidity -> expect lower predicted quality
    {"fixed acidity": 7.8, "volatile acidity": 0.88, "citric acid": 0.0, "residual sugar": 2.6,
     "chlorides": 0.098, "free sulfur dioxide": 25, "total sulfur dioxide": 67, "density": 0.9968,
     "pH": 3.2, "sulphates": 0.68, "alcohol": 9.8},
    # mid-range, balanced profile -> expect average quality
    {"fixed acidity": 7.2, "volatile acidity": 0.50, "citric acid": 0.25, "residual sugar": 2.2,
     "chlorides": 0.08, "free sulfur dioxide": 15, "total sulfur dioxide": 45, "density": 0.9965,
     "pH": 3.30, "sulphates": 0.62, "alcohol": 10.2},
    # high alcohol, high sulphates, low volatile acidity -> expect higher predicted quality
    {"fixed acidity": 8.5, "volatile acidity": 0.28, "citric acid": 0.45, "residual sugar": 2.0,
     "chlorides": 0.06, "free sulfur dioxide": 12, "total sulfur dioxide": 30, "density": 0.9950,
     "pH": 3.35, "sulphates": 0.90, "alcohol": 13.0},
], columns=feature_names)

predictions = model.predict(new_samples)
probabilities = model.predict_proba(new_samples)

results = new_samples.copy()
results["Predicted Quality"] = predictions
for i, cls in enumerate(model.classes_):
    results[f"P(quality={cls})"] = probabilities[:, i].round(3)

results.to_csv("./new_wine_predictions.csv", index=False)

print("Predictions for 3 new wine samples:\n")
for i, row in results.iterrows():
    print(f"Sample {i+1}: alcohol={row['alcohol']}, volatile acidity={row['volatile acidity']}, "
          f"sulphates={row['sulphates']}")
    print(f"  -> Predicted quality: {row['Predicted Quality']}")
    top_probs = probabilities[i].argsort()[::-1][:3]
    prob_str = ", ".join(f"q={model.classes_[j]}: {probabilities[i][j]:.2f}" for j in top_probs)
    print(f"  -> Top class probabilities: {prob_str}\n")

print("Saved: new_wine_predictions.csv")
