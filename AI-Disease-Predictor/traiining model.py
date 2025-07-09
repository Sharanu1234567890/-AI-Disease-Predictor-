import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and clean data
df = pd.read_csv("C:\\Users\\sharanu malipatil\\Downloads\\archive (12)\\dataset.csv")
df.fillna("None", inplace=True)

symptom_cols = [col for col in df.columns if col.startswith("Symptom_")]
all_symptoms = sorted(set(symptom.strip() for col in symptom_cols for symptom in df[col].unique() if symptom != "None"))

def encode_symptoms(row):
    present = set(symptom.strip() for symptom in row[symptom_cols] if symptom != "None")
    return [1 if symptom in present else 0 for symptom in all_symptoms]

X = df.apply(encode_symptoms, axis=1, result_type='expand')
y = df['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and symptom list
joblib.dump(model, "disease_model.pkl")
joblib.dump(all_symptoms, "symptoms_list.pkl")

print("✅ Model saved successfully.")

