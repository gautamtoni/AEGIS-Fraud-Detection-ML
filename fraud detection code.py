import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import time

# -----------------------------------------
# 1. LOAD DATA
# -----------------------------------------
df = pd.read_csv("transactions.csv")
print("\nDataset Loaded Successfully!\n")
print(df.head())

# -----------------------------------------
# 2. CHECK IF 'isFraud' EXISTS
# -----------------------------------------
if "isFraud" not in df.columns:
    raise ValueError("ERROR: Your CSV does NOT contain 'isFraud' column (target variable).")

# -----------------------------------------
# 3. ENCODE STRING COLUMNS
# -----------------------------------------
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col].astype(str))

print("\nData After Encoding:\n")
print(df.head())

# -----------------------------------------
# 4. SPLIT FEATURES & TARGET
# -----------------------------------------
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# -----------------------------------------
# 5. TRAIN-TEST SPLIT
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------
# 6. APPLY SMOTE
# -----------------------------------------
try:
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print("\nSMOTE Applied Successfully!")
except:
    print("\n⚠ SMOTE Failed (Too few samples). Proceeding without SMOTE...")

# -----------------------------------------
# 7. FEATURE SCALING
# -----------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------------------
# 8. TRAIN RANDOM FOREST
# -----------------------------------------
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# -----------------------------------------
# 9. PREDICTIONS
# -----------------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------------------
# 10. CONFUSION MATRIX
# -----------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------------------
# 11. FEATURE IMPORTANCE
# -----------------------------------------
importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
sns.barplot(x=importance, y=features, palette="viridis")
plt.title("Feature Importance in Fraud Detection")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# -----------------------------------------
# 12. REAL-TIME FRAUD STREAM SIMULATION
# -----------------------------------------
def realtime_fraud_check():
    sample = X.sample(1)
    sample_scaled = scaler.transform(sample)
    result = model.predict(sample_scaled)[0]

    print("\n🔵 New Incoming Transaction")
    print(sample)

    if result == 1:
        print("⚠ FRAUD ALERT: High-risk transaction detected!")
    else:
        print("✔ SAFE: Transaction is normal.")

print("\n🔁 Starting Real-Time Monitoring...\n")
for i in range(5):
    realtime_fraud_check()
    time.sleep(2)
