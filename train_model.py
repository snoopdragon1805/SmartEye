import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit


DATASET_PATH = "dataset/fatigue_data.csv"


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATASET_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nClass distribution:")
print(df["fatigue_label"].value_counts().sort_index())


# --------------------------------------------------
# 2. Define features and target
# --------------------------------------------------

FEATURES = [
    "mean_ear",
    "min_ear",
    "blink_rate",
    "mean_blink_duration",
    "max_blink_duration",
    "perclos"
]

X = df[FEATURES]
y = df["fatigue_label"]

groups = df["session_id"]


# --------------------------------------------------
# 3. Split by session
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining classes:")
print(y_train.value_counts().sort_index())

print("\nTesting classes:")
print(y_test.value_counts().sort_index())


# --------------------------------------------------
# 4. Train Random Forest
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# --------------------------------------------------
# 5. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 6. Evaluation
# --------------------------------------------------

print("\n==============================")
print("Classification Report")
print("==============================")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=[
            "Not Tired",
            "Tired",
            "Very Tired"
        ],
        zero_division=0
    )
)


print("\n==============================")
print("Confusion Matrix")
print("==============================")

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1, 2]
    )
)


# --------------------------------------------------
# 7. Feature importance
# --------------------------------------------------

print("\n==============================")
print("Feature Importance")
print("==============================")

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print(importance)