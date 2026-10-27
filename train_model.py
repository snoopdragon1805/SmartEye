import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit


DATASET_PATH = "dataset/fatigue_data.csv"
FEATURES = [
    "mean_ear",
    "min_ear",
    "blink_rate",
    "mean_blink_duration",
    "max_blink_duration",
    "perclos",
]
LABELS = [0, 1, 2]


def train_and_evaluate(dataset_path=DATASET_PATH):
    """Train and evaluate only when a participant-independent split exists."""
    df = pd.read_csv(dataset_path)
    required_columns = ["participant_id", *FEATURES, "fatigue_label"]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")
    if df.empty:
        raise ValueError("Dataset contains no labeled observations.")

    original_rows = len(df)
    df = df.dropna(subset=required_columns).copy()
    if len(df) != original_rows:
        print(
            f"Dropped {original_rows - len(df)} rows with missing "
            "required values."
        )
    if df.empty:
        raise ValueError("No complete labeled observations remain.")
    labels = pd.to_numeric(df["fatigue_label"], errors="raise")
    if not np.isfinite(labels.to_numpy(dtype=float)).all() or (
        labels % 1 != 0
    ).any():
        raise ValueError("fatigue_label values must be finite integers.")
    df["fatigue_label"] = labels.astype(int)
    if not set(df["fatigue_label"].unique()).issubset(LABELS):
        raise ValueError("fatigue_label values must be 0, 1, or 2.")
    features = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Dataset features must be finite.")

    print("Dataset shape:", df.shape)
    print("Class distribution:")
    print(df["fatigue_label"].value_counts().reindex(LABELS, fill_value=0))
    print("Participants:", df["participant_id"].nunique())

    if df["participant_id"].nunique() < 2:
        print(
            "\nTraining/evaluation skipped: at least two participants are "
            "required for a participant-independent holdout. This dataset "
            "contains only one participant."
        )
        return None

    if set(df["fatigue_label"].unique()) != set(LABELS):
        print(
            "\nTraining/evaluation skipped: labeled examples for all "
            "three classes (0, 1, and 2) are required."
        )
        return None

    features = features.astype(float)
    target = df["fatigue_label"]
    groups = df["participant_id"]
    splitter = GroupShuffleSplit(
        n_splits=100,
        test_size=0.25,
        random_state=42,
    )

    split = None
    for train_index, test_index in splitter.split(features, target, groups):
        train_labels = set(target.iloc[train_index])
        test_labels = set(target.iloc[test_index])
        if train_labels == set(LABELS) and test_labels == set(LABELS):
            split = train_index, test_index
            break

    if split is None:
        print(
            "\nTraining/evaluation skipped: no participant-independent "
            "holdout contains all three classes in both partitions."
        )
        return None

    train_index, test_index = split
    train_participants = set(groups.iloc[train_index])
    test_participants = set(groups.iloc[test_index])
    if train_participants & test_participants:
        raise RuntimeError("Participant leakage detected in validation split.")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(features.iloc[train_index], target.iloc[train_index])
    predictions = model.predict(features.iloc[test_index])

    print("\nParticipant-independent holdout")
    print("Training participants:", len(train_participants))
    print("Testing participants:", len(test_participants))
    print("Testing samples:", len(test_index))
    print(
        classification_report(
            target.iloc[test_index],
            predictions,
            labels=LABELS,
            target_names=["Not Tired", "Tired", "Very Tired"],
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(
        confusion_matrix(
            target.iloc[test_index],
            predictions,
            labels=LABELS,
        )
    )

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES,
    ).sort_values(ascending=False)
    print("Feature importance:")
    print(importance)
    print(
        "\nMetrics describe only this participant-independent split; "
        "they are not a clinical or general accuracy claim."
    )
    return model


if __name__ == "__main__":
    train_and_evaluate()
