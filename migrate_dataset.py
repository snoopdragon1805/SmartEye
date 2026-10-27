
import pandas as pd

path = "dataset/fatigue_data.csv"

df = pd.read_csv(path)

new_columns = ["ear_std", "ear_range"]

for column in new_columns:
    if column not in df.columns:
        df[column] = pd.NA

column_order = [
    "participant_id",
    "session_id",
    "window_id",
    "mean_ear",
    "min_ear",
    "ear_std",
    "ear_range",
    "blink_rate",
    "mean_blink_duration",
    "max_blink_duration",
    "perclos",
    "fatigue_label",
]

df = df[column_order]
df.to_csv(path, index=False)

print("Dataset migration complete.")
print("Columns:", list(df.columns))
print("Rows:", len(df))

