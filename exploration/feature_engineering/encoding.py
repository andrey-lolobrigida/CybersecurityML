import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

RAW_PATH = Path("data/raw/cybersecurity_intrusion_data.csv")
OUTPUT_PATH = Path("data/interim/encoded.csv")

df = pd.read_csv(RAW_PATH)
df.drop(columns=["session_id"], inplace=True)

# NaN in encryption_used means unencrypted
df["encryption_used"] = df["encryption_used"].fillna("Unencrypted")

cols_to_encode = ["protocol_type", "browser_type", "encryption_used"]

encoder = OneHotEncoder(sparse_output=False, dtype=int)
encoded_array = encoder.fit_transform(df[cols_to_encode])
encoded_cols = encoder.get_feature_names_out(cols_to_encode)

encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
df = pd.concat([df.drop(columns=cols_to_encode), encoded_df], axis=1)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved to {OUTPUT_PATH}")
print(f"Shape: {df.shape}")
print(f"New columns: {list(encoded_cols)}")