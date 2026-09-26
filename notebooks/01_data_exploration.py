import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion.loader import load_train_data

def explore_data():
    try:
        data = load_train_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the TSV files are placed in the 'data/train/' and 'data/test/' folders as per the roadmap.")
        return

    print("\n--- Data Exploration ---")
    for name, df in data.items():
        print(f"\n[{name.upper()}]")
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for nulls/empty strings
        empty_counts = (df == "").sum()
        if empty_counts.sum() > 0:
            print("Empty string counts per column:")
            print(empty_counts[empty_counts > 0])
            
        # Check ID uniqueness if applicable
        id_col = "entity_id" if "entity_id" in df.columns else None
        if id_col:
            duplicates = df.duplicated(subset=[id_col]).sum()
            print(f"Duplicate IDs: {duplicates}")
            
        if "country" in df.columns:
            print(f"Top 5 Countries:")
            print(df["country"].value_counts().head(5))
            
    print("\nExploration complete.")

if __name__ == "__main__":
    explore_data()
