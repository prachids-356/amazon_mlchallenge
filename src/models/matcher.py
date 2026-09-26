import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLS = [
    'name_edit_sim', 
    'name_jaccard', 
    'address_edit_sim', 
    'address_jaccard', 
    'country_match'
]

class EntityMatcher:
    def __init__(self, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        
    def fit(self, df_train: pd.DataFrame):
        """Train the classifier on engineered features."""
        if df_train.empty:
            return
            
        X = df_train[FEATURE_COLS].fillna(0)
        y = df_train['label']
        
        print(f"Training Matcher on {len(X)} pairs...")
        self.model.fit(X, y)
        self.is_fitted = True
        print("Training complete.")
        
    def predict_proba(self, df_test: pd.DataFrame) -> pd.Series:
        """Predict match probabilities."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        if df_test.empty:
            return pd.Series([])
            
        import numpy as np
        X = df_test[FEATURE_COLS].fillna(0)
        probs = self.model.predict_proba(X)
        
        if probs.shape[1] == 1:
            if self.model.classes_[0] == 1:
                return pd.Series(probs[:, 0], index=df_test.index)
            else:
                return pd.Series(np.zeros(len(df_test)), index=df_test.index)
        else:
            return pd.Series(probs[:, 1], index=df_test.index)
