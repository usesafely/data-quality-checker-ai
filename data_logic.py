import pandas as pd
import io

class DataManager:
    def __init__(self, file):
        # Load the CSV file into a pandas DataFrame
        self.df = pd.read_csv(file)
        self.original_shape = self.df.shape

    def analyze_issues(self):
        # 1. Detect Missing Values
        missing = self.df.isnull().sum()
        missing = missing[missing > 0] # Only show columns with errors
        
        # 2. Detect Duplicates
        duplicates = self.df.duplicated().sum()
        
        # 3. Detect Data Types
        types = self.df.dtypes.astype(str)
        
        return {
            "missing": missing.to_dict(),
            "duplicates": int(duplicates),
            "types": types.to_dict(),
            "rows": self.df.shape[0],
            "cols": self.df.shape[1]
        }

    def clean_data(self):
        # Logic 1: Remove completely empty rows
        self.df.dropna(how='all', inplace=True)
        
        # Logic 2: Remove duplicates
        self.df.drop_duplicates(inplace=True)
        
        # Logic 3: Fill missing numbers with 0 (or Mean)
        # We select only number columns
        num_cols = self.df.select_dtypes(include=['number']).columns
        self.df[num_cols] = self.df[num_cols].fillna(0)
        
        # Logic 4: Fill missing text with "Unknown"
        text_cols = self.df.select_dtypes(include=['object']).columns
        self.df[text_cols] = self.df[text_cols].fillna("Unknown")
        
        return self.df

    def get_summary_stats(self):
        # Generates math stats (Mean, Max, Min) for number columns
        return self.df.describe().to_string()