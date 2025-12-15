import pandas as pd
import os

local_path = os.path.dirname(__file__)

test_path = os.path.join(local_path, 'test.csv')
train_path = os.path.join(local_path, 'train.csv')

def load_data() -> pd.DataFrame:
    """
    Load the training and testing data from CSV files 
    and return them as pandas DataFrames.
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

if __name__ == "__main__":
    train_data, test_data = load_data()
    print("Training Data:")
    print(train_data.head())
    print("\nTesting Data:")
    print(test_data.head())