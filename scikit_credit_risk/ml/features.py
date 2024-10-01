from typing import List
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.utils import resample

class DataCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, age_column:str=None, age_threshold:int=None, column_to_drop=None, emp_length_column:str=None, emp_length_threshold:int=None):
        self.age_column = age_column
        self.age_threshold = age_threshold
        self.column_to_drop = column_to_drop
        self.emp_length_column = emp_length_column
        self.emp_length_threshold = emp_length_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # Check for nulls before dropping them
        if X.isnull().values.any():
            X = X.dropna()

        # Filter by age threshold if the column and threshold are set
        if self.age_column and self.age_threshold is not None:
            X.loc[:, self.age_column] = X[self.age_column].astype(int)
            max_age = X[self.age_column].max()
            if max_age > self.age_threshold:
                X = X[X[self.age_column] <= self.age_threshold]

        # Filter by employment length
        if self.emp_length_column and self.emp_length_threshold is not None:
            X = X[X[self.emp_length_column] <= self.emp_length_threshold]

        # Drop the specified column
        if self.column_to_drop and self.column_to_drop in X.columns:
            X = X.drop(columns=[self.column_to_drop])

        # Reset the index of the DataFrame
        X = X.reset_index(drop=True)

        return X


class AgeGroupCategorizer(BaseEstimator, TransformerMixin):
    def __init__(self, input_col:str=None, output_col:str=None, bins=None, labels=None):
        self.input_col = input_col
        self.output_col = output_col
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        
        # Ensure bins and labels are provided
        if self.bins is None or self.labels is None:
            raise ValueError("Both bins and labels must be provided.")
        
        # Check that the number of labels is one less than the number of bins
        if len(self.labels) != len(self.bins) - 1:
            raise ValueError("The number of labels must be equal to len(bins) - 1.")

        # Use pandas cut to assign age groups
        X[self.output_col] = pd.cut(X[self.input_col], bins=self.bins, labels=self.labels, right=False, include_lowest=True)
        X[self.output_col] = X[self.output_col].astype(str)

        return X
class IncomeGroupCategorizer(BaseEstimator, TransformerMixin):
    def __init__(self, input_col:str=None, output_col:str=None):
        self.input_col = input_col
        self.output_col = output_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Check if the input is a valid pandas DataFrame or Series
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise ValueError("Input must be a pandas DataFrame or Series.")
        
        # If X is a scalar value (a single income value), wrap it in a DataFrame
        if isinstance(X, pd.Series) or np.isscalar(X):
            X = pd.DataFrame({self.input_col: [X]})
        
        # Create a new column for the income group categorization
        X[self.output_col] = np.select(
            [
                X[self.input_col].between(0, 25000),
                X[self.input_col].between(25001, 50000),
                X[self.input_col].between(50001, 75000),
                X[self.input_col].between(75001, 100000)
            ],
            [
                'low',
                'low_middle',
                'middle',
                'high_middle'
            ],
            default='high'
        )
        
        return X
    
class LoanAmountCategorizer(BaseEstimator, TransformerMixin):
    def __init__(self, input_col:str = None, output_col:str=None):
        self.input_col = input_col
        self.output_col = output_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Check if the input is a valid pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # Create a new column for the loan amount group categorization
        X[self.output_col] = np.select(
            [
                X[self.input_col].between(0, 5000),
                X[self.input_col].between(5001, 10000),
                X[self.input_col].between(10001, 15000)
            ],
            [
                'small',
                'medium',
                'high'
            ],
            default='very_high'
        )
        
        return X
    
class RatioFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # No parameters needed for this transformer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Check if the input is a valid pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # Ensure required columns exist
        required_columns = ['loan_amnt', 'person_income', 'person_emp_length', 'loan_int_rate']
        for col in required_columns:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' is not in the DataFrame.")

        # Create new ratio columns 
        X['loan_to_income_ratio'] = (X['loan_amnt'] / X['person_income'])
        X['loan_to_emp_length_ratio'] = (X['person_emp_length'] / X['loan_amnt'])
        X['int_rate_to_loan_amt_ratio'] = (X['loan_int_rate'] / X['loan_amnt'])

        return X

class Upsampling:
    def __init__(self, target_column: str):
        """
        Initialize the upsampling class with the target column.

        Parameters:
        target_column (str): The name of the target variable in the DataFrame.
        """
        self.target_column = target_column

    def fit(self, dataframe: pd.DataFrame):
        """
        Fit the upsampling model to the provided DataFrame.

        Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.

        Returns:
        None
        """
        self.dataframe = dataframe
        self.majority_class = self.dataframe[self.target_column].value_counts().idxmax()
        self.minority_class = self.dataframe[self.target_column].value_counts().idxmin()

    def upsample(self):
        """
        Upsample the minority class in the DataFrame.

        Returns:
        pd.DataFrame: A new DataFrame with the upsampled minority class.
        """
        # Separate majority and minority classes
        majority = self.dataframe[self.dataframe[self.target_column] == self.majority_class]
        minority = self.dataframe[self.dataframe[self.target_column] == self.minority_class]

        # Upsample the minority class
        minority_upsampled = resample(minority,
                                       replace=True,  # Sample with replacement
                                       n_samples=len(majority),  # To match the majority class
                                       random_state=42)  # Reproducible results

        # Combine majority class with upsampled minority class
        upsampled_df = pd.concat([majority, minority_upsampled])

        # Shuffle the dataset
        upsampled_df = upsampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

        return upsampled_df

    
