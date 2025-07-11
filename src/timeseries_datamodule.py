import pandas as pd
from typing import Dict, List, Tuple, Union, Literal
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

target_variable = "ethanol_scaled"
batch_size = 64
val_test_ratio = 0.2
lookback_days = 365
daily_window = 14
weekly_horizon = 7
monthly_horizon = 30
test_start = pd.Timestamp("2023-01-01")
valid_start  = pd.Timestamp("2022-01-01")

def load_parquet(obj: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Return a DataFrame given a path or an already-loaded DataFrame."""
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    obj = Path(obj)
    if not obj.exists():
        raise FileNotFoundError(obj)
    return pd.read_parquet(obj)

def merge_calendars(calendar_scaled: Union[str, Path, pd.DataFrame], cyclical_calendar: Union[str, Path, pd.DataFrame] = "cyclical_calendar.parquet",*, 
                    date_column: str = "date", how: Literal["left", "right", "outer", "inner", "cross"] = "left") -> pd.DataFrame:
    """
    This function merges a scaled calendar DataFrame with a cyclical calendar DataFrame on the specified date column.
    Args:
        calendar_scaled (Union[str, Path, pd.DataFrame]): Path to the scaled calendar DataFrame or a DataFrame object.
        cyclical_calendar (Union[str, Path, pd.DataFrame]): Path to the cyclical calendar DataFrame or a DataFrame object.
        date_col (str): The name of the date column to merge on. Default is "date".
        how (str): The type of merge to perform. Default is "left".
    Returns
    -------
    pd.DataFrame
        A merged DataFrame with one row per day containing every feature.
    """
    calendar1  = load_parquet(calendar_scaled).sort_values(date_column).reset_index(drop=True)
    calendar2 = load_parquet(cyclical_calendar)

    if date_column not in calendar1.columns or date_column not in calendar2.columns:
        raise KeyError(f"'{date_column}' must exist in both frames")

    merged_calendars = calendar1.merge(calendar2, on=date_column, how=how, validate="1:1")
    return merged_calendars

class RollingOrigin(Dataset):
    """A PyTorch Dataset that creates rolling slices of a DataFrame for time series forecasting.
    This dataset is designed to handle time series data with a specified lookback period and horizon for forecasting.
    It asumes a sliding window approach where each sample consists of:
    - A memory of the past `lookback_days` days of features.
    - A daily window of the last `daily_window` days of features.
    - A target value for the day, or multiple target values for the next days (week, month).
    It's important to note that the dataset is designed to work with a DataFrame that has been preprocessed 
    to include the necessary features and target variable.
    """
    def __init__(self, df: pd.DataFrame, features: List[str]):
        self.df = df.reset_index(drop=True)
        self.X = self.df[features].to_numpy("float32")
        self.y = self.df[target_variable].to_numpy("float32")
        self.first_origin = lookback_days 
        self.last_origin = len(df) - (daily_window + monthly_horizon)

    def __len__(self):
        """Return the number of samples in the dataset."""
        # The number of samples is the range from first to last origin, inclusive
        return self.last_origin - self.first_origin

    def __getitem__(self, i: int):
        """Get a single item from the dataset.
        Args:
            i (int): The index of the item to retrieve.
        It returns a tuple containing: (memory, daily, target_day, target_week, target_month)
        - memory: A tensor of shape [365, F] representing the past year of features.
        - daily: A tensor of shape [14, F] representing the features for the last 14 days.
        - target_day: A scalar tensor representing the target value for the day after the daily window.
        - target_week: A tensor of shape [7] representing the target values for the next 7 days.
        - target_month: A tensor of shape [30] representing the target values for the next 30 days.
        """
        # Calculate the origin index for the rolling slice
        origin = self.first_origin + i
        # Extract the lookback memory and daily window features
        # lookback_memory is the past 365 days of features, x_daily is the last 14 days of features
        # The targets start after the daily window, so we calculate the target index accordingly
        lookback_memory  = self.X[origin - lookback_days : origin]
        daily_features = self.X[origin : origin + daily_window]

        # targets start after daily window
        target = origin + daily_window
        daily_target = self.y[target] # scalar target for the day after the daily window
        # Targets for week and month are on top of the daily window and are sliced accordingly
        weekly_target = self.y[target : target + weekly_horizon] # weekly target is a 7-vector
        # monthly target is a 30-vector, it is the target for the next month after the daily window
        monthly_target = self.y[target : target + monthly_horizon] # 30-vector

        return (torch.from_numpy(lookback_memory), # 365-day lookback memory
            torch.from_numpy(daily_features),  # 14-day daily window
            torch.tensor(daily_target, dtype=torch.float32),
            torch.from_numpy(weekly_target),
            torch.from_numpy(monthly_target))

def get_features_columns(df: pd.DataFrame) -> List[str]:
    """Get feature columns from the DataFrame excluding 'date'
    """
    return [x for x in df.columns if x not in ("date")] #It supposed the dataset has a 'date' column.

def build_loaders(df: pd.DataFrame, batch_size: int = batch_size) -> Tuple[DataLoader, ...]:
    """Build DataLoaders for training, validation, and testing from a DataFrame.
    The DataFrame should contain a 'date' column and the target variable.
    The function splits the DataFrame into training, validation, and test sets based on the specified start dates.
    """
    features_cols = get_features_columns(df) # Get feature columns excluding 'date'
    test_index = df.index[df["date"] >= test_start][0]
    validation_index  = df.index[df["date"] >= valid_start][0]

    train_ds = RollingOrigin(df.iloc[:validation_index ], features_cols)
    val_ds   = RollingOrigin(df.iloc[validation_index:test_index], features_cols)
    test_ds  = RollingOrigin(df.iloc[test_index:], features_cols)

    # Create DataLoaders for each dataset
    # The DataLoader will shuffle the training data, but not the validation and test data
    # drop_last=True ensures that the last incomplete batch is dropped
    # This is important for training, as it ensures that all batches have the same size
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(val_ds, batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, drop_last=True)
    return train_loader, valid_loader, test_loader
