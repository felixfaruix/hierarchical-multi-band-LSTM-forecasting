""" This file contains functions to build a calendar DataFrame with engineered temporal features.
It includes features such as:
- Day of week (sin/cos)
- Month of year (sin/cos)
- Week of month (1-5)
- End-of-month flag (is_eom)
Everything is made cyclical to avoid discontinuities in the data. Especially because we are using
recurrent neural networks, which are sensitive to discontinuities in the data. 
We want to preserve the cyclical nature of time because of the temporal dependencies in the data.
"""

import pandas as pd
import numpy as np

# Function to calculate the week of the month
# This function takes a datetime object and returns the week of the month (1-5).
def weeks_of_month(dt: pd.Timestamp) -> int:
    """
    Calculate the week of the month for a given datetime.
    Returns integer 1-5.
    """
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()  # align with week start
    return int(np.ceil(adjusted_dom / 7.0)) # Adjusted to 1-5 weeks in the month

# Function to build a calendar DataFrame with engineered temporal features
# This function creates a DataFrame with daily dates and adds features such as:
def build_calendar_df(start_date: str ="2010-01-01", end_date: str ="2030-12-31") -> pd.DataFrame:
    """
    Create a calendar DataFrame with engineered temporal features.
    This function generates a DataFrame with daily dates and adds features such as:
    - Day of week (sin/cos)
    - Month of year (sin/cos)
    - Week of month (1-5)
    - End-of-month flag (is_eom)
    """
    # Creating a dataframe with a continuous range of daily dates
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    df = pd.DataFrame({"date": date_range})
    
    # Ensuring 'date' is a datetime column
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("The 'date' column must be of datetime type.")
    
    # Day-of-week (0–6, Monday = 0) 
    days = df["date"].dt.dayofweek  # Monday = 0
    df["dow_sin"] = np.sin(2 * np.pi * days / 7)
    df["dow_cos"] = np.cos(2 * np.pi * days / 7)

    # Month-of-year (1–12)
    months = df["date"].dt.month
    df["mon_sin"] = np.sin(2 * np.pi * months / 12)
    df["mon_cos"] = np.cos(2 * np.pi * months / 12)

    # Week of month (1–5)
    df["week_of_month"] = df["date"].apply(weeks_of_month)


    # End-of-month flag
    # We create a boolean column for month end
    # This will be True if the date is the last day of the month. We need the helper because pandas does not have a built-in function for this.
    # We will use this to create a flag for end-of-month (EOM) events
    eom_flags = pd.concat([df["is_month_end"], df["is_month_end"].shift(-1), df["is_month_end"].shift(-2)], axis=1).fillna(False)
    df["is_eom"] = eom_flags.any(axis=1).astype(int)
    return df.drop(columns=["is_month_end"])

calendar = build_calendar_df()
calendar.to_parquet("calendar.parquet")
