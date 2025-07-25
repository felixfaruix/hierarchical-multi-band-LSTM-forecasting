"""
This script loads multiple financial time series (ethanol, corn, FX, PPI, Brent),
cleans and aligns them to daily frequency, handles missing data, **adds lag / return
features**, scales the features, and outputs both raw and normalized versions for modeling.

Outputs:
- calendar.parquet: Cleaned daily time series with all features
- calendar_scaled.parquet: Scaled version for LSTM input
- feature_scaler.pkl: Saved scaler for inverse transformation
"""

from typing  import List, Dict, Tuple, Any
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pandas as pd, numpy as np, textwrap
import pickle as pkl
from dateutil.easter import easter

verbose = False
file_root = Path(__file__).resolve().parent
project_root = file_root.parent
raw_data_path: Path = project_root / "raw_data"
processed_data: Path = project_root / "processed_data"
processed_data.mkdir(exist_ok=True)
start_date: str = "2010-01-01"
lags_list= ["ethanol", "corn", "brent", "fx"]
lags= (7, 30)
return_bool= True

# Dictionary to hold file configurations
# Each key corresponds to a feature, with its date column, price column, filename, and date format
# The date format is used to parse the date column correctly and interpolate if necessary
# The "interpolate" key indicates whether to linearly interpolate missing values in the time series
files: Dict[str, Dict[str, Any]] = {
    "ethanol": {"date_col": "Time", "price_col": "Last", "fname": "d2_daily_historical.csv", "date_format": "%Y-%m-%d"},
    "ethanol_volume": {"date_col": "Time", "price_col": "Volume", "fname": "d2_daily_historical.csv", "date_format": "%Y-%m-%d"},
    "corn": {"date_col": "Time", "price_col": "Last", "fname": "zc_daily_historical.csv", "date_format": "%Y-%m-%d"},
    "fx": {"date_col": "Date", "price_col": "Price", "fname": "usd_brl_historical.csv", "date_format": "%m/%d/%Y"},
    "brent": {"date_col": "Time", "price_col": "Last", "fname": "wti_daily_historical.csv", "date_format": "%Y-%m-%d"},
    "ppi": {"date_col": "date", "price_col": "ppi", "fname": "ethanol_ppi_weekly.csv", "date_format": "%Y-%m-%d", "interpolate": True},
}
daily_columns: List[str] = ["ethanol", "ethanol_volume", "corn", "fx", "brent", "ppi"]
price_columns_for_mask: List[str] = ["ethanol", "corn", "brent"]

def raw_to_processed(key, *, date_col, price_col, fname, start_date=start_date, date_format=None, **kwargs) -> pd.DataFrame:
    """
    Reads a raw CSV file and returns a cleaned DataFrame with standardized column names.
    Returns:
        pd.DataFrame: A DataFrame with columns ['date', key] sorted by date.
    """
    path = (raw_data_path / fname)
    dataframe: pd.DataFrame = pd.read_csv(path) # Read the CSV file
    dataframe[date_col] = pd.to_datetime(dataframe[date_col], format=date_format, errors="coerce") # Convert date column to datetime
    dataframe = dataframe[dataframe[date_col] >= pd.to_datetime(start_date)] # Filter by start date
    dataframe = dataframe[[date_col, price_col]].rename(columns={date_col: "date", price_col: key}) # Rename columns to 'date' and 'value'
    return dataframe.sort_values("date") # Sort by date 

def build_calendar(start_date: str) -> pd.DataFrame:
    """
    Creates a DataFrame with a continuous range of daily dates from start_date to today.
    """
    return pd.DataFrame({"date": pd.date_range(start_date, pd.Timestamp.today(), freq="D")})

def merge_all_data(calendar: pd.DataFrame, files: Dict[str, Dict]) -> pd.DataFrame:
    """
    Merges all configured raw data files into a single calendar DataFrame.
    """
    for key, cfg in files.items():
        df = raw_to_processed(key, **cfg)
        if cfg.get("interpolate", False):
            df = df.set_index("date").interpolate(method="linear").reset_index()
        calendar = calendar.merge(df, on="date", how="left")
    return calendar

def event_window_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds binary event window flags to a DataFrame with a 'date' column.
    Flags include:
        - Christmas to New Year: Dec 24 – Jan 2
        - Easter window: Good Friday to Easter Monday
        - Driving season: May 15 – Jun 14
        - Corn harvest: Sep 15 – Oct 14
    Flags are added only for the dates in the DataFrame.
    """
    idx = df['date']
    flags = {}

    # Christmas to New Year (cross-year window)
    flags['event_xmas_newyear'] = (((idx.dt.month == 12) & (idx.dt.day >= 24)) | ((idx.dt.month == 1) & (idx.dt.day <= 2)))
    # Driving season: May 15 – Jun 14
    flags['event_driving_season'] = (((idx.dt.month == 5) & (idx.dt.day >= 15)) | ((idx.dt.month == 6) & (idx.dt.day <= 14)))
    # Corn harvest: Sep 15 – Oct 14
    flags['event_corn_harvest'] = (((idx.dt.month == 9) & (idx.dt.day >= 15)) | ((idx.dt.month == 10) & (idx.dt.day <= 14)))

    # Easter window: Good Friday to Easter Monday
    easter_window = pd.Series(False, index=idx)
    for year in range(idx.min().year, idx.max().year + 1):
        e = easter(year)
        window = pd.date_range(e - pd.Timedelta(days=2), e + pd.Timedelta(days=1))
        easter_window |= idx.isin(window)
    flags['event_easter'] = easter_window.values
    # Adding flags to the DataFrame
    for col, val in flags.items():
        df[col] = val.astype(int)
    return df

def fill_and_mask(calendar: pd.DataFrame, daily_columns: List[str], mask_columns: List[str]) -> pd.DataFrame:
    """
    Forward-fill daily series (ethanol, corn, fx, brent) across non-trading days.
    """
    # Add a 'market_closed' column: set to 1 if all mask_columns are NaN (i.e., all relevant markets are closed on that day), otherwise 0
    calendar["market_closed"] = calendar[mask_columns].isna().all(axis=1).astype(int)
    calendar[daily_columns] = calendar[daily_columns].ffill()
    calendar = calendar.dropna(subset=daily_columns)
    return calendar

def lag_returns(df: pd.DataFrame, columns: List[str], lags: Tuple[int, ...], return_bool: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds lagged level features for the specified lags and 1-day log returns
    for every column in `cols`. Returns the updated DataFrame and a list of
    new column names added.
    """
    new_columns: List[str] = []

    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        # lagged levels
        for l in lags:
            lag_columns = f"{col}_lag_{l}"
            df[lag_columns] = series.shift(l)
            new_columns.append(lag_columns)

        # 1-day log return
        if return_bool:
            ret_col = f"{col}_log_ret_1"
            safe_series = series.where(series > 0, np.nan)
            df[ret_col] = np.log(safe_series).diff()  # type: ignore[assignment]
            new_columns.append(ret_col)
    return df, new_columns

def rolling_stats_and_spread(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds 30-day rolling mean & std, a 90-day z-score for ethanol,
    and two cross-asset spreads (corn/ethanol, brent/ethanol).
    Returns the updated DataFrame and the list of new columns.
    """
    new_columns: List[str] = []

    # Rolling mean & std (28-day) for ethanol
    mean30 = "ethanol_roll_mean_28"
    std30  = "ethanol_roll_std_28"
    df[mean30] = df["ethanol"].rolling(28).mean()
    df[std30]  = df["ethanol"].rolling(28).std()
    new_columns.extend([mean30, std30])

    # 90-day z-score for ethanol
    z90 = "ethanol_z_90"
    roll_mean_90 = df["ethanol"].rolling(90).mean()
    roll_std_90  = df["ethanol"].rolling(90).std()
    df[z90] = (df["ethanol"] - roll_mean_90) / roll_std_90
    new_columns.append(z90)

    # Cross-asset spreads
    spread1 = "corn_ethanol_spread"
    spread2 = "brent_ethanol_spread"
    df[spread1] = df["corn"] / df["ethanol"]
    df[spread2] = df["brent"] / df["ethanol"]
    new_columns.extend([spread1, spread2])

    print(f"Added {len(new_columns)} rolling and spread columns to the DataFrame: {new_columns}")
    return df, new_columns

def save_outputs(df_raw: pd.DataFrame, df_scaled: pd.DataFrame, scaler: MinMaxScaler) -> None:
    """
    Saves the raw and scaled DataFrames to Parquet, and the scaler as a pickle file.

    Takes as input: 
        df_raw (pd.DataFrame): Cleaned unscaled data.
        df_scaled (pd.DataFrame): Normalized data.
        scaler (MinMaxScaler): Trained scaler for future inverse transformations.
    """
    # Saving the scaler to a pickle file for later use
    with open(processed_data / "feature_scaler.pkl", "wb") as f:
        pkl.dump(scaler, f)
    print(f"Saved feature scaler to: {processed_data / 'feature_scaler.pkl'}")

    df_raw.to_parquet(processed_data / "calendar.parquet", index=False)
    df_scaled.to_parquet(processed_data / "calendar_scaled.parquet", index=False)

def summarize(df_scaled: pd.DataFrame) -> None:
    """
    Prints a summary of the scaled dataset including shape, date range, and value stats.
    """
    print(
        textwrap.dedent(f"""
        **Calendar Scaled**
        Shape: {df_scaled.shape}   (rows, cols)
        Date range: {df_scaled['date'].min().date()} → {df_scaled['date'].max().date()}
        Market Closed: {df_scaled['market_closed'].value_counts().to_dict()}
        Summary (first 5 scaled cols)
        {df_scaled.filter(like='_scaled').iloc[:, :5].describe().loc[['min','max']]}
        """))

if __name__ == "__main__":
    """
    Preprocessing pipeline with feature engineering and leakage-free scaling
    """
    # 1. Load and merge raw data files into a calendar DataFrame
    calendar = build_calendar(start_date)
    # 2. Merge all data files into the calendar DataFrame
    merged = merge_all_data(calendar, files)
    # 3. Fill missing values and mask non-trading days
    #    - Forward-fill daily series (ethanol, corn, fx, brent)
    merged = fill_and_mask(merged, daily_columns, price_columns_for_mask)
    print(merged[daily_columns].isna().sum())
    # 4. Add event window flags
    #    - Christmas to New Year, Easter, Driving season, Corn harvest
    merged = event_window_flag(merged)
    # 5. Add lagged features and returns
    #    - Lagged levels and 1-day log returns for ethanol, corn, brent, fx
    merged, new_feature_cols = lag_returns(merged, columns=lags_list, lags=lags, return_bool=return_bool)
    # 6. Add rolling statistics and spreads
    #    - 28-day rolling mean & std, 90-day z-score for ethanol
    merged, new_feature_cols_rolling = rolling_stats_and_spread(merged)
    # 7. Scaling the features
    # We extend the daily_columns with the new feature columns and rolling stats
    daily_columns_extended = (daily_columns + new_feature_cols + new_feature_cols_rolling)
    merged.dropna(subset=daily_columns_extended, inplace=True)
    bad = merged[daily_columns_extended].isna().sum()
    print(bad[bad > 0])
    # 8. Scale the features
    # We use MinMaxScaler to scale the features to [0, 1]
    # We fit the scaler only on the training slice of the data
    # Fit scaler only on training slice (before validation start date)
    train_mask = merged["date"] < pd.to_datetime("2022-01-01")  # Use actual validation start date
    # Fit scaler only on training slice
    scaler = MinMaxScaler().fit(merged.loc[train_mask, daily_columns_extended])
    # Transform the entire DataFrame with that single scaler
    scaled_values = scaler.transform(merged[daily_columns_extended])
    merged_scaled = pd.DataFrame(scaled_values, columns=[f"{c}_scaled" for c in daily_columns_extended], index=merged.index)
    # Concatenate meta-columns
    meta_cols = ["date", "market_closed"] + [col for col in merged.columns if col.startswith("event_")]
    merged_scaled = pd.concat([merged[meta_cols], merged_scaled], axis=1)
    # 8. Save the outputs
    save_outputs(merged, merged_scaled, scaler)
    if verbose:
        summarize(merged_scaled)
        
