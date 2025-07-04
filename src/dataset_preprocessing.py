"""
This script loads multiple financial time series (ethanol, corn, FX, PPI, Brent),
cleans and aligns them to daily frequency, handles missing data, scales the features,
and outputs both raw and normalized versions for modeling.

Outputs:
- calendar.parquet: Cleaned daily time series with all features
- calendar_scaled.parquet: Scaled version for LSTM input
- feature_scaler.pkl: Saved scaler for inverse transformation

Author: Felice Faruolo (VU Amsterdam)
Repository: https://github.com/felixfaruix/ethanol-hierarchical-multi-band-LSTM.git
"""
from typing  import List, Dict, Tuple
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pandas as pd, numpy as np, textwrap
import pickle as pkl

file_root = Path(__file__).resolve().parent
project_root = file_root.parent
raw_data_path: Path = project_root / "raw_data"
processed_data: Path = project_root / "processed_data"
processed_data.mkdir(exist_ok=True)
start_date: str = "2010-01-01"

# Dictionary to hold file configurations
# Each key corresponds to a feature, with its date column, price column, filename, and date format
# The date format is used to parse the date column correctly and interpolate if necessary
# The "interpolate" key indicates whether to linearly interpolate missing values in the time series
files: Dict[str, Dict[str, str]] = {
    "ethanol": {"date_col": "Time", "price_col": "Last", "fname": "d2_daily_historical.csv", "date_format": "%Y-%m-%d"},
    "ethanol_volume": {"date_col": "Time", "price_col": "Volume", "fname": "d2_daily_historical.csv", "date_format": "%Y-%m-%d"},
    "corn": {"date_col": "Time", "price_col": "Last", "fname": "zc_daily_historical.csv", "date_format": "%Y-%m-%d"},
    "fx": {"date_col": "Date", "price_col": "Price", "fname": "usd_brl_historical.csv", "date_format": "%m/%d/%Y"},
    "brent": {"date_col": "Time", "price_col": "Last", "fname": "wti_daily_historical.csv", "date_format": "%Y-%m-%d"},
    "ppi": {"date_col": "date", "price_col": "ppi", "fname": "ethanol_ppi_weekly.csv", "date_format": "%Y-%m-%d", "interpolate": "True"},
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

def fill_and_mask(calendar: pd.DataFrame, daily_columns: List[str], mask_columns: List[str]) -> pd.DataFrame:
    """
    Forward-fill daily series (ethanol, corn, fx, brent) across non-trading days.
    """
    # Add a 'market_closed' column: set to 1 if all mask_columns are NaN (i.e., all relevant markets are closed on that day), otherwise 0
    calendar["market_closed"] = calendar[mask_columns].isna().all(axis=1).astype(int)
    calendar[daily_columns] = calendar[daily_columns].ffill()
    calendar = calendar.dropna(subset=daily_columns)
    return calendar

def scale_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Scales selected feature columns to [0, 1] using MinMaxScaler.
    This is important for models that are sensitive to feature scales
    """
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df[feature_cols])
    # Create a new DataFrame with scaled values and appropriate column names
    # The scaled columns will have "_scaled" suffix to differentiate them from original values
    df_scaled = pd.DataFrame(scaled_values, columns=[f"{c}_scaled" for c in feature_cols], index=df.index)
    df_scaled.insert(0, "market_closed", df["market_closed"])
    df_scaled.insert(0, "date", df["date"])
    return df_scaled, scaler

def save_outputs(df_raw: pd.DataFrame, df_scaled: pd.DataFrame, scaler: MinMaxScaler) -> None:
    """
    Saves the raw and scaled DataFrames to Parquet, and the scaler as a pickle file.

    Takes as input: 
        df_raw (pd.DataFrame): Cleaned unscaled data.
        df_scaled (pd.DataFrame): Normalized data.
        scaler (MinMaxScaler): Trained scaler for future inverse transformations.
    """
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
        ── calendar_scaled ──────────────────────────────────────────
        shape        : {df_scaled.shape}   (rows, cols)
        date range   : {df_scaled['date'].min().date()} → {df_scaled['date'].max().date()}
        market_closed: {df_scaled['market_closed'].value_counts().to_dict()}
        summary (first 5 scaled cols)
        {df_scaled.filter(like='_scaled').iloc[:, :5].describe().loc[['min','max']]}
        """)
    )