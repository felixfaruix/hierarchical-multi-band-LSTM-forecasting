# This script processes raw financial data into a structured format for time series analysis.
# It reads various CSV files containing daily, weekly, and monthly financial data, processes them to create a unified dataset for modeling and forecasting.
# It includes features such as ethanol prices, corn prices, foreign exchange rates, and producer price index (PPI) data.
# The processed data is saved in Parquet format for efficient storage and retrieval.

from typing  import List, Dict
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd, numpy as np, textwrap
import pickle as pkl

file_root = Path(__file__).resolve().parent
project_root = file_root.parent
raw_data_path: Path = project_root / "raw_data"
processed_data: Path = project_root / "processed_data"
processed_data.mkdir(exist_ok=True)
start_date: str = "2010-01-01"

def raw_to_processed(key, *, date_col, price_col, fname, start_date=start_date, date_format=None, **kwargs) -> pd.DataFrame:
    """Read raw CSV and return a DataFrame with columns: date, value
    """
    path = (raw_data_path / fname)
    dataframe: pd.DataFrame = pd.read_csv(path) # Read the CSV file
    dataframe[date_col] = pd.to_datetime(dataframe[date_col], format=date_format, errors="coerce") # Convert date column to datetime
    dataframe = dataframe[dataframe[date_col] >= pd.to_datetime(start_date)] # Filter by start date
    dataframe = dataframe[[date_col, price_col]].rename(columns={date_col: "date", price_col: key}) # Rename columns to 'date' and 'value'
    return dataframe.sort_values("date") # Sort by date 

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
# Create a DataFrame for the calendar with daily dates starting from the specified start date
calendar = pd.DataFrame({"date": pd.date_range(start_date, pd.Timestamp.today(), freq="D")})

# Pass each file configuration to the raw_to_processed function and merge the results into the calendar DataFrame
# Loop through each file configuration and linearly interpolate if specified
for key, cfg in files.items():
    df = raw_to_processed(key, **cfg)
    if cfg.get("interpolate", False):
        df = df.set_index("date").interpolate(method="linear").reset_index()
    calendar = calendar.merge(df, on="date", how="left")

# Forward-fill daily series (ethanol, corn, fx, brent) across non-trading days.
daily_columns: List[str] = ["ethanol", "ethanol_volume", "corn", "fx", "brent", "ppi"]
# Adding a market closed column to indicate non-trading days
price_columns_for_mask: List[str] = ["ethanol", "corn", "brent"]
calendar["market_closed"] = calendar[price_columns_for_mask].isna().all(axis=1).astype(int)
# Forward fill the daily columns
calendar[daily_columns] = calendar[daily_columns].ffill()
# Drop rows where any daily proxy is still missing (i.e., pre-2010 or gaps)
calendar = calendar.dropna(subset=daily_columns)

# Scale daily features to [0, 1] range
# This is important for models that are sensitive to feature scales
scaler = MinMaxScaler()
scaled_values: np.ndarray = scaler.fit_transform(calendar[daily_columns])
# Create a new DataFrame with scaled values and appropriate column names
# The scaled columns will have "_scaled" suffix to differentiate them from original values
calendar_scaled = pd.DataFrame(scaled_values, columns=[c + "_scaled" for c in daily_columns], index=calendar.index)
calendar_scaled.insert(0, "market_closed", calendar["market_closed"])
calendar_scaled.insert(0, "date", calendar["date"])

# Save the scaler so the model can inverse-transform predictions later
with open(processed_data / "feature_scaler.pkl", "wb") as f:
    pkl.dump(scaler, f)
print(f"Saved feature scaler to: {processed_data / 'feature_scaler.pkl'}")

# Saving the processed calendar data and scaled features to Parquet files
# Parquet is a columnar storage file format that is efficient for large datasets
calendar.to_parquet(processed_data / "calendar.parquet", index=False)
calendar_scaled.to_parquet(processed_data / "calendar_scaled.parquet", index=False)

# Index the calendar DataFrame by date for easier time series operations
calendar_index = calendar.set_index("date")

cs = pd.read_parquet(processed_data / "calendar_scaled.parquet")

print(
    textwrap.dedent(f"""
    ── calendar_scaled ──────────────────────────────────────────
    shape        : {cs.shape}   (rows, cols)
    date range   : {cs['date'].min().date()} → {cs['date'].max().date()}
    market_closed: {cs['market_closed'].value_counts().to_dict()}
    summary (first 5 scaled cols)
    {cs.filter(like='_scaled').iloc[:, :5].describe().loc[['min','max']]}
    """)
)