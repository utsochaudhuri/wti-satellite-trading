
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

file_name = "oklahoma_cloud_coverage_July2024.csv"
df = pd.read_csv(file_name)
df = df.iloc[108:]

def get_date_list(start_date, end_date):
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates

start = datetime(2018, 12, 11)
end = datetime.today()
dates = get_date_list(start, end)[:-1]

# Add missing dates
for i in dates:
    if i not in df["date"].values:
        new_row = pd.DataFrame([{"date": i, "city": "NA", "cloud_coverage_percent": np.nan}])
        df = pd.concat([df, new_row], ignore_index=True)

# Convert date column to datetime for proper sorting and operations
df["date"] = pd.to_datetime(df["date"])

# Sort by date to ensure proper order
df = df.sort_values("date").reset_index(drop=True)

# Download WTI data
ticker = "CL=F"
start_date = "2018-12-11"
end_date = datetime.today().strftime('%Y-%m-%d')
cl_data = yf.download(ticker, start=start_date, end=end_date, progress=False)

# FIX: Handle MultiIndex columns
if isinstance(cl_data.columns, pd.MultiIndex):
    # Flatten the MultiIndex columns
    cl_data.columns = cl_data.columns.get_level_values(0)

# Reset index to make Date a column instead of index
cl_data = cl_data.reset_index()
cl_data = cl_data[['Date','Close']].copy()
cl_data.rename(columns={'Close': 'wti_price'}, inplace=True)

# Convert date columns to datetime
df['date'] = pd.to_datetime(df['date'])
cl_data['Date'] = pd.to_datetime(cl_data['Date'])

# Perform the merge
df_joined = df.merge(cl_data, left_on='date', right_on='Date', how='left')

# Clean up - remove the duplicate Date column
df_joined = df_joined.drop('Date', axis=1)

# Save the joined data (not the original df)
df_joined.to_csv("output.csv", index=False)