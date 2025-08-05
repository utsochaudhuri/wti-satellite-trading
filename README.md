# wti-satellite-trading

## Overview
This program uses satellite imaging to extract cloud coverage data over Cushing, Oklahoma, leveraging it to estimate and trade West Texas Intermediate (WTI) crude oil futures.

Cushing is critical to the oil market as it serves as the official settlement point for WTI crude futures and acts as the central hub for US crude oil storage. Analysts often rely on satellite imagery of floating-roof tanks to monitor oil inventories—however, cloud cover can obscure these observations, introducing noise and uncertainty into supply estimates.

By integrating cloud coverage information with basic uncertainty principles, the model attempts to predict movements in WTI futures and generate trading signals.

How to Run
To execute the pipeline:
1. Run cloud_satellite_download.py to download cloud coverage data.
2. Run cloud_coverage_cleaning.py to clean the data and fetch WTI pricing via Yahoo Finance.
3. Finally, execute ML_price_sat_insight.py to generate model predictions, simulate trades, and evaluate portfolio performance.

Required Libraries:
Install the required packages using:

pip install pandas numpy matplotlib scikit-learn yfinance sentinelhub

## Methodology
Due to limited satellite imagery directly over Cushing, cloud coverage data from Tulsa and Oklahoma City (each ~75 km from Cushing) were used as proxies.

WTI futures prices were sourced from Yahoo Finance using the yfinance API.

A Random Forest Regressor and Classifier were trained using features such as:

- Cloud coverage percentage
- Short and long term moving averages
- Price volatility and momentum
- Basic lagged variables

The model outputs long or short signals, which were then executed in a simulated portfolio.

## Results
* Benchmark return (buy-and-hold strategy): 17%
* Model return (cloud-driven trading): 11%

While the model underperforms the benchmark, it is noteworthy that:
* It only uses cloud coverage and historical WTI prices, ignoring macro, geopolitical, and fundamental factors.
* In highly volatile markets like oil, additional predictive features (e.g., inventory reports, macro data, supply disruptions) would significantly improve accuracy and profitability.
