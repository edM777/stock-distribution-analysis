import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd
from sqlalchemy import create_engine

# Establish a connection to your PostgreSQL database
# Note: Replace with your actual database credentials if they differ.
engine = create_engine('postgresql://postgres:<pwd>@localhost:<port>/<market-data-analysis-db>')

# Query to retrieve all tests results data
query_results = "SELECT * FROM tests_results;"
tests_results_df = pd.read_sql(query_results, engine)

# Query to retrieve all stock data
query_data = "SELECT * FROM stock_data;"
stock_data_df = pd.read_sql(query_data, engine)


# Function to calculate prices based on the data type given
def get_calculated_prices(data, data_type):
    data = data.copy()  # Prevents SettingWithCopyWarning
    if data_type == 'eod_avg':
        data['calculated_values'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    elif data_type == 'eod_typical':
        data['calculated_values'] = (data['high'] + data['low'] + data['close']) / 3
    elif data_type == 'eod_close':
        data['calculated_values'] = data['close']
    elif data_type == 'eod_high_low':
        data['calculated_values'] = (data['high'] + data['low']) / 2
    return data


# List of run IDs to process
# runId_list = [2304, 2240]
# runId_list = [13442, 23175, 3335, 7, 785, 7929, 1739, 399, 1507, 2287]
runId_list = [13442, 23175, 3335, 7, 785, 7929, 1739, 399, 1507, 2287, 3204, 2240]

for runId in runId_list:
    my_rows = tests_results_df[tests_results_df['runid'] == runId]

    for index, row in my_rows.iterrows():
        # Filter stock data for the specific ISIN
        stock_data_subset = stock_data_df[stock_data_df['isin'] == row['isin']].copy()

        # Convert 'bar_date' to datetime, sort, and reset index
        stock_data_subset['bar_date'] = pd.to_datetime(stock_data_subset['bar_date'], utc=True)
        stock_data_subset = stock_data_subset.sort_values('bar_date').reset_index(drop=True)

        # Calculate price based on the specified data type
        stock_data_subset = get_calculated_prices(stock_data_subset, row['data_type'])

        # Apply CAGR adjustment if specified
        if row['cagr']:
            initial_value = stock_data_subset['calculated_values'].iloc[0]
            final_value = stock_data_subset['calculated_values'].iloc[-1]
            days = (stock_data_subset['bar_date'].max() - stock_data_subset['bar_date'].min()).days
            if days > 0 and initial_value > 0:
                cagr_factor = ((final_value / initial_value) ** (365 / days)) - 1
                daily_price_factor = cagr_factor / 365
                day_deltas = (stock_data_subset['bar_date'] - stock_data_subset['bar_date'].min()).dt.days
                stock_data_subset['calculated_values'] /= (1 + daily_price_factor) ** day_deltas

        # Calculate log returns if specified
        if row['log_returns']:
            stock_data_subset['calculated_values'] = np.log(stock_data_subset['calculated_values'] /
                                                            stock_data_subset['calculated_values'].shift(1))
            # Drop the first row (NaN) and reset index
            stock_data_subset = stock_data_subset.iloc[1:].reset_index(drop=True)

        # Filter data by the specified time window
        stock_data_subset = stock_data_subset[
            (stock_data_subset['bar_date'] >= row['start_time']) &
            (stock_data_subset['bar_date'] <= row['end_time'])
            ]

        # Final data series for fitting
        final_data = stock_data_subset['calculated_values'].dropna()

        if final_data.empty:
            print(f"No data available for runId {runId} and ISIN {row['isin']} in the given time range.")
            continue

        # --- Distribution Fitting ---
        # 1. Student's t-distribution
        t_params = stats.t.fit(final_data)
        t_dist = stats.t(*t_params)

        # 2. Normal (Gaussian) distribution
        norm_params = stats.norm.fit(final_data)
        norm_dist = stats.norm(*norm_params)

        # --- Plotting Section (Modified for Frequency) ---
        num_bins = 50
        fig, ax = plt.subplots(figsize=(10, 7))

        # Get histogram data to calculate bin width
        counts, bin_edges, _ = ax.hist(final_data, bins=num_bins, alpha=0.6, color='purple',
                                       label="Frequency Histogram")

        # Calculate scaling factor
        bin_width = bin_edges[1] - bin_edges[0]
        scaling_factor = len(final_data) * bin_width

        # Generate x-values for the PDF plots
        x = np.linspace(final_data.min(), final_data.max(), 1000)

        # Calculate scaled PDF values
        pdf_t = t_dist.pdf(x) * scaling_factor
        pdf_norm = norm_dist.pdf(x) * scaling_factor

        # Overlay the scaled fits on the histogram
        ax.plot(x, pdf_t, 'r-', lw=2, label="Student's t-distribution fit")
        ax.plot(x, pdf_norm, 'b--', lw=2, label="Normal (Gaussian) fit")

        # --- Chart Customization ---
        titleStr = (f"Histogram for {row['isin']} (Run ID: {runId})\n"
                    f"Start: {row['start_time'].date()} End: {row['end_time'].date()}\n"
                    f"Log Returns: {row['log_returns']}, CAGR: {row['cagr']}, Data Type: {row['data_type']}")

        ax.set_title(titleStr, fontsize=12)
        ax.set_xlabel("Value (Log Returns)", fontsize=10)
        ax.set_ylabel("Frequency (Count)", fontsize=10)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        file_name = titleStr.replace(',', '').replace(':', '')  # Remove commas and colons
        file_name = file_name.replace(' ', '-').replace('\n', '-')  # Replace spaces and new line with dashes
        file_name = file_name + "-ID" + str(runId) + "-wStudentTqGaussianOver2-frequency"
        print(file_name)
        # plt.savefig(file_name)
        # Show the plot
        plt.tight_layout()
        plt.show()
