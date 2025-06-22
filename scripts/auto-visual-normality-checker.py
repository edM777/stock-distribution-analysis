import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
from scipy.signal import find_peaks

# --- (Database connection and data loading - same as before) ---
engine = create_engine('postgresql://postgres:<pwd>@localhost:<port>/<market-data-analysis-db>')
query = "SELECT * FROM tests_results;"
tests_results_df = pd.read_sql(query, engine)
query_data = "SELECT * FROM stock_data"
stock_data_df = pd.read_sql(query_data, engine)


def get_calculated_prices(data, data_type):
    # (Same as before)
    data = data.copy()
    if data_type == 'eod_avg':
        data['calculated_values'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    elif data_type == 'eod_typical':
        data['calculated_values'] = (data['high'] + data['low'] + data['close']) / 3
    elif data_type == 'eod_close':
        data['calculated_values'] = data['close']
    elif data_type == 'eod_high_low':
        data['calculated_values'] = (data['high'] + data['low']) / 2
    return data


# Calculate the average p-value for each row
normality_columns = ['ks_test', 'jarque_bera_test', 'lilliefors_test', 'shapiro_test', 'dagostino_test']
tests_results_df['average_p_value'] = tests_results_df[normality_columns].mean(axis=1)

# Determine normal distribution based on combined decision rule
tests_results_df['is_normal'] = ((tests_results_df['average_p_value'] > 0.05) & (tests_results_df['skew'].abs() < 0.5) &
                                 ((tests_results_df['kurtosis'] - 3).abs() < 0.5))

# Filter rows that resulted in normal distribution
tests_normal_distributions = tests_results_df[tests_results_df['is_normal']]
pass_combined_rule_runs = tests_normal_distributions['runid'].tolist()
print("ALL RUNIDs which PASSED combined decision Rule: ", pass_combined_rule_runs)

# Display the count of normal distributions
normal_distribution_count = tests_normal_distributions.shape[0]
total_count = tests_results_df.shape[0]
normal_distribution_likelihood = normal_distribution_count / total_count

print(f"Number of normal distributions: {normal_distribution_count}")
print(f"Total number of runs: {total_count}")
print(f"Likelihood of normal distribution: {normal_distribution_likelihood * 100:.2f}%")


# Parameters
has_start = True
has_end = True
has_cagr = True
has_log_returns = True
has_data_type = True
max_bins = 210  # Maximum number of bins

# --- Thresholds (NEED TUNING!) ---
ssd_threshold = 13  # Threshold for Sum of Squared Differences (Weighted Chi^2)
max_abs_diff_threshold = 8  # Threshold for Maximum Absolute Difference
max_peaks = 1

N_list = []
days_list = []
pass_list = []
counter = 0
ln_count = tests_normal_distributions['log_returns'].sum()  # Counts LN runs for passes of combined rule, of course


# Iterate through tests results
for index, row in tests_results_df.iterrows():
    stock_data = stock_data_df[stock_data_df['isin'] == row['isin']].copy()
    stock_data['bar_date'] = pd.to_datetime(stock_data['bar_date'], utc=True)
    stock_data = stock_data.sort_values('bar_date').reset_index(drop=True)

    days = (stock_data['bar_date'].max() - stock_data['bar_date'].min()).days
    if has_data_type:
        stock_data = get_calculated_prices(stock_data, row['data_type'])
    if has_cagr and row['cagr']:
        initial_value = stock_data['calculated_values'].iloc[0]
        final_value = stock_data['calculated_values'].iloc[-1]

        cagr_factor = ((final_value / initial_value) ** (365 / days)) - 1
        daily_price_factor = cagr_factor / 365
        stock_data['calculated_values'] /= (1 + daily_price_factor) ** (
                    stock_data['bar_date'] - stock_data['bar_date'].min()).dt.days
    if has_log_returns and row['log_returns']:
        stock_data['calculated_values'] = np.log(
            stock_data['calculated_values'] / stock_data['calculated_values'].shift(1))
        stock_data.dropna(subset=['calculated_values'], inplace=True)
        stock_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        stock_data.dropna(subset=['calculated_values'], inplace=True)
        stock_data.reset_index(drop=True, inplace=True)
    if has_start:
        stock_data = stock_data[stock_data['bar_date'] >= row['start_time']]
    if has_end:
        stock_data = stock_data[stock_data['bar_date'] <= row['end_time']]
    true_days = (stock_data['bar_date'].max() - stock_data['bar_date'].min()).days

    titleStr = ("Histogram for " + row['isin'] + " - Start: " + str(row['start_time']) + " End: " + str(row['end_time'])
                + "\n" + "Log_returns: " + str(row['log_returns']) + ", CAGR: "
                + str(row['cagr']) + ", Data Type: " + str(row['data_type']))
    if has_log_returns and not  row['log_returns']:
        titleStr = titleStr + "(log transformation)"  # Added to make sure I remember how data is displayed, transformed
        stock_data = stock_data[stock_data['calculated_values'] > 0].copy()
        if not stock_data.empty:  # Check if data remains after filtering
            stock_data['calculated_values'] = np.log(stock_data['calculated_values'])
            # Clean up any resulting NaNs (though filtering > 0 should prevent most issues)
            stock_data.dropna(subset=['calculated_values'], inplace=True)
        else:
            print(f"Warning: Run {row['runid']} - No positive data left for log transformation. Skipping plot.")
            continue  # Skip to the next iteration if no data left
    # Check if DataFrame is empty after potential log transformation and dropna
    if stock_data.empty:
        print(f"Warning: Run {row['runid']} - DataFrame is empty before plotting. Skipping.")
        continue

    data = stock_data['calculated_values']
    mean, std_dev = np.mean(data), np.std(data)

    # Get the total data points for a figure / run - N
    N = len(data)
    # print ("This run has data points N = ", N)
    N_list.append(N)
    days_list.append(true_days)
    

    # --- 1. Generate Histogram Data ---
    try:
        hist_counts, bin_edges = np.histogram(data, bins='auto')  # Use 'auto' or a fixed number
        if len(hist_counts) > max_bins:
            hist_counts, bin_edges = np.histogram(data, bins=max_bins)
    except MemoryError:
        hist_counts, bin_edges = np.histogram(data, bins=max_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # --- 2. Generate Expected Normal Distribution Values ---
    expected_counts = []
    for i in range(len(bin_edges) - 1):
        prob, _ = quad(norm.pdf, bin_edges[i], bin_edges[i + 1], args=(mean, std_dev))
        expected_counts.append(prob * len(data))
    expected_counts = np.array(expected_counts)

    # --- 3. Calculate Difference Metrics ---
    ssd = np.sum((hist_counts - expected_counts) ** 2)  # Sum of Squared Differences (Weighted Chi^2)
    max_abs_diff = np.max(np.abs(hist_counts - expected_counts))
    peaks, _ = find_peaks(hist_counts)
    num_peaks = len(peaks)

    # --- 4. Automated Decision based on Combined Criteria ---
    if ssd < ssd_threshold and max_abs_diff < max_abs_diff_threshold and num_peaks <= max_peaks:
        print(
            f"Run {row['runid']}: Visually Normal (SSD = {ssd:.2f}, Max Abs Diff = {max_abs_diff:.2f}, Num Peaks = {num_peaks})")
        pass_list.append(row['runid'])
        counter += 1
        # continue
    else:
        print(
            f"Run {row['runid']}: NOT Visually Normal (SSD = {ssd:.2f}, Max Abs Diff = {max_abs_diff:.2f}, Num Peaks = {num_peaks})")
        # continue

    # --- Plotting (for Visualization) ---
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bin_edges, color='purple', alpha=0.7, label='Histogram')
    x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 200)
    plt.plot(x, norm.pdf(x, mean, std_dev) * len(data) * (bin_edges[1] - bin_edges[0]), 'k', linewidth=2,
             label='Normal Distribution')
    plt.title(titleStr)
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.legend()
    file_name = titleStr.replace(',', '').replace(':', '')  # Remove commas and colons
    file_name = file_name.replace(' ', '-')  # Replace spaces with dashes
    file_name = (file_name + "-ID" + str(row['runid']))
    file_name = file_name + "-wOver-scaleCorrection.png"
    file_name = "correction-2-normal-line-scaled" + "/" + file_name
    print("FILE NAME: ", file_name)
    # plt.savefig(file_name)
    plt.show()


print("ALL RunIds that are visually normal according to these tests:")
print(pass_list)
print("TOTAL Count: ", counter)

print("Stats on data points, N:")
print("N_list = ", N_list)
average_N = sum(N_list) / len(N_list)
print("Average N: ", average_N, "    Range of N: ", min(N_list), " - ", max(N_list))
print("\ndays_list = ", days_list)
average_days = sum(days_list) / len(days_list)
print("Average days: ", average_days, "    Range of N: ", min(days_list), " - ", max(days_list))
print("Total runs which have log_returns=True in Combined decision rule results: ", ln_count)
