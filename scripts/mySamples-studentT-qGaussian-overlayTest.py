import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from PIL import Image
import pandas as pd
from sqlalchemy import create_engine, insert, Table, MetaData


# Establish a connection to your PostgreSQL database
engine = create_engine('postgresql://postgres:<pwd>@localhost:<port>/<market-data-analysis-db>')
# Query to retrieve all tests results data (any results, of course, whether filtered or normal)
query = "SELECT * FROM tests_results;"
tests_results_df = pd.read_sql(query, engine)
# Now, get all the stock data (which was used to obtain results). Later on, the stock data can be extracted to get
# specific samples from runs
query_data = "SELECT * FROM stock_data"
stock_data_df = pd.read_sql(query_data, engine)

# Function to calculate prices based on the data type given (End Of Day Average, Close Price, etc.)
# Used by "essential" 'data_type' IF condition, so this generates the required column - 'calculated_values'
def get_calculated_prices(data, data_type):
    data = data.copy()  # Used to prevent Pandas "SettingWithCopyWarning"
    if (data_type == 'eod_avg'):
        data['calculated_values'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        has_inf = np.isinf(data['calculated_values']).any()
        has_nan = np.isnan(data['calculated_values']).any()

        if has_inf:
            print("The data contains infinity values.")
        if has_nan:
            print("The data contains NaN values.")
        if not has_inf and not has_nan:
            print("The data does not contain any infinity or NaN values.")
        print("my data")
        print(data)
        print("\n")
    if (data_type == 'eod_typical'):
        data['calculated_values'] = (data['high'] + data['low'] + data['close']) / 3
        print("TYPICAL")
    if (data_type == 'eod_close'):
        data['calculated_values'] = data['close']
    if (data_type == 'eod_high_low'):
        data['calculated_values'] = (data['high'] + data['low']) / 2
    return data


# runId_list = [13442, 23175, 3335, 7, 785, 7929, 1739, 399, 1507, 2287]
runId_list = [2304, 2240]
for runId in runId_list:
    my_row = tests_results_df[tests_results_df['runid'] == runId]
    print(my_row)
    for index, row in my_row.iterrows():
        stock_data = stock_data_df[stock_data_df['isin'].isin([row['isin']])]
        stock_data = stock_data.copy()
        # Convert 'bar_date' to datetime and sort DataFrame
        stock_data['bar_date'] = pd.to_datetime(stock_data['bar_date'], utc=True)
        stock_data = stock_data.sort_values('bar_date')
        stock_data.reset_index(drop=True, inplace=True)
        stock_data = get_calculated_prices(stock_data, row['data_type'])
        if row['cagr']:
            initial_value = stock_data['calculated_values'].iloc[0]
            final_value = stock_data['calculated_values'].iloc[-1]
            latest_date = stock_data['bar_date'].max()
            earliest_date = stock_data['bar_date'].min()
            days = (latest_date - earliest_date).days
            cagr_factor = ((final_value / initial_value) ** (365 / days)) - 1
            daily_price_factor = cagr_factor / 365
            stock_data['calculated_values'] = (stock_data['calculated_values'] /
                                               ((1 + daily_price_factor) ** (stock_data['bar_date'] -
                                                                             stock_data['bar_date'].min()).dt.days))
            # print("my CAGR data")
            # print(stock_data)
        if row['log_returns']:
            stock_data['calculated_values'] = np.log((stock_data['calculated_values']) /
                                                     (stock_data['calculated_values'].shift(1)))
            stock_data = stock_data.drop(0)  # Drop just the first row since it will always be Nan due to the shift.
            stock_data.reset_index(drop=True, inplace=True)
        stock_data = stock_data[(stock_data['bar_date'] >= row['start_time'])]
        stock_data = stock_data[(stock_data['bar_date'] <= row['end_time'])]


        # Fit the synthetic data to both distributions (Student's t and Tsallis q-Gaussian)
        # Student's t-distribution
        stock_data = stock_data['calculated_values']
        t_params = stats.t.fit(stock_data)
        t_dist = stats.t(*t_params)
        
        # Tsallis q-Gaussian approximation (using a Generalized Gaussian)
        # We'll estimate q empirically for simplicity
        q_params = stats.norm.fit(stock_data)  # Assuming a Gaussian-like fit for q ~ 1
        q_gaussian = stats.norm(*q_params)
        
        # Overlay the original histogram and the two fits
        x = np.linspace(-0.1, 0.1, 1000)
        pdf_t = t_dist.pdf(x)
        pdf_q = q_gaussian.pdf(x)
        
        # Plotting the original histogram with both fits
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(stock_data, bins=30, density=True, alpha=0.6, color='purple', label="Histogram (synthetic)")
        ax.plot(x, pdf_t, 'r-', label="Student's t-distribution fit")
        ax.plot(x, pdf_q, 'b--', label="Tsallis q-Gaussian fit")

        titleStr = ("Histogram for " + row['isin'] + " - Start: " + str(row['start_time']) + " End: " + str(
            row['end_time'])
                    + "\n" + "Log_returns: " + str(row['log_returns']) + ", CAGR: "
                    + str(row['cagr']) + ", Data Type: " + str(row['data_type']))
        # Customization
        ax.set_title(titleStr)
        ax.set_xlabel("Price")
        ax.set_ylabel("Density")
        ax.legend()
        file_name = titleStr.replace(',', '').replace(':', '')  # Remove commas and colons
        file_name = file_name.replace(' ', '-').replace('\n', '-')  # Replace spaces and new line with dashes
        file_name = file_name + "-ID" + str(runId) + "-wStudentTqGaussianOver"
        print(file_name)
        # plt.savefig(file_name)
        # Show the plot
        plt.tight_layout()
        plt.show()
