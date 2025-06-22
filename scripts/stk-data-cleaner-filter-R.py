import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from datetime import datetime

# Establish a connection to your PostgreSQL database
engine = create_engine('postgresql://postgres:<pwd>@localhost:<port>/<market-data-analysis-db>')

# Query to retrieve data
query = "SELECT * FROM stock_data;"
df = pd.read_sql(query, engine)
df['bar_date'] = pd.to_datetime(df['bar_date'], utc=True)
df = df.sort_values('bar_date')

# Function to detect and replace outliers with the median
def replace_outliers_with_median(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 16.625 * IQR
    upper = Q3 + 16.625 * IQR
    median = data[column].median()
    data[column] = np.where((data[column] < lower) | (data[column] > upper), median, data[column])
    return data

def replace_outliers(row, prev_day_price, average_change, change_lower, change_upper, column):
    if pd.isnull(row['price_change']):
        return row[column]
    # if row['price_change'] < change_lower or row['price_change'] > change_upper:
    # In the end, only check for large ABS value, very small difference is fine
    if abs(row['price_change']) > change_upper:
        return prev_day_price + average_change
    return row[column]

finished = 0

def clean_data(data, change_upper_bound, column):
    global finished
    for index, row in data.iterrows():
        if finished == 1:
            return
        date_format = '%Y-%m-%d %H:%M:%S%z'
        date_obj = datetime.strptime('2023-05-29 00:00:00+00:00', date_format)
        if row['bar_date'] == date_obj:
        if abs(row['price_change']) > change_upper_bound:
            data.drop(index, inplace=True)
            data['price_change'] = data[column].diff()
            clean_data(data, change_upper_bound, column)
    finished = 1



def replace_outliers_with_avgChangeAdded(data, column):
    data['price_change'] = data[column].diff()

    # Compute IQR
    Q1 = (data['price_change'].abs()).quantile(0.25)
    Q3 = (data['price_change'].abs()).quantile(0.75)
    IQR = Q3 - Q1
    change_lower_bound = Q1 - 16.5 * IQR
    change_upper_bound = Q3 + 16.5 * IQR

    # Calculate the average change
    average_change = data['price_change'].median()  # EITHER MEDIAN Or MEAN doesn't make a difference ...
    # average_change = 1

    return data


# Replace outliers in each price column
price_columns = ['open', 'high', 'low', 'close']
for column in price_columns:
    df = replace_outliers_with_avgChangeAdded(df, column)
    finished = 0
    # dl = replace_outliers_with_avgChangeAdded(df, column)


# Optionally, update the database with the cleaned data
# df.to_sql('stock_prices_cleaned', con=engine, if_exists='fail', index=False)

# print(df.head())

# Define the date range
start_date = '2019-09-10'
end_date = '2019-10-20'

# Filtering the DataFrame for both ISIN and date range to get SAMPLE of outliers
filtered_df = df[(df['isin'] == 'ARDEUT110046') &
                 (df['bar_date'] >= start_date) &
                 (df['bar_date'] <= end_date)]
print("filtered_df: ", filtered_df[['bar_date', 'open', 'high', 'low', 'close', 'price_change']])

check_df = df.sort_values(by='price_change')
print("my final Adjusted DF, with price change showing close last:\n", check_df)

filtered_df = df[(df['isin'] == 'ARDEUT110046')]
df_sorted = filtered_df.sort_values('bar_date')
plt.figure(figsize=(10, 6))
plt.plot(df_sorted['bar_date'], df_sorted['close'], label='Close Price')
# Adding points to better visualize discrete data
plt.scatter(df_sorted['bar_date'], df_sorted['close'], color='red', label='Data Points')
plt.title(f'Market Data for ARDEUT110046')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
