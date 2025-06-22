# This script is responsible for collecting the EOD Stock Market Data and uploading it to my local PostgreSQL DB

import pandas
import pandas as pd
import sqlalchemy
import requests
import datetime
from datetime import timedelta
from sqlalchemy import create_engine, insert, Table, MetaData

engine = create_engine('postgresql://postgres:<pwd>@localhost:<port>/<market-data-analysis-db>')
metadata = MetaData()
stock_data_table = Table('stock_data', metadata, autoload_with=engine)
conn = engine.connect()

max_years = 30
days_in_year = 365
max_duration = timedelta(days=days_in_year * max_years) #Keeping this as the max historical data,\
# according to docs: https://github.com/FinancialModelingPrepAPI/Financial-Modeling-Prep-API
max_step_years = 5
max_step_duration = timedelta(days=days_in_year * max_step_years)
step_duration = max_step_duration
serie_type = "bar"

symbol_column = 'Symbol'
small_cap_df = pd.read_csv('cap-0.csv')
small_cap_list = small_cap_df[symbol_column].to_list()
mid_cap_df = pd.read_csv('cap-1.csv')
mid_cap_list = mid_cap_df[symbol_column].to_list()
large_cap_df = pd.read_csv('cap-2.csv')
large_cap_list = large_cap_df[symbol_column].to_list()
symbolList = small_cap_list + mid_cap_list + large_cap_list
print(symbolList)
print(len(symbolList))


EOD_base_endpoint ="https://financialmodelingprep.com/api/v3/historical-price-full"
isin_endpoint = "https://financialmodelingprep.com/api/v3/profile"
split_endpoint = "https://financialmodelingprep.com/api/v3/historical-price-full/stock_split"
merger_endpoint = "https://financialmodelingprep.com/api/v4/mergers-acquisitions/search"
FMP_APIKey = "<FMP_API_KEY>"


now = datetime.datetime.now()

for symbol in symbolList:
    end = now
    years_added = timedelta()
    isin_request = isin_endpoint + "/" + symbol
    EOD_data_request = EOD_base_endpoint + "/" + symbol
    queryParams = {'apikey': FMP_APIKey}
    response = requests.get(isin_request, queryParams)
    if response.status_code != 200:
        print("Symbol Search Error: ", response.status_code, "\n", "Symbol: ", symbol)
        break
    company_data = response.json()
    isin = company_data[0]["isin"]
    split_request = split_endpoint + "/" + symbol
    response = requests.get(split_request, params=queryParams)
    split_data = response.json()
    # Only get the first word of the company name, as it weirdly seems to be the std per FMP API Testing
    queryParams["name"] = (company_data[0]["companyName"]).split()[0]
    response = requests.get(merger_endpoint, queryParams)
    merger_data = response.json()
    while years_added < max_duration:
        start = end - step_duration
        queryParams = {'from': start.strftime("%Y-%m-%d"), 'to': end.strftime("%Y-%m-%d"), 'serietype': serie_type,
                       'apikey': FMP_APIKey}
        response = requests.get(EOD_data_request, params=queryParams)


        if response.status_code != 200:
            print("Error: ", response.status_code, "\n", "Symbol: ", symbol, "startDate: ", start)
            break
        bar_data = response.json()
        if "historical" not in bar_data:  # Basically, check if data object is empty,\
            # meaning no EOD data for the date range
            break
        for item in bar_data["historical"]:
            if "historical" in split_data:  # Also check if any split data available
                split = next((split_item["numerator"]/split_item["denominator"] for split_item
                              in split_data["historical"] if split_item["date"] == item["date"]), 0)
                # print("SPLIT: ", split, item["date"])
            else:
                split = 0
            if len(merger_data) != 0:
                merger = next((True for merger_item  # Check that transaction date is same, and that it is same symbol
                              in merger_data if (merger_item["transactionDate"] == item["date"])
                               and (merger_item["symbol"] == symbol)), False)
            else:
                merger = False
            rows = [{'bar_date': item["date"]+"T00:00:00+00:00", 'isin': isin,
                     'open': item["open"], 'high': item["high"], 'low': item["low"], 'close': item["close"],
                     'volume': item["volume"], 'split': split, 'merger': merger}]
            stmt = insert(stock_data_table).values(rows)
            with engine.begin() as conn:
                conn.execute(stmt)
        years_added += step_duration
        end = start
        end -= timedelta(days=1)  # Subtract a day due to INCLUSIVE nature of EOD data time range.

