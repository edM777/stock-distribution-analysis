# An Empirical Analysis of Stock Market Distributions
[![Kaggle](https://img.shields.io/badge/Kaggle-View%20Notebook-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/eduardomurillo/exploring-stock-market-distributions)
For an interactive version of this report where you can run the code and visualizations yourself, please see the public Kaggle Notebook [here](https://www.kaggle.com/code/eduardomurillo/exploring-stock-market-distributions).


## 🚀 Project Overview & Motivation

This project is an in-depth exploratory data analysis that critically evaluates a core tenet of financial theory: the assumption that stock returns are normally distributed. While models like Black-Scholes are built on this premise, this analysis uses rigorous statistical testing on a large, real-world dataset to investigate whether empirical data truly conforms to these theoretical models.

The goal is to provide a data-driven perspective on the actual characteristics of stock price and return distributions, identifying where classical theories align with, and diverge from, observed market behavior.

This project showcases skills in data analysis, statistical modeling, Python programming, and database management.

## 🔑 Key Findings

A comprehensive analysis of **25,530 unique test results** yielded several key insights that challenge classical assumptions:

1.  **Normality is a Rare Anomaly:** Under a strict, combined statistical decision rule, only **7.8%** of the 25,530 samples could be classified as normal. This strongly suggests that normality is not an inherent characteristic of stock market data.
2.  **Distributions are Predominantly Leptokurtic ("Fat-Tailed"):** The most consistent characteristic observed is leptokurtosis. Over **61%** of samples exhibited "fat tails," meaning extreme price movements occur far more frequently than a normal distribution would predict.
3.  **Time Duration is the Most Critical Factor:** The length of the time series was the single most influential parameter. There is a strong negative correlation (-0.43) between duration and normality; as the time frame increases, distributions are significantly less likely to pass normality tests.
4.  **Student's t-Distribution Provides a Superior Fit:** Given the prevalence of leptokurtosis, alternative models were investigated. The **Student's t-distribution** consistently provides a visually and theoretically superior fit to the empirical data, accurately capturing the high peaks and heavy tails that the normal distribution misses.
5.  **Log Returns vs. Raw Prices Show High Agreement:** A paired analysis confirmed a very high correlation in normality classification between log-transformed raw prices and log returns, with over 97% of pairs having matching outcomes. This supports the theoretical relationship between the two.

## 🛠️ Methodology

The analysis followed a multi-stage process:

1.  **Data Acquisition:** Collected daily price data for 100 stocks from the US, Hong Kong, and Sweden across small, mid, and large market caps using the Financial Modeling Prep (FMP) API.
2.  **Data Storage & Preparation:** Data was stored in a PostgreSQL database. Python scripts using `SQLAlchemy` and `pandas` were used to extract, clean, and manipulate the data.
3.  **Statistical Testing Framework:**
    * A Python script (`normality-scanner-tester.py`) was developed to systematically apply a wide range of parameters (e.g., time duration, log returns, CAGR) and generate test results.
    * **Five statistical tests** were used for normality assessment: Kolmogorov-Smirnov, Shapiro-Wilk, Jarque-Bera, Lilliefors, and D'Agostino's K² test.
    * Distribution shape was quantified using **Skewness**, **Kurtosis**, and the **Hurst Exponent**.
4.  **Combined Decision Rules:** To avoid reliance on a single metric, two custom decision rules were created:
    * A **statistical rule** combining p-values, skew, and kurtosis thresholds.
    * A **visual heuristic rule** algorithmically comparing the actual distribution to the expected normal curve using error metrics like Sum of Squared Differences (SSD).

## 💻 Tech Stack

* **Programming/Analysis:** Python, Jupyter Notebook
* **Libraries:** pandas, NumPy, SciPy, Matplotlib, SQLAlchemy, statsmodels
* **Database:** PostgreSQL

## Data Acquisition & Reproducibility

The data for this project was collected from the [Financial Modeling Prep (FMP) API](https://site.financialmodelingprep.com/developer/docs).

**Important:** Due to the FMP API's Terms of Service, the raw data is not included in this repository. To reproduce this analysis, you will need to acquire the data yourself.

**Steps to Acquire the Data:**

1.  **Get an API Key:** Sign up for an account on the FMP website to get your own free or paid API key.
2.  **Set Up Your Environment:** Make sure you have a PostgreSQL instance running and have installed all the packages from `requirements.txt`.
3.  **Run the Collection Script:** Configure and run the `scripts/data-collector-storage.py` script. This will connect to the FMP API using your key, download the necessary stock data, and store it in your local PostgreSQL database, which the analysis notebook can then access.

## 📂 Repository Structure
├── notebooks/ - Contains the main Jupyter Notebook for the analysis.

├── scripts/ - Supporting Python scripts used for data collection and testing.

│   ├── auto-visual-normality-checker.py

│   ├── data-collector-storage.py

│   └── mySamples-studentT-qGaussian-overlayTest-2-frequencyDistAddition.py

|   └── mySamples-studentT-qGaussian-overlayTest.py

│   └── stk-data-cleaner-filter-R.py

├── data/ - A small sample of the stock data. Full data can be acquired via the FMP API. Also includes full test results data.

├── visuals/ - Key plots and figures from the analysis, like correlation matrices and example distributions.

├── README.md - You are here!

├── requirements.txt - Required Python packages for reproducibility.

└── .gitignore - Standard Python gitignore.

## Usage

This project is intended for educational purposes. The code and methodology presented here can be used as a reference or template for similar analyses.

If you use the code or findings from this project in your own work, please provide attribution by citing this repository.
