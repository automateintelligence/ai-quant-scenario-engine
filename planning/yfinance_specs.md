# The yfinance API provides a wealth of stock information beyond just price data, accessible through attributes and methods of the Ticker object. These include: 
## Historical Market Data
When using the .history() method, the resulting pandas DataFrame includes these fields in addition to Open, High, Low, and Close price: 
- Adjusted Close: The closing price after adjustments for corporate actions like dividends and stock splits
- Volume: The number of shares traded during the period.
- Dividends: The dividend amount paid on a specific date.
- Stock Splits: Information on stock splits that occurred. 
## Company Fundamentals and Information
The .info attribute returns a dictionary containing a wide range of metadata and financial metrics, including: 
- Company Information: Name, sector, industry, and a summary description.
- Key Metrics: Market cap, P/E ratios, earnings per share (EPS), and revenue.
- Dates: Earnings dates and ex-dividend dates.
- Valuation Data: Various financial ratios and statistics.
- Employee Count. 
## Financial Statements
Methods are available to access detailed financial reports as pandas DataFrames: 
- .income_stmt / .quarterly_income_stmt: Income statements.
- .balance_sheet / .quarterly_balance_sheet: Balance sheets.
- .cashflow / .quarterly_cashflow: Cash flow statements. 
## Other Data Types
- Corporate Actions: .actions provides a history of dividends and stock splits.
- Options Data: .option_chain() allows access to specific options contracts, including strike prices, bid/ask, volume, open interest, and implied volatility.
- Analyst Recommendations: .analyst_price_targets provides data on what analysts are recommending for the stock.
- Sustainability Scores: .sustainability provides environmental, social, and governance (ESG) data.
- Institutional/Insider Holdings: Information on major shareholders and insider transactions.
- News: Access to related market news articles. 
REF: https://ranaroussi.github.io/yfinance/reference/index.html