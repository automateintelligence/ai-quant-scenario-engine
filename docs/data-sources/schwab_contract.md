# Schwab Trader API Contracts (Market Data)

Source docs: `docs/Trader API - Individual _ Products _ Charles Schwab Developer Portal*.pdf` and `planning/yfinance_specs.md` (fallback behavior). This summary maps request/response fields we need for FR-004/FR-005 and highlights gaps for edge handling.

## Authentication & Transport
- **Base URL:** `https://api.schwabapi.com/marketdata/v1`
- **Auth:** Bearer token from OAuth `POST /token` (access token required on every call). Include `Authorization: Bearer <token>` and `accept: application/json`.
- **Headers surfaced in responses:**
  - `Schwab-Client-CorrelId` – GUID for traceability.
  - `Schwab-Resource-Version` – requested API version.
- **Errors:** JSON array under `errors[]` with `status`, `title`/`detail`, and `source` pointer. Common statuses: 400 (missing symbols/header), 401 (missing/expired token), 404 (symbol not found), 500 (server fault).

## Quotes
`GET /quotes?symbols=SYMA,SYMB&fields=quote,fundamental,extended,reference&indicative=false`
- **Params:**
  - `symbols` (comma-separated, supports equities, indexes `$SPX`, options `SYMB 20250117C00100000`).
  - `fields` root selectors: `quote`, `fundamental`, `extended`, `reference`, `regular` (omit for full payload).
  - `indicative` (boolean) to include ETF iNAV symbols.
- **Response (per symbol):**
  - Top-level keys: `assetMainType`, `assetSubType`, `quoteType`, `realtime`, `symbol`, `ssid`.
  - `quote` node: bid/ask/last sizes and MICs, `highPrice/lowPrice/closePrice`, 52-week stats.
  - `fundamental` node: EPS, PE, beta, div data, analyst ratings/targets.
  - Error payloads follow standard `errors[]` envelope.

## Option Chains
`GET /chains?symbol=SYMB&fromDate=2025-01-17&toDate=2025-01-17`
- **Behavior:** Returns `callExpDateMap` and `putExpDateMap` keyed by `EXPIRY:daysToExpiry -> strike -> [contracts]`.
- **Contract fields observed:** `strikePrice`, `bid`, `ask`, `volatility` (aka IV), `openInterest`, `totalVolume`, plus Greeks (`delta`, `gamma`, `theta`, `vega`) when available.
- **Normalization for loader:** flatten to rows with columns `expiry`, `strike`, `option_type`, `bid`, `ask`, `implied_volatility`, `open_interest`, `volume`, and optional Greeks. Missing legs or empty maps should trigger fallback to yfinance.

## Price History (OHLCV)
`GET /pricehistory?symbol=SYMB&periodType=year&period=5&frequencyType=daily&frequency=1`
- **Params:**
  - `symbol` (required).
  - `periodType`: `day|month|year|ytd` (default `year`).
  - `period`: values depend on periodType (day: 1/2/3/4/5/10, month: 1/2/3/6, year: 1/2/3/5/10/15/20, ytd: 1).
  - `frequencyType`: `minute` (for day), `daily|weekly` (month/year/ytd), `monthly` (year).
  - `frequency`: minute granularity `1|5|10|15|30`, otherwise `1`.
  - Optional: `startDate`, `endDate` (epoch ms), `needExtendedHoursData`, `needPreviousClose`.
- **Response:** `candles[]` with `datetime` (epoch ms), `open`, `high`, `low`, `close`, `volume`. Used for OHLCV cache; empty candles trigger fallback.

## Fundamentals & Analyst Info
- Delivered via `fields=fundamental` on `/quotes`.
- Includes dividend info, EPS, beta, valuation ratios; analyst stats surface as ratings/target fields. If the fundamental node is absent, log and fallback to yfinance fundamentals (Ticker.info).

## Rate Limits & Retries
- API docs note OAuth + correlation IDs but omit exact rate limits. Implement caller-side timeouts (default 10s) and warn when falling back.
- Fallback plan: on `DataSourceError` from Schwab, call yfinance with the same symbol/interval, keeping retries/backoff from `planning/yfinance_specs.md` (default 3 attempts with 1/2/4s backoff).

## Gaps / Edge Handling
- Option chain request parameters beyond symbol/fromDate/toDate (e.g., `strikeCount`, `includeQuotes`) are not fully specified in the PDFs; keep defaults and surface warnings when maps are empty.
- Fundamental/analyst coverage varies by instrument; treat missing nodes as recoverable and fall back.
- Price history queries rely on periodType/period when explicit start/end are absent; our loader passes explicit dates and validates candles to avoid future timestamps.
