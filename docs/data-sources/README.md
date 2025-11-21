# Data Sources: Schwab (primary) and yfinance (fallback)

This feature uses Schwab Trader API as the primary source and yfinance as a fallback for gaps or errors.

## Providers and Capabilities

- **Schwab** (Trader API; see `docs/*.pdf`)
  - Quotes, OHLCV, option chains (bid/ask/IV/OI/volume), market data ticks
  - Fundamentals and analyst info (per Trader API docs)
  - Auth: token-based; respect documented rate limits and throttling
  - Use for: Stage 0-2 filters, pricing inputs, monitoring refreshes
- **yfinance** (fallback)
  - Quotes/ohlcv/option_chain; no authenticated limits
  - Use when Schwab capability missing or request fails; emit warning

## Usage Patterns

- DataSource interface (config names: `schwab`, `yfinance`)
- Fallback chaining: attempt Schwab, fallback to yfinance when enabled
- DataLoader caches under `data/historical` partitions; option chains should capture bid/ask/IV/OI/volume

## Notes

- Keep rate limits in mind when running in parallel; avoid hammering Schwab.
- Log provenance (provider, timestamp, interval) with cached data for diagnostics.
- Reference: planning/yfinance_specs.md and Trader API PDFs in `docs/` for contract details.
