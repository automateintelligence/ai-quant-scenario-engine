"""Load validated distribution models from cached audit results (US6a AS9)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from quant_scenario_engine.distributions.cache.cache_manager import (
    TTL_DAYS,
    get_cache_path,
    is_fresh,
    load_cache,
)
from quant_scenario_engine.distributions.cache.serializer import deserialize_payload
from quant_scenario_engine.distributions.distribution_audit import instantiate_distribution
from quant_scenario_engine.distributions.integration.cache_checker import warn_if_stale
from quant_scenario_engine.distributions.integration.fallback import laplace_fallback_distribution
from quant_scenario_engine.distributions.integration.metadata_logger import build_audit_metadata
from quant_scenario_engine.exceptions import DistributionFitError
from quant_scenario_engine.interfaces.distribution import ReturnDistribution
from quant_scenario_engine.utils.logging import get_logger

log = get_logger(__name__, component="distribution_model_loader")


@dataclass
class LoadedModel:
    """Container for a loaded distribution and its provenance."""

    distribution: ReturnDistribution
    metadata: Dict[str, Any]
    source: str  # "cache" | "fallback"
    cache_path: Optional[Path]


def _apply_cached_params(dist: ReturnDistribution, params: Dict[str, Any]) -> None:
    """Set distribution attributes from cached FitResult params when possible."""
    for key, value in params.items():
        # attribute names on distribution classes match parameter keys for laplace/student_t
        if hasattr(dist, key.replace("[1]", "1")):
            setattr(dist, key.replace("[1]", "1"), value)


def _extract_best(payload: Dict[str, Any]) -> tuple[Optional[str], Dict[str, Any]]:
    best_model = None
    best_fit: Dict[str, Any] = {}
    if payload.get("best_model"):
        raw = payload["best_model"]
        if isinstance(raw, dict):
            best_model = raw.get("name") or raw.get("model_name")
    if payload.get("best_fit"):
        if isinstance(payload["best_fit"], dict):
            best_fit = payload["best_fit"]
            best_model = best_model or best_fit.get("model_name")
    return best_model, best_fit


def _find_latest_cache_entry(
    cache_dir: Path,
    symbol: str,
    *,
    allow_stale: bool,
) -> Tuple[Optional[Path], Optional[dict], bool]:
    pattern = f"{symbol}_*.json"
    candidates = sorted(cache_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        payload = load_cache(path)
        if not payload:
            continue
        stale_flag = not is_fresh(path, ttl_days=TTL_DAYS)
        if stale_flag:
            warn_if_stale(path, ttl_days=TTL_DAYS)
            if not allow_stale:
                continue
        return path, payload, stale_flag
    return None, None, False


def load_validated_model(
    *,
    symbol: str,
    lookback_days: int | None,
    end_date: str | None,
    data_source: str | None,
    cache_dir: str | None = None,
    allow_stale: bool = True,
    allow_symbol_fallback: bool = True,
) -> LoadedModel:
    """Load the validated distribution model for a symbol from cache or fallback.

    This function is used by US1/US6 Monte Carlo workflows to automatically
    prefer the empirically selected model from the distribution audit.
    """

    cache_base = Path(cache_dir) if cache_dir else Path("output") / "distribution_audits"
    cache_path = get_cache_path(cache_base, symbol, lookback_days, end_date, data_source)
    payload = load_cache(cache_path)

    def _build_from_payload(raw_payload: dict, path: Path, stale_flag: bool) -> LoadedModel:
        data = deserialize_payload(json.dumps(raw_payload))
        best_model, best_fit = _extract_best(data)
        params = best_fit.get("params") if isinstance(best_fit.get("params"), dict) else {}

        if best_model is None:
            log.warning(
                "Cached audit missing best_model; falling back to Laplace",
                extra={"path": str(path)},
            )
            dist, meta = laplace_fallback_distribution("cached audit missing best model")
            return LoadedModel(distribution=dist, metadata=meta, source="fallback", cache_path=None)

        try:
            dist = instantiate_distribution(best_model)
            _apply_cached_params(dist, params)
        except DistributionFitError as exc:
            log.warning(
                "Cached model unsupported; defaulting to Laplace",
                extra={"model": best_model, "error": str(exc)},
            )
            dist, meta = laplace_fallback_distribution(f"unsupported cached model: {best_model}")
            return LoadedModel(distribution=dist, metadata=meta, source="fallback", cache_path=None)

        metadata = build_audit_metadata(
            best_model=best_model,
            best_fit=best_fit,
            scores=data.get("scores") if isinstance(data.get("scores"), list) else [],
            cache_path=path,
            stale=stale_flag,
        )
        log.info(
            "Loaded validated distribution model from cache",
            extra={"model": best_model, "path": str(path), "stale": stale_flag},
        )
        return LoadedModel(distribution=dist, metadata=metadata, source="cache", cache_path=path)

    stale_flag = False
    if payload:
        stale_flag = not is_fresh(cache_path, ttl_days=TTL_DAYS)
        if stale_flag:
            warn_if_stale(cache_path, ttl_days=TTL_DAYS)
            if not allow_stale:
                payload = None

    if payload:
        return _build_from_payload(payload, cache_path, stale_flag)

    should_try_symbol_cache = allow_symbol_fallback and (
        lookback_days is None or end_date is None or data_source is None
    )
    if should_try_symbol_cache and cache_base.exists():
        alt_path, alt_payload, alt_stale = _find_latest_cache_entry(
            cache_base, symbol, allow_stale=allow_stale
        )
        if alt_payload:
            log.info(
                "Using latest cached audit entry for symbol",
                extra={"symbol": symbol, "path": str(alt_path)},
            )
            return _build_from_payload(alt_payload, alt_path, alt_stale)

    dist, meta = laplace_fallback_distribution("no cached audit available")
    return LoadedModel(distribution=dist, metadata=meta, source="fallback", cache_path=None)


__all__ = ["load_validated_model", "LoadedModel"]
