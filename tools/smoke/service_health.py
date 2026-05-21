"""HTTP health checks for platform microservices."""
from __future__ import annotations

from typing import Any

import httpx


def check_service_health(
    service: str,
    url: str,
    *,
    timeout: float = 5.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """GET *url* and validate JSON health payload.

    Raises
    ------
    httpx.HTTPError
        Connection / timeout failures (caller formats message with *service*).
    AssertionError
        Non-200 status or unhealthy body.
    """
    if client is not None:
        try:
            response = client.get(url)
        except httpx.HTTPError as exc:
            raise httpx.ConnectError(
                f"{service} unreachable at {url}: {exc}",
            ) from exc
    else:
        try:
            response = httpx.get(url, timeout=timeout)
        except httpx.HTTPError as exc:
            raise httpx.ConnectError(
                f"{service} unreachable at {url}: {exc}",
            ) from exc

    if response.status_code != 200:
        raise AssertionError(
            f"{service} returned HTTP {response.status_code} from {url}"
        )
    raw_body = response.json()
    if not isinstance(raw_body, dict):
        raise AssertionError(f"{service} health body is not a JSON object: {raw_body!r}")
    body: dict[str, Any] = raw_body
    status = body.get("status")
    if status not in ("ok", "healthy"):
        raise AssertionError(f"{service} unhealthy status={status!r} body={body}")
    return body
