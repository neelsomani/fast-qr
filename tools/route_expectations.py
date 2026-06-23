from __future__ import annotations

from typing import Any


CASE_SPECIFIC_ROUTES: dict[int, dict[str, str]] = {
    512: {
        "dense": "qr512_dense_fast",
        "mixed": "qr512_mixed_fast",
        "rankdef": "qr512_rankdef_fast",
        "clustered": "qr512_clustered_fast",
    },
    1024: {
        "dense": "qr1024_dense_fast",
        "mixed": "qr1024_mixed_fast",
        "rankdef": "qr1024_rankdef_fast",
        "clustered": "qr1024_clustered_fast",
        "nearrank": "qr1024_nearrank_fast",
    },
    2048: {
        "dense": "qr2048_dense_fast",
        "mixed": "qr2048_mixed_fast",
        "rankdef": "qr2048_rankdef_fast",
    },
    4096: {
        "dense": "qr4096_dense_fast",
    },
}


GENERIC_FAMILY_ROUTES: dict[int, set[str]] = {
    512: {
        "qr512_cuda_fast",
        "qr512_blocked_cuda_fast",
        "qr512_blocked_cuda_auto_fast",
    },
    1024: {
        "qr1024_cuda_fast",
        "qr1024_blocked_cuda_fast",
        "qr1024_blocked_cuda_auto_fast",
    },
    2048: {
        "qr2048_blocked_cuda_fast",
        "qr2048_blocked_cuda_auto_fast",
    },
    4096: {
        "qr4096_blocked_cuda_fast",
        "qr4096_blocked_cuda_auto_fast",
    },
}


def expected_case_route(spec: dict[str, Any]) -> str | None:
    n = int(spec["n"])
    case = str(spec.get("case", "dense"))
    return CASE_SPECIFIC_ROUTES.get(n, {}).get(case)


def allowed_family_routes(spec: dict[str, Any]) -> set[str] | None:
    n = int(spec["n"])
    expected = expected_case_route(spec)
    routes = GENERIC_FAMILY_ROUTES.get(n)
    if routes is None:
        return None if expected is None else {expected}
    allowed = set(routes)
    if expected is not None:
        allowed.add(expected)
    return allowed


def route_ok_for_spec(route: str, spec: dict[str, Any]) -> bool:
    allowed = allowed_family_routes(spec)
    return allowed is None or route in allowed
