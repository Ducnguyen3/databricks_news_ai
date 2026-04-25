from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_DROP_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "zarsrc",
}


def normalize_url(url: str | None) -> str:
    if not url:
        return ""
    value = url.strip()
    if not value:
        return ""
    parsed = urlsplit(value)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"

    query_pairs = []
    for key, val in parse_qsl(parsed.query, keep_blank_values=False):
        lowered_key = key.lower()
        if lowered_key.startswith("utm_") or lowered_key in _DROP_QUERY_KEYS:
            continue
        query_pairs.append((key, val))

    query = urlencode(sorted(query_pairs))
    normalized = urlunsplit((scheme, netloc, path, query, ""))
    return normalized.rstrip("/")

