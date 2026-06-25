"""Fetch arXiv HTML or PDF files with conservative robots and cache handling."""

import json
import os
import tempfile
import time
import urllib.robotparser
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

BASE = "https://arxiv.org"
CACHE_DIR = Path("cache")
ROBOTS_URL = f"{BASE}/robots.txt"

# arXiv declares Crawl-delay: 15 for ordinary crawlers. One extra second
# provides a scheduling margin.
MIN_CRAWL_DELAY = 16.0
CRAWL_DELAY_MARGIN = 0.0
BACKOFF_429 = 60.0
REQUEST_TIMEOUT = 30

_last_request_at = None


def _user_agent():
    """Build an identifiable user agent from the configured contact email.

    Raises RuntimeError if ARXIV_CONTACT_EMAIL is not set.
    """
    email = os.environ.get("ARXIV_CONTACT_EMAIL", "").strip()
    if not email or "@" not in email:
        raise RuntimeError(
            "Set ARXIV_CONTACT_EMAIL to a real contact address before fetching arXiv"
        )
    return f"OTHAW-NLP-Project/1.0 (student research; contact: {email})"


def _wait_for_request_slot(delay):
    """Block until at least delay seconds have passed since the last request."""
    global _last_request_at
    now = time.monotonic()
    if _last_request_at is not None:
        remaining = delay - (now - _last_request_at)
        if remaining > 0:
            time.sleep(remaining)
    _last_request_at = time.monotonic()


def _request(url, delay):
    """Make one rate-limited GET request to arXiv.
    Raises RuntimeError on connection failure or timeout.
    """
    _wait_for_request_slot(delay)
    try:
        return requests.get(
            url,
            headers={"User-Agent": _user_agent()},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"arXiv request failed for {url}: {exc}") from exc


def _robots():
    """Load and parse arXiv robots.txt. Attaches effective_delay to the parser.

    Raises RuntimeError if robots.txt cannot be retrieved. Fail-closed: no
    paper request is made without a confirmed allow decision.
    """
    response = _request(ROBOTS_URL, MIN_CRAWL_DELAY + CRAWL_DELAY_MARGIN)
    if response.status_code != 200 or not response.text.strip():
        raise RuntimeError(
            f"Could not load arXiv robots.txt (HTTP {response.status_code}); stopping"
        )
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(ROBOTS_URL)
    parser.parse(response.text.splitlines())
    user_agent = _user_agent()
    declared_delay = parser.crawl_delay(user_agent) or parser.crawl_delay("*") or MIN_CRAWL_DELAY
    parser.effective_delay = max(float(declared_delay), MIN_CRAWL_DELAY) + CRAWL_DELAY_MARGIN
    parser.request_user_agent = user_agent
    print(f"arXiv robots.txt loaded; effective request interval={parser.effective_delay:.0f}s")
    return parser


def _cache_path(arxiv_id, kind):
    """Return the cache path for arxiv_id with extension html or pdf."""
    ext = "html" if kind == "html" else "pdf"
    return CACHE_DIR / f"{arxiv_id}.{ext}"


def _valid_payload(body, kind, content_type):
    """Return True if body matches the expected media type and file signature."""
    media_type = content_type.split(";", 1)[0].strip().lower()
    if kind == "pdf":
        return media_type == "application/pdf" and body.startswith(b"%PDF-")
    prefix = body[:4096].lower()
    return media_type in {"text/html", "application/xhtml+xml"} and (
        b"<html" in prefix or b"<!doctype html" in prefix
    )


def _valid_cached_file(path, kind):
    """Return True if the cached file exists and has a valid HTML or PDF signature."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as f:
        prefix = f.read(4096).lower()
    if kind == "pdf":
        return prefix.startswith(b"%pdf-")
    return b"<html" in prefix or b"<!doctype html" in prefix


def _atomic_cache_write(path, body, kind):
    """Write body to path atomically via a temp file, ensuring no partial writes."""
    CACHE_DIR.mkdir(exist_ok=True)
    data = body if kind == "pdf" else body.decode("utf-8", "replace").encode("utf-8")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=CACHE_DIR,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _audit_fetch(arxiv_id, url, kind, status, robots_allowed, delay, outcome):
    """Append one fetch record to fetch_audit.jsonl for provenance and compliance."""
    CACHE_DIR.mkdir(exist_ok=True)
    record = {
        "arxiv_id": arxiv_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "url": url,
        "kind": kind,
        "http_status": status,
        "robots_allowed": robots_allowed,
        "request_interval_seconds": delay,
        "outcome": outcome,
    }
    with (CACHE_DIR / "fetch_audit.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _retry_after_seconds(response):
    """Parse the Retry-After header and return seconds to wait, or None."""
    value = response.headers.get("Retry-After", "").strip()
    if not value:
        return None
    if value.isdigit():
        return float(value)
    try:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _get_with_backoff(url, delay):
    """GET once, with one server-directed retry on HTTP 429.

    Raises RuntimeError on repeated 429 or blocking status codes (403, 503).
    Continuing after those would risk an IP-level block.
    """
    for attempt in (1, 2):
        response = _request(url, delay)
        if response.status_code == 429:
            if attempt == 2:
                raise RuntimeError(f"Repeated HTTP 429 from {url}; stopping fetch run")
            wait = max(BACKOFF_429, _retry_after_seconds(response) or 0.0)
            print(f"HTTP 429 received; backing off {wait:.0f}s before one retry")
            time.sleep(wait)
            continue
        if response.status_code in {403, 503}:
            raise RuntimeError(
                f"HTTP {response.status_code} from {url}; stopping fetch run conservatively"
            )
        return response
    raise RuntimeError(f"No response returned for {url}")


def fetch_one(arxiv_id, rp):
    """Download HTML (preferred) or PDF for arxiv_id, caching the result.

    Returns (path, kind) where kind is 'html', 'pdf', or 'missing'.
    """
    CACHE_DIR.mkdir(exist_ok=True)

    html_path = _cache_path(arxiv_id, "html")
    if _valid_cached_file(html_path, "html"):
        print(f"{arxiv_id}: cached (html), skipping download")
        return html_path, "html"
    if html_path.exists():
        print(f"{arxiv_id}: cached html failed validation; requesting a clean copy")

    pdf_path = _cache_path(arxiv_id, "pdf")
    cached_pdf = _valid_cached_file(pdf_path, "pdf")
    if pdf_path.exists() and not cached_pdf:
        print(f"{arxiv_id}: cached pdf failed validation; requesting a clean copy")

    delay = getattr(rp, "effective_delay", MIN_CRAWL_DELAY + CRAWL_DELAY_MARGIN)
    user_agent = getattr(rp, "request_user_agent", None) or _user_agent()

    for kind, url_template in (("html", "/html/{}"), ("pdf", "/pdf/{}")):
        if kind == "pdf" and cached_pdf:
            print(f"{arxiv_id}: cached (pdf), no HTML available")
            return pdf_path, "pdf"

        url = BASE + url_template.format(arxiv_id)
        if not rp.can_fetch(user_agent, url):
            print(f"{arxiv_id}: {kind} disallowed by robots.txt, skipping")
            _audit_fetch(arxiv_id, url, kind, None, False, delay, "robots_disallowed")
            continue

        response = _get_with_backoff(url, delay)
        if response.status_code == 200:
            if not _valid_payload(response.content, kind, response.headers.get("Content-Type", "")):
                _audit_fetch(arxiv_id, url, kind, 200, True, delay, "invalid_content")
                raise RuntimeError(f"arXiv returned invalid {kind} content for {url}")
            output = _cache_path(arxiv_id, kind)
            _atomic_cache_write(output, response.content, kind)
            _audit_fetch(arxiv_id, url, kind, 200, True, delay, "downloaded")
            print(f"{arxiv_id}: downloaded {kind} ({len(response.content)} bytes)")
            return output, kind

        _audit_fetch(arxiv_id, url, kind, response.status_code, True, delay, "unavailable")

    print(f"{arxiv_id}: no html or pdf available")
    return None, "missing"
