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

# arXiv currently declares Crawl-delay: 15 for ordinary crawlers. The fallback is
# deliberately no lower, and one extra second provides a small scheduling margin.
MIN_CRAWL_DELAY = 16.0
CRAWL_DELAY_MARGIN = 0.0
BACKOFF_429 = 60.0
REQUEST_TIMEOUT = 30

_last_request_at = None


def _user_agent():
    """Build an identifiable user agent from the configured contact email.

    Returns
    -------
    str
        User agent identifying the student research project and its contact.

    Raises
    ------
    RuntimeError
        If ``ARXIV_CONTACT_EMAIL`` is not configured before network access.
    """

    email = os.environ.get("ARXIV_CONTACT_EMAIL", "").strip()
    if not email or "@" not in email:
        raise RuntimeError(
            "Set ARXIV_CONTACT_EMAIL to a real contact address before fetching arXiv"
        )
    return f"OTHAW-NLP-Project/1.0 (student research; contact: {email})"


def _wait_for_request_slot(delay):
    """Enforce one process-wide minimum interval between arXiv requests.

    Parameters
    ----------
    delay : float
        Required interval in seconds. This includes the safety margin.
    """

    global _last_request_at
    now = time.monotonic()
    if _last_request_at is not None:
        remaining = delay - (now - _last_request_at)
        if remaining > 0:
            time.sleep(remaining)
    _last_request_at = time.monotonic()


def _request(url, delay):
    """Make one rate-limited request to arXiv.

    Parameters
    ----------
    url : str
        Allowed arXiv URL to retrieve.
    delay : float
        Minimum interval since the previous arXiv request.

    Returns
    -------
    requests.Response
        Completed response. Status handling remains with the caller.

    Raises
    ------
    RuntimeError
        If the connection fails or times out. The run stops rather than issuing
        repeated requests during an uncertain server or network state.
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
    """Load arXiv robots rules with the same identity and limiter as paper requests.

    Returns
    -------
    urllib.robotparser.RobotFileParser
        Parsed policy carrying ``effective_delay`` and ``request_user_agent``
        attributes for later fetches.

    Raises
    ------
    RuntimeError
        If robots.txt cannot be retrieved or parsed safely. This is a fail-closed
        policy: no paper request is made without a confirmed allow decision.
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
    declared_delay = parser.crawl_delay(user_agent)
    if declared_delay is None:
        declared_delay = parser.crawl_delay("*")
    if declared_delay is None:
        declared_delay = MIN_CRAWL_DELAY

    parser.effective_delay = max(float(declared_delay), MIN_CRAWL_DELAY) + CRAWL_DELAY_MARGIN
    parser.request_user_agent = user_agent
    print(
        "arXiv robots.txt loaded; "
        f"effective request interval={parser.effective_delay:.0f}s"
    )
    return parser


def _cache_path(arxiv_id, kind):
    """Map an arXiv identifier to its HTML or PDF cache path.

    Parameters
    ----------
    arxiv_id : str
        Bare arXiv identifier.
    kind : str
        Either ``html`` or ``pdf``.

    Returns
    -------
    pathlib.Path
        Destination inside ``cache/``.
    """

    ext = "html" if kind == "html" else "pdf"
    return CACHE_DIR / f"{arxiv_id}.{ext}"


def _valid_payload(body, kind, content_type):
    """Check that a response resembles the requested HTML or PDF document.

    Parameters
    ----------
    body : bytes
        Response body.
    kind : str
        Expected source type.
    content_type : str
        HTTP Content-Type header.

    Returns
    -------
    bool
        True only for a matching media type and basic file signature.
    """

    media_type = content_type.split(";", 1)[0].strip().lower()
    if kind == "pdf":
        return media_type == "application/pdf" and body.startswith(b"%PDF-")
    prefix = body[:4096].lower()
    return media_type in {"text/html", "application/xhtml+xml"} and (
        b"<html" in prefix or b"<!doctype html" in prefix
    )


def _valid_cached_file(path, kind):
    """Validate a cached file by its basic HTML or PDF signature."""

    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as cached:
        prefix = cached.read(4096).lower()
    if kind == "pdf":
        return prefix.startswith(b"%pdf-")
    return b"<html" in prefix or b"<!doctype html" in prefix


def _atomic_cache_write(path, body, kind):
    """Validate first, then atomically replace the destination cache file."""

    CACHE_DIR.mkdir(exist_ok=True)
    mode = "wb"
    data = body if kind == "pdf" else body.decode("utf-8", "replace").encode("utf-8")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode=mode,
            dir=CACHE_DIR,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _audit_fetch(arxiv_id, url, kind, status, robots_allowed, delay, outcome):
    """Append one concise acquisition record for provenance and compliance checks."""

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
    with (CACHE_DIR / "fetch_audit.jsonl").open("a", encoding="utf-8") as audit_file:
        audit_file.write(json.dumps(record, sort_keys=True) + "\n")


def fetch_one(arxiv_id, rp):
    """Return one cached arXiv source, downloading HTML first when absent.

    Parameters
    ----------
    arxiv_id : str
        Bare identifier such as ``2403.05230``.
    rp : urllib.robotparser.RobotFileParser
        Policy returned by :func:`_robots`.

    Returns
    -------
    tuple[pathlib.Path | None, str]
        Cached path and ``html``/``pdf``, or ``(None, "missing")``.
    """

    CACHE_DIR.mkdir(exist_ok=True)

    for kind in ("html", "pdf"):
        path = _cache_path(arxiv_id, kind)
        if _valid_cached_file(path, kind):
            print(f"{arxiv_id}: cached ({kind}), skipping download")
            return path, kind
        if path.exists():
            print(f"{arxiv_id}: cached {kind} failed validation; requesting a clean copy")

    delay = getattr(rp, "effective_delay", MIN_CRAWL_DELAY + CRAWL_DELAY_MARGIN)
    user_agent = getattr(rp, "request_user_agent", None)
    if user_agent is None:
        user_agent = _user_agent()
    for kind, path_template in (("html", "/html/{}"), ("pdf", "/pdf/{}")):
        url = BASE + path_template.format(arxiv_id)
        allowed = rp.can_fetch(user_agent, url)
        if not allowed:
            print(f"{arxiv_id}: {kind} disallowed by robots.txt, skipping")
            _audit_fetch(arxiv_id, url, kind, None, False, delay, "robots_disallowed")
            continue

        response = _get_with_backoff(url, delay)
        if response.status_code == 200:
            if not _valid_payload(response.content, kind, response.headers.get("Content-Type", "")):
                _audit_fetch(
                    arxiv_id, url, kind, 200, True, delay, "invalid_content"
                )
                raise RuntimeError(f"arXiv returned invalid {kind} content for {url}")
            output = _cache_path(arxiv_id, kind)
            _atomic_cache_write(output, response.content, kind)
            _audit_fetch(arxiv_id, url, kind, 200, True, delay, "downloaded")
            print(f"{arxiv_id}: downloaded {kind} ({len(response.content)} bytes)")
            return output, kind

        _audit_fetch(
            arxiv_id,
            url,
            kind,
            response.status_code,
            True,
            delay,
            "unavailable",
        )

    print(f"{arxiv_id}: no html or pdf available")
    return None, "missing"


def _retry_after_seconds(response):
    """Convert an HTTP Retry-After header to seconds, when possible."""

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
    """GET once, with one server-directed retry after HTTP 429.

    Repeated 429 responses and blocking/service statuses stop the run. Continuing to
    the next paper in those conditions would increase the risk of an IP-level block.
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


def probe(arxiv_id, path, kind):
    """Print equation-structure counts for a cached HTML development sample."""

    if kind != "html":
        print(f"  {arxiv_id}: PDF-only, HTML probe not applicable")
        return
    from lxml import html as lxml_html

    tree = lxml_html.parse(str(path))
    numbers = tree.xpath('//span[contains(@class, "ltx_tag_equation")]')
    latex = tree.xpath('//annotation[@encoding="application/x-tex"]')
    print(f"  {arxiv_id}: numbered-eq spans={len(numbers)}, x-tex annotations={len(latex)}")
    if numbers:
        print(f"    sample number: {numbers[0].text_content().strip()!r}")
    if latex:
        print(f"    sample latex:  {latex[0].text_content().strip()[:70]!r}")


if __name__ == "__main__":
    ids = [line.strip().replace("arXiv:", "") for line in open("paper_list_29.txt")][:5]
    robots_policy = _robots()
    for paper_id in ids:
        source_path, source_kind = fetch_one(paper_id, robots_policy)
        if source_path:
            probe(paper_id, source_path, source_kind)
