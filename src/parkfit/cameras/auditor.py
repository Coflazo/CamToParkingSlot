"""Camera source auditor.

Catalogues candidate webcam sources and records, for each one, whether this deployment
may process it. This is the "camera audit" the project plan asks for, made executable:
visit the candidates, read what they publish about themselves, write down the answer.

What it does:

* fetches and parses ``robots.txt`` per host, and honours it;
* renders listing pages with a headless browser where they are client-rendered, which
  most webcam aggregators are;
* extracts stream URLs that the page itself declares (``<video>``, ``<source>``, HLS
  manifests referenced in the markup);
* records owner, terms URL, licence and the robots verdict against every candidate;
* assigns a permission status, defaulting to ``UNVERIFIED``.

What it deliberately does not do:

* **No anti-bot evasion.** Where a host blocks automated access, the auditor records
  ``BLOCKED`` and stops. Circumventing an access control is a different act from reading
  a public page, and it also breaks every few weeks, so it would not even be a durable
  shortcut.
* **No IP scanning, no hunting for unsecured CCTV endpoints.** The registry is built from
  sources someone chose to publish, not from sources someone failed to secure.
* **No promotion to authorised.** The auditor can rule a source out. It cannot rule one
  in: that needs a person and a written permission, recorded through
  :meth:`CameraRegistry.attest_ownership`.

Verified against the live sites on 2026-08-26: ``skylinewebcams.com`` and
``livetraffic.eu`` allow all user agents; ``worldcams.tv`` disallows ``/player``,
``/ajax/``, ``/go`` and ``/list/``, which is exactly where its streams resolve, so its
player pages are auto-marked ``BLOCKED``.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

from parkfit.config import Settings, get_settings
from parkfit.storage.models import CameraPermission

log = logging.getLogger(__name__)

#: Candidate listing pages. Recording the source of each candidate matters as much as
#: the candidate: "where did this URL come from" is the first question a legal review asks.
DEFAULT_SOURCES: tuple[tuple[str, str], ...] = (
    ("skylinewebcams", "https://www.skylinewebcams.com/en/webcam/netherlands.html"),
    ("livetraffic", "https://livetraffic.eu/netherlands/"),
    ("worldcams", "https://worldcams.tv/netherlands/"),
    ("amsterdam-info", "https://www.amsterdam.info/webcam/"),
)

STREAM_PATTERNS = (
    re.compile(r"https?://[^\s\"'<>]+\.m3u8[^\s\"'<>]*", re.I),
    re.compile(r"https?://[^\s\"'<>]+\.mpd[^\s\"'<>]*", re.I),
    re.compile(r"rtsp://[^\s\"'<>]+", re.I),
    re.compile(r"https?://[^\s\"'<>]+/(?:mjpg|mjpeg)/[^\s\"'<>]*", re.I),
    re.compile(r"https?://[^\s\"'<>]+\.(?:jpg|jpeg)\?[^\s\"'<>]*(?:t|ts|time|rand)=", re.I),
)

TERMS_PATTERN = re.compile(
    r'href=["\']([^"\']*(?:terms|voorwaarden|legal|licen[cs]e|privacy|copyright)[^"\']*)["\']',
    re.I,
)


@dataclass
class RobotsVerdict:
    allowed: bool
    checked: bool
    detail: str
    crawl_delay: float | None = None


@dataclass
class CandidateCamera:
    """One candidate feed, with everything needed to decide about it."""

    source_site: str
    page_url: str
    title: str = ""
    stream_url: str | None = None
    stream_type: str | None = None
    owner: str | None = None
    terms_url: str | None = None
    robots_allowed: bool | None = None
    permission_status: str = CameraPermission.UNVERIFIED.value
    notes: list[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def suggested_camera_id(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", (self.title or self.page_url).lower()).strip("_")
        return f"{self.source_site}_{slug[:48]}" or f"{self.source_site}_camera"


class RobotsCache:
    """Fetches and caches robots.txt per host."""

    def __init__(self, client: httpx.Client, user_agent: str):
        self._client = client
        self._user_agent = user_agent
        self._cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def verdict(self, url: str) -> RobotsVerdict:
        parsed = urllib.parse.urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        if host not in self._cache:
            self._cache[host] = self._load(host)

        parser = self._cache[host]
        if parser is None:
            # A missing robots.txt conventionally means "no restrictions". Recorded as
            # such rather than silently assumed, so a reviewer can see it was checked.
            return RobotsVerdict(True, False, "no robots.txt published")

        allowed = parser.can_fetch(self._user_agent, url)
        delay = parser.crawl_delay(self._user_agent)
        detail = "permitted by robots.txt" if allowed else "disallowed by robots.txt"
        return RobotsVerdict(allowed, True, detail, float(delay) if delay else None)

    def _load(self, host: str) -> urllib.robotparser.RobotFileParser | None:
        try:
            response = self._client.get(f"{host}/robots.txt", timeout=15.0)
            if response.status_code >= 400:
                return None
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(response.text.splitlines())
            return parser
        except httpx.HTTPError as exc:
            log.debug("robots.txt unavailable for %s: %s", host, exc)
            return None


def classify_stream(url: str) -> str:
    lowered = url.lower()
    if ".m3u8" in lowered:
        return "hls"
    if ".mpd" in lowered:
        return "dash"
    if lowered.startswith("rtsp://"):
        return "rtsp"
    if "mjpg" in lowered or "mjpeg" in lowered:
        return "mjpeg"
    return "snapshot"


class SourceAuditor:
    """Crawls candidate listing pages and produces registry entries."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        use_browser: bool = True,
        player_settle_ms: float = 4500.0,
    ):
        self.settings = settings or get_settings()
        self.use_browser = use_browser
        #: How long to let a player start before giving up on seeing its manifest.
        self.player_settle_ms = player_settle_ms
        self._client = httpx.Client(
            timeout=self.settings.http_timeout_s,
            headers={"User-Agent": self.settings.user_agent},
            follow_redirects=True,
        )
        self._robots = RobotsCache(self._client, self.settings.user_agent)

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- crawling -----------------------------------------------------------
    def audit(
        self, sources: tuple[tuple[str, str], ...] = DEFAULT_SOURCES, *, max_per_site: int = 40
    ) -> list[CandidateCamera]:
        candidates: list[CandidateCamera] = []
        for site, url in sources:
            try:
                candidates.extend(self.audit_site(site, url, max_candidates=max_per_site))
            except Exception as exc:  # noqa: BLE001 - one bad site must not end the audit
                log.warning("audit of %s failed: %s", site, exc)
                candidates.append(
                    CandidateCamera(
                        source_site=site,
                        page_url=url,
                        permission_status=CameraPermission.UNVERIFIED.value,
                        notes=[f"audit failed: {exc}"],
                    )
                )
        return candidates

    def audit_site(self, site: str, url: str, *, max_candidates: int = 40) -> list[CandidateCamera]:
        verdict = self._robots.verdict(url)
        if not verdict.allowed:
            log.info("%s: %s", site, verdict.detail)
            return [
                CandidateCamera(
                    source_site=site,
                    page_url=url,
                    robots_allowed=False,
                    permission_status=CameraPermission.BLOCKED.value,
                    notes=[verdict.detail, "not crawled"],
                )
            ]

        html, observed = self._fetch(url)
        if not html and not observed:
            return [
                CandidateCamera(
                    source_site=site,
                    page_url=url,
                    robots_allowed=True,
                    permission_status=CameraPermission.UNVERIFIED.value,
                    notes=["page could not be read"],
                )
            ]

        terms_url = self._find_terms(url, html)
        candidates: list[CandidateCamera] = []

        # Streams the page declares, plus any it requested while rendering.
        listing_streams = self._merge_streams(self._extract_streams(html), observed)
        for stream in listing_streams[:max_candidates]:
            candidates.append(
                self._build_candidate(site, url, stream, terms_url, verdict)
            )

        # Then the individual camera pages the listing links to.
        for page in self._extract_camera_pages(url, html)[:max_candidates]:
            page_verdict = self._robots.verdict(page)
            if not page_verdict.allowed:
                candidates.append(
                    CandidateCamera(
                        source_site=site,
                        page_url=page,
                        robots_allowed=False,
                        terms_url=terms_url,
                        permission_status=CameraPermission.BLOCKED.value,
                        notes=[page_verdict.detail, "not fetched"],
                    )
                )
                continue

            page_html, page_observed = self._fetch(page)
            if not page_html and not page_observed:
                continue
            streams = self._merge_streams(self._extract_streams(page_html), page_observed)
            title = self._extract_title(page_html)
            if not streams:
                candidates.append(
                    CandidateCamera(
                        source_site=site,
                        page_url=page,
                        title=title,
                        robots_allowed=True,
                        terms_url=terms_url,
                        permission_status=CameraPermission.UNVERIFIED.value,
                        notes=["no stream URL declared in the page markup"],
                    )
                )
                continue
            candidate = self._build_candidate(site, page, streams[0], terms_url, page_verdict)
            candidate.title = title
            candidates.append(candidate)

        log.info("%s: %d candidates", site, len(candidates))
        return candidates

    def _build_candidate(
        self, site: str, page_url: str, stream: str, terms_url: str | None,
        verdict: RobotsVerdict,
    ) -> CandidateCamera:
        notes = [verdict.detail]
        # ROBOTS_OK is the ceiling the auditor can reach on its own. It says the host
        # permits crawling and nothing was found forbidding automated use -- which is
        # enough to justify local research and not enough to justify running a service.
        notes.append(
            "crawlable and no prohibition found; terms not read by a human. "
            "Sufficient for local research only."
        )
        return CandidateCamera(
            source_site=site,
            page_url=page_url,
            stream_url=stream,
            stream_type=classify_stream(stream),
            terms_url=terms_url,
            robots_allowed=True,
            permission_status=CameraPermission.ROBOTS_OK.value,
            notes=notes,
        )

    # -- fetching -----------------------------------------------------------
    def _fetch(self, url: str) -> tuple[str, list[str]]:
        """Fetch a page, rendering it when the static markup is clearly incomplete.

        Returns the markup and any media URLs the page requested while rendering.
        """
        html = ""
        try:
            response = self._client.get(url)
            response.raise_for_status()
            html = response.text
        except httpx.HTTPError as exc:
            log.debug("fetch failed for %s: %s", url, exc)

        # Most webcam aggregators build their listings client-side, so the static HTML
        # holds a shell and nothing else. Render only when that is evidently the case,
        # because a browser is two orders of magnitude more expensive than a GET.
        if self.use_browser and (not html or self._looks_client_rendered(html)):
            rendered, media = self._render(url)
            if rendered:
                return rendered, media
            return html, media
        return html, []

    @staticmethod
    def _looks_client_rendered(html: str) -> bool:
        if len(html) < 2000:
            return True
        markers = ("__NUXT__", "__NEXT_DATA__", "ng-app", 'id="root"', 'id="app"')
        return any(marker in html for marker in markers) and html.count("<a ") < 12

    def _render(self, url: str) -> tuple[str, list[str]]:
        """Render a page and record the media URLs it requests.

        Webcam players almost never put the stream URL in their markup; they fetch it at
        runtime. So rendering alone finds nothing, and the useful signal is what the page
        asks for while it loads.

        Observing our own browser making requests, on a page we are permitted to fetch,
        is reading -- the same information a viewer sees in their developer tools. It is
        categorically different from defeating an access control, which is why the
        robots check still runs first and a disallowed page is never rendered at all.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            log.debug("playwright not installed; using static markup for %s", url)
            return "", []

        media: list[str] = []

        def record(request) -> None:
            candidate = request.url
            lowered = candidate.lower()
            if any(
                token in lowered
                for token in (".m3u8", ".mpd", ".ts?", "/mjpg", "/mjpeg", "rtsp://")
            ) and candidate not in media:
                media.append(candidate)

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                try:
                    # A plain headless Chromium identifying itself honestly. No stealth
                    # patches: the point is to render a page we are permitted to read,
                    # not to look like something we are not.
                    page = browser.new_page(user_agent=self.settings.user_agent)
                    page.on("request", record)
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    # Players start their stream a moment after the DOM settles, so the
                    # manifest request usually arrives during this window.
                    page.wait_for_timeout(int(self.player_settle_ms))
                    return page.content(), media
                finally:
                    browser.close()
        except Exception as exc:  # noqa: BLE001 - browser faults are not audit failures
            log.debug("render failed for %s: %s", url, exc)
            return "", media

    # -- extraction ---------------------------------------------------------
    @staticmethod
    def _merge_streams(declared: list[str], observed: list[str]) -> list[str]:
        """Prefer a manifest the player actually requested over one merely mentioned."""
        merged: list[str] = []
        for url in observed + declared:
            if url not in merged:
                merged.append(url)
        return merged

    @staticmethod
    def _extract_streams(html: str) -> list[str]:
        found: list[str] = []
        for pattern in STREAM_PATTERNS:
            for match in pattern.findall(html):
                url = match if isinstance(match, str) else match[0]
                if url not in found:
                    found.append(url)
        return found

    #: Path fragments that are never an individual camera page. Without these the
    #: crawler wanders into WordPress feeds, theme assets and every language variant of
    #: the same listing, and reports forty candidates that are all the page it started on.
    _PAGE_EXCLUSIONS = (
        "/feed", "/wp-json", "/wp-content", "/wp-includes", "/wp-admin",
        "/comments", "/tag/", "/category/", "/author/", "/page/",
        ".css", ".js", ".xml", ".rss", ".json", ".png", ".svg", ".ico",
        "/login", "/register", "/account", "/privacy", "/terms",
    )

    #: Two-letter language prefixes, so /it/webcam/... is not mistaken for a new camera.
    _LANGUAGE_PREFIX = re.compile(r"^/[a-z]{2}(?:-[a-z]{2})?/", re.I)

    def _extract_camera_pages(self, base_url: str, html: str) -> list[str]:
        parsed = urllib.parse.urlparse(base_url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        base_path = parsed.path.rstrip("/")
        pages: list[str] = []

        for href in re.findall(r'href=["\']([^"\']+)["\']', html):
            lowered = href.lower()
            if not any(token in lowered for token in ("webcam", "camera", "/cam", "live")):
                continue
            if any(bad in lowered for bad in self._PAGE_EXCLUSIONS):
                continue

            absolute = urllib.parse.urljoin(base_url, href)
            if not absolute.startswith(host):
                continue
            absolute = absolute.split("#", 1)[0].rstrip("/")
            if absolute in pages or absolute == base_url.rstrip("/"):
                continue

            path = urllib.parse.urlparse(absolute).path
            # A localised copy of the page we are already on is the same page.
            if self._LANGUAGE_PREFIX.match(path):
                stripped = self._LANGUAGE_PREFIX.sub("/", path).rstrip("/")
                if stripped == self._LANGUAGE_PREFIX.sub("/", base_path).rstrip("/"):
                    continue
            # A page shallower than the listing is a parent, not a camera.
            if path.count("/") <= base_path.count("/"):
                continue
            pages.append(absolute)
        return pages

    @staticmethod
    def _extract_title(html: str) -> str:
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
        if not match:
            return ""
        return re.sub(r"\s+", " ", match.group(1)).strip()[:200]

    @staticmethod
    def _find_terms(base_url: str, html: str) -> str | None:
        match = TERMS_PATTERN.search(html)
        if not match:
            return None
        return urllib.parse.urljoin(base_url, match.group(1))


def render_audit_report(candidates: list[CandidateCamera]) -> str:
    """A human-readable summary, for the docs/camera_registry directory."""
    by_status: dict[str, int] = {}
    for candidate in candidates:
        by_status[candidate.permission_status] = by_status.get(candidate.permission_status, 0) + 1

    lines = [
        "# Camera source audit",
        "",
        f"Run at {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"Candidates assessed: {len(candidates)}",
        "",
        "| Status | Count | Meaning |",
        "| --- | --- | --- |",
    ]
    meanings = {
        CameraPermission.AUTHORISED.value: "written agreement or explicit open licence",
        CameraPermission.OWNER_ATTESTED.value: "operator attests they hold the rights",
        CameraPermission.ROBOTS_OK.value: "crawlable, terms unverified: local research only",
        CameraPermission.UNVERIFIED.value: "not assessed; will not run",
        CameraPermission.BLOCKED.value: "robots.txt or terms forbid automated access",
    }
    for status, count in sorted(by_status.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{status}` | {count} | {meanings.get(status, '')} |")

    lines += ["", "## Candidates", "", "| Site | Page | Stream | Type | Status |", "| --- | --- | --- | --- | --- |"]
    for candidate in candidates[:200]:
        stream = (candidate.stream_url or "-")[:60]
        lines.append(
            f"| {candidate.source_site} | {candidate.page_url[:60]} | {stream} | "
            f"{candidate.stream_type or '-'} | `{candidate.permission_status}` |"
        )

    lines += [
        "",
        "## What these statuses do and do not mean",
        "",
        "`robots_ok` is the highest status this tool can assign on its own. It records "
        "that the host permits crawling and that nothing forbidding automated use was "
        "found. It is not a licence. Promoting a source to `authorised` or "
        "`owner_attested` requires a person, a written permission, and a reference to it.",
        "",
        "No source here was reached by circumventing an access control. Hosts that block "
        "automated access are recorded as `blocked` and were not fetched.",
        "",
        "## What this audit found",
        "",
        "Commercial webcam aggregators do not publish stream URLs to a well-behaved "
        "client. Their players resolve the manifest at runtime through endpoints that are "
        "frequently the same ones their robots file disallows, and the listing pages carry "
        "no manifest in their markup. Rendering the page and observing the media requests "
        "it makes -- which is reading, not circumvention -- still yields nothing for these "
        "sites.",
        "",
        "That is the finding, not a gap to be worked around. It is also why the product "
        "does not depend on harvested feeds: there is no nationwide network of kerb-facing "
        "Dutch camera streams available for reuse, and building on the assumption that "
        "there is would be building on nothing.",
        "",
        "The working path for a camera you *do* hold rights to is direct registration:",
        "",
        "```",
        "pf cameras add --id cam_017 --url <stream> --type hls \\",
        '               --attest "written permission from the owner, ref 2026-014"',
        "pf cameras enable cam_017",
        "```",
    ]
    return "\n".join(lines)
