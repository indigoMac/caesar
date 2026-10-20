"""Collect public S&P MidCap 400 and SmallCap 600 change tables.

Walks the S&P Dow Jones Indices news list on PR Newswire and reads each
article that mentions those two indexes. Only the change table is stored.
Article prose is not written to disk.

The date printed in the table is the morning the change takes effect when
the release says "prior to the open". The file records the previous NYSE
close, which is the exit this study trades. A release that says "after the
close" keeps that session as the exit.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

from rebalance.calendar import TradingCalendar
from rebalance.errors import DataContractError

ROOT = Path(__file__).resolve().parents[1]
LIST_URL = "https://www.prnewswire.com/news/s%26p-dow-jones-indices/"
ORIGIN = "https://www.prnewswire.com"
USER_AGENT = "caesar-research/0.1 (personal index-announcement study)"
NEW_YORK = ZoneInfo("America/New_York")
OUTPUT = ROOT / "data" / "raw" / "announcements" / "sp_mid_small_announcements.csv"
REJECTS = ROOT / "data" / "raw" / "announcements" / "sp_mid_small_rejects.csv"
STATE = ROOT / "data" / "raw" / "announcements" / "sp_mid_small_fetch_state.json"

COLUMNS = (
    "published_at",
    "effective_close",
    "ticker",
    "company_name",
    "index_name",
    "action",
    "version",
    "rebalance_id",
    "source_url",
    "retrieved_at",
    "data_status",
)

_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_MONTH_PATTERN = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
_DATE_RE = re.compile(
    rf"\b({_MONTH_PATTERN})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b",
    re.IGNORECASE,
)
_DMY_RE = re.compile(rf"\b(\d{{1,2}})-({_MONTH_PATTERN})-(\d{{2,4}})\b", re.IGNORECASE)
_MONTH_DAY_RE = re.compile(rf"\b({_MONTH_PATTERN})\.?\s+(\d{{1,2}})\b", re.IGNORECASE)
_CLOCK_RE = re.compile(
    r"\b([A-Z][a-z]{2})\s+(\d{1,2}),\s+(\d{4}),\s+(\d{1,2}):(\d{2})\s+ET\b"
)
_PRIOR_OPEN_RE = re.compile(r"prior to the open(?:ing)?|before the open", re.IGNORECASE)
_AFTER_CLOSE_RE = re.compile(r"after the close", re.IGNORECASE)
_TICKER_MENTION_RE = re.compile(
    r"([A-Z][A-Za-z0-9&.'’\-, ]{1,100}?)\s*"
    r"\(\s*(?:NYSE|NASD|NASDAQ|AMEX|NYSE\s+ARCA|NYSE\s+MKT)\s*:\s*"
    r"([A-Za-z][A-Za-z0-9.\-]{0,8})\s*\)"
)
_TICKER_CELL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{0,7}$")
_LEGAL_SUFFIXES = {
    "inc",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "llc",
    "plc",
    "sa",
    "nv",
    "ag",
    "lp",
    "the",
}


@dataclass(frozen=True)
class ChangeRow:
    published_at: str
    effective_close: str
    ticker: str
    company_name: str
    index_name: str
    action: str
    rebalance_id: str
    source_url: str
    retrieved_at: str


class TableExtractor(HTMLParser):
    """Collect text rows from HTML tables. Script and style tags are ignored."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip += 1
            return
        if self._skip:
            return
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1
            return
        if self._skip:
            return
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if any(self._row):
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None

    def handle_data(self, data: str) -> None:
        if self._skip or self._cell is None:
            return
        self._cell.append(data)


def _fold_index(text: str) -> str | None:
    folded = re.sub(r"[^a-z0-9]+", " ", text.lower())
    if "smallcap 600" in folded or "small cap 600" in folded or re.search(r"\bs p 600\b", folded):
        return "SP600"
    if "midcap 400" in folded or "mid cap 400" in folded or re.search(r"\bs p 400\b", folded):
        return "SP400"
    return None


def _fold_action(text: str) -> str | None:
    folded = re.sub(r"[^a-z]", "", text.lower())
    if folded in {"addition", "add", "added"}:
        return "ADD"
    if folded in {"deletion", "delete", "deleted"}:
        return "DELETE"
    return None


def _full_year(value: int) -> int:
    if value >= 100:
        return value
    return 2000 + value if value < 70 else 1900 + value


def _parse_date(text: str) -> date | None:
    match = _DATE_RE.search(text)
    if match is not None:
        month = _MONTHS[match.group(1).lower().rstrip(".")]
        return date(int(match.group(3)), month, int(match.group(2)))
    match = _DMY_RE.search(text)
    if match is None:
        return None
    month = _MONTHS.get(match.group(2).lower().rstrip("."))
    if month is None:
        return None
    return date(_full_year(int(match.group(3))), month, int(match.group(1)))


def _all_dates(text: str) -> list[date]:
    found: list[date] = []
    for pattern in (_DATE_RE, _DMY_RE):
        for match in pattern.finditer(text):
            parsed = _parse_date(match.group(0))
            if parsed is not None:
                found.append(parsed)
    return found


def _dates_named_in(sentence: str, table_dates: set[date]) -> list[date]:
    named = [item for item in _all_dates(sentence) if item in table_dates]
    if named:
        return named
    for match in _MONTH_DAY_RE.finditer(sentence):
        month = _MONTHS[match.group(1).lower().rstrip(".")]
        day = int(match.group(2))
        named.extend(item for item in table_dates if item.month == month and item.day == day)
    return list(dict.fromkeys(named))


def _release_html(html: str) -> str:
    start = html.find('class="release-body')
    if start < 0:
        start = 0
    end = html.lower().find("about s&amp;p dow jones", start)
    if end < 0:
        end = html.lower().find("about s&p dow jones", start)
    if end < 0:
        end = len(html)
    return html[start:end]


def _visible_text(html: str) -> str:
    class _Text(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.parts: list[str] = []
            self._skip = 0

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag in {"script", "style"}:
                self._skip += 1

        def handle_endtag(self, tag: str) -> None:
            if tag in {"script", "style"} and self._skip:
                self._skip -= 1

        def handle_data(self, data: str) -> None:
            if not self._skip:
                self.parts.append(data)

    parser = _Text()
    parser.feed(html)
    return re.sub(r"\s+", " ", "".join(parser.parts))


def _published_at(html: str) -> datetime | None:
    match = re.search(r'id="prop26" value="([^"]+)"', html)
    source = match.group(1) if match else html
    clock = _CLOCK_RE.search(source)
    if clock is None:
        clock = _CLOCK_RE.search(html)
    if clock is None:
        return None
    month = _MONTHS[clock.group(1).lower()]
    local = datetime(
        int(clock.group(3)),
        month,
        int(clock.group(2)),
        int(clock.group(4)),
        int(clock.group(5)),
        tzinfo=NEW_YORK,
    )
    return local


def _normalize_name(name: str) -> str:
    folded = name.lower().replace("&", " and ")
    folded = re.sub(r"[^a-z0-9]+", " ", folded)
    tokens = [token for token in folded.split() if token not in _LEGAL_SUFFIXES]
    return " ".join(tokens)


def _ticker_index(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for match in _TICKER_MENTION_RE.finditer(text):
        key = _normalize_name(match.group(1))
        if key:
            found[key] = match.group(2).upper()
    return found


def _lookup_ticker(company: str, ticker_index: dict[str, str]) -> str | None:
    key = _normalize_name(company)
    if not key:
        return None
    if key in ticker_index:
        return ticker_index[key]
    candidates = [
        ticker
        for name, ticker in ticker_index.items()
        if name.startswith(key) or key.startswith(name)
    ]
    unique = set(candidates)
    if len(unique) == 1 and (len(key) >= 8 or len(key.split()) >= 2):
        return unique.pop()
    return None


def _timing_by_date(text: str, table_dates: set[date]) -> dict[date, str]:
    """Map each table date to prior_open or after_close. Conflicts are omitted."""
    assigned: dict[date, set[str]] = {item: set() for item in table_dates}
    sentences = re.split(r"(?<=\.)\s+", text)
    for sentence in sentences:
        rules: list[str] = []
        if _PRIOR_OPEN_RE.search(sentence):
            rules.append("prior_open")
        if _AFTER_CLOSE_RE.search(sentence):
            rules.append("after_close")
        if not rules:
            continue
        named = _dates_named_in(sentence, table_dates)
        targets = named or (list(table_dates) if "effective date" in sentence.lower() else [])
        for item in targets:
            assigned[item].update(rules)
    resolved: dict[date, str] = {}
    for item, rules in assigned.items():
        if len(rules) == 1:
            resolved[item] = next(iter(rules))
    if len(resolved) < len(table_dates):
        global_rules: set[str] = set()
        if _PRIOR_OPEN_RE.search(text):
            global_rules.add("prior_open")
        if _AFTER_CLOSE_RE.search(text):
            global_rules.add("after_close")
        if len(global_rules) == 1:
            only = next(iter(global_rules))
            for item in table_dates:
                resolved.setdefault(item, only)
    return resolved


def _exit_session(calendar: TradingCalendar, effective: date, rule: str) -> str:
    label = effective.isoformat()
    if rule == "after_close":
        return calendar.require_session(label).date().isoformat()
    if rule == "prior_open":
        return calendar.shift_session(label, -1).date().isoformat()
    raise DataContractError(f"unknown effective-date rule {rule}")


def _rows_from_tables(
    tables: list[list[list[str]]],
    ticker_index: dict[str, str],
) -> list[tuple[date, str, str, str, str]]:
    """Return (effective_date, index_name, action, company, ticker) before the exit adjustment."""
    parsed: list[tuple[date, str, str, str, str]] = []
    for table in tables:
        title_index: str | None = None
        title_date: date | None = None
        carried_index: str | None = None
        carried_date: date | None = None
        for cells in table:
            joined = " ".join(cells)
            header_index = _fold_index(joined)
            header_date = _parse_date(joined)
            if header_index and header_date and not any(_fold_action(cell) for cell in cells):
                title_index = header_index
                title_date = header_date
                continue
            if cells and _fold_action(cells[0]) and len(cells) >= 2:
                action = _fold_action(cells[0])
                company = cells[1]
                ticker = _lookup_ticker(company, ticker_index)
                if action and title_index and title_date and ticker:
                    parsed.append((title_date, title_index, action, company, ticker))
                continue
            if len(cells) < 5:
                continue
            parsed_date = _parse_date(cells[0])
            if parsed_date is not None:
                carried_date = parsed_date
            row_date = parsed_date or carried_date
            action = _fold_action(cells[2]) if len(cells) > 2 else None
            index_name = _fold_index(cells[1]) if len(cells) > 1 else None
            if index_name is not None:
                carried_index = index_name
            else:
                index_name = carried_index
            if row_date is None or action is None or index_name is None:
                continue
            company = cells[3].strip()
            ticker = cells[4].strip().upper() if len(cells) > 4 else ""
            if not _TICKER_CELL_RE.fullmatch(ticker):
                ticker = _lookup_ticker(company, ticker_index) or ""
            if not ticker or not company:
                continue
            parsed.append((row_date, index_name, action, company, ticker))
    return parsed


def parse_release(
    html: str,
    source_url: str,
    retrieved_at: str,
    calendar: TradingCalendar,
) -> tuple[list[ChangeRow], str | None]:
    """Parse one public release. The reason is set when the release yields no rows."""
    published = _published_at(html)
    if published is None:
        return [], "missing published timestamp"
    body_html = _release_html(html)
    extractor = TableExtractor()
    extractor.feed(body_html)
    text = _visible_text(body_html)
    extracted = _rows_from_tables(extractor.tables, _ticker_index(text))
    if not extracted:
        return [], "no SP400 or SP600 rows"
    rules = _timing_by_date(text, {item[0] for item in extracted})
    rows: list[ChangeRow] = []
    for effective, index_name, action, company, ticker in extracted:
        rule = rules.get(effective)
        if rule is None:
            return [], f"effective timing is ambiguous for {effective.isoformat()}"
        try:
            exit_session = _exit_session(calendar, effective, rule)
        except DataContractError as exc:
            return [], str(exc)
        rows.append(
            ChangeRow(
                published_at=published.isoformat(),
                effective_close=exit_session,
                ticker=ticker,
                company_name=company,
                index_name=index_name,
                action=action,
                rebalance_id=effective.isoformat(),
                source_url=source_url,
                retrieved_at=retrieved_at,
            )
        )
    return rows, None


class FetchError(RuntimeError):
    def __init__(self, url: str, status: int, detail: str) -> None:
        super().__init__(f"{status} for {url}: {detail}")
        self.status = status


def _fetch(url: str, timeout: float) -> str:
    completed = subprocess.run(
        [
            "curl",
            "--silent",
            "--show-error",
            "--location",
            "--fail-with-body",
            "--max-time",
            str(int(timeout)),
            "--user-agent",
            USER_AGENT,
            url,
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        status = 0
        match = re.search(r"\b(\d{3})\b", detail)
        if match:
            status = int(match.group(1))
        raise FetchError(url, status, detail or f"curl exit {completed.returncode}")
    return completed.stdout.decode("utf-8", "replace")


def _fetch_with_retry(url: str, timeout: float, attempts: int = 4) -> str:
    delay = 5.0
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return _fetch(url, timeout)
        except FetchError as exc:
            last_error = exc
            if exc.status not in {0, 429, 500, 502, 503, 504}:
                raise
        if attempt + 1 < attempts:
            time.sleep(delay)
            delay *= 2
    assert last_error is not None
    raise last_error


def _article_urls(list_html: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(
        r'<a class="newsreleaseconsolidatelink[^"]*" href="([^"]+)"[\s\S]*?</a>',
        list_html,
    ):
        href = match.group(1)
        if "/news-releases/" not in href or href.rstrip("/").endswith("news-releases"):
            continue
        text = re.sub(r"<[^>]+>", " ", match.group(0))
        if _fold_index(text) is None:
            continue
        absolute = href if href.startswith("http") else ORIGIN + href
        if absolute in seen:
            continue
        seen.add(absolute)
        urls.append(absolute)
    return urls


def _load_state() -> set[str]:
    if not STATE.exists():
        return set()
    payload = json.loads(STATE.read_text())
    return set(payload.get("fetched_urls", []))


def _save_state(fetched: set[str]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"fetched_urls": sorted(fetched)}, indent=2))


def _append_rows(rows: list[ChangeRow]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUTPUT.exists()
    with OUTPUT.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "published_at": row.published_at,
                    "effective_close": row.effective_close,
                    "ticker": row.ticker,
                    "company_name": row.company_name,
                    "index_name": row.index_name,
                    "action": row.action,
                    "version": 1,
                    "rebalance_id": row.rebalance_id,
                    "source_url": row.source_url,
                    "retrieved_at": row.retrieved_at,
                    "data_status": "PUBLIC_PRESS_RELEASE",
                }
            )


def _append_reject(source_url: str, reason: str) -> None:
    REJECTS.parent.mkdir(parents=True, exist_ok=True)
    new_file = not REJECTS.exists()
    with REJECTS.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source_url", "reason"))
        if new_file:
            writer.writeheader()
        writer.writerow({"source_url": source_url, "reason": reason})


def _assign_versions() -> int:
    if not OUTPUT.exists():
        return 0
    with OUTPUT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    deduped: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            row["source_url"],
            row["rebalance_id"],
            row["index_name"],
            row["ticker"],
            row["action"],
        )
        deduped[key] = row
    unique = list(deduped.values())
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in unique:
        groups.setdefault((row["rebalance_id"], row["index_name"], row["ticker"]), []).append(row)
    for group in groups.values():
        group.sort(key=lambda item: (item["published_at"], item["source_url"], item["action"]))
        for version, row in enumerate(group, start=1):
            row["version"] = str(version)
    unique.sort(
        key=lambda item: (
            item["rebalance_id"],
            item["index_name"],
            item["ticker"],
            item["published_at"],
            item["version"],
        )
    )
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(unique)
    return len(unique)


def collect(delay_seconds: float, max_pages: int, timeout: float) -> None:
    calendar = TradingCalendar("XNYS")
    retrieved_at = datetime.now(NEW_YORK).isoformat(timespec="seconds")
    fetched = _load_state()
    seen_pages: set[str] = set()
    article_urls: list[str] = []
    for page in range(1, max_pages + 1):
        separator = "&" if "?" in LIST_URL else "?"
        url = f"{LIST_URL}{separator}page={page}&pagesize=100"
        try:
            html = _fetch_with_retry(url, timeout)
        except FetchError as exc:
            if exc.status == 404:
                break
            raise
        found = _article_urls(html)
        new_urls = [item for item in found if item not in seen_pages]
        print(f"list page {page}: {len(found)} matching articles, {len(new_urls)} new", flush=True)
        if not new_urls:
            break
        seen_pages.update(new_urls)
        article_urls.extend(new_urls)
        time.sleep(delay_seconds)
    pending = [url for url in article_urls if url not in fetched]
    print(f"fetching {len(pending)} articles, {len(fetched)} already stored", flush=True)
    for number, url in enumerate(pending, start=1):
        try:
            html = _fetch_with_retry(url, timeout)
        except (FetchError, TimeoutError) as exc:
            _append_reject(url, f"fetch failed: {exc}")
            fetched.add(url)
            _save_state(fetched)
            continue
        rows, reason = parse_release(html, url, retrieved_at, calendar)
        if rows:
            _append_rows(rows)
        else:
            _append_reject(url, reason or "no rows")
        fetched.add(url)
        _save_state(fetched)
        if number % 25 == 0 or number == len(pending):
            print(f"parsed {number} of {len(pending)}", flush=True)
        time.sleep(delay_seconds)
    count = _assign_versions()
    print(f"wrote {count} rows to {OUTPUT}", flush=True)


def retry_rejects(delay_seconds: float, timeout: float) -> None:
    if not REJECTS.exists():
        print("no reject file", flush=True)
        return
    with REJECTS.open(newline="") as handle:
        pending = [row["source_url"] for row in csv.DictReader(handle)]
    calendar = TradingCalendar("XNYS")
    retrieved_at = datetime.now(NEW_YORK).isoformat(timespec="seconds")
    remaining: list[tuple[str, str]] = []
    print(f"retrying {len(pending)} articles", flush=True)
    for number, url in enumerate(pending, start=1):
        try:
            html = _fetch_with_retry(url, timeout)
        except (FetchError, TimeoutError) as exc:
            remaining.append((url, f"fetch failed: {exc}"))
            time.sleep(delay_seconds)
            continue
        rows, reason = parse_release(html, url, retrieved_at, calendar)
        if rows:
            _append_rows(rows)
        else:
            remaining.append((url, reason or "no rows"))
        if number % 25 == 0 or number == len(pending):
            print(f"retried {number} of {len(pending)}", flush=True)
        time.sleep(delay_seconds)
    with REJECTS.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source_url", "reason"))
        writer.writeheader()
        for url, reason in remaining:
            writer.writerow({"source_url": url, "reason": reason})
    count = _assign_versions()
    print(f"wrote {count} rows, {len(remaining)} articles still rejected", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retry-rejects", action="store_true")
    args = parser.parse_args()
    if args.retry_rejects:
        retry_rejects(args.delay_seconds, args.timeout)
    else:
        collect(args.delay_seconds, args.max_pages, args.timeout)


if __name__ == "__main__":
    main()
