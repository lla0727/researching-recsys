#!/usr/bin/env python3
"""
Multi-source Academic Paper Search Tool
Supports: arxiv, Semantic Scholar, Papers with Code, dblp, Google Scholar

Features:
- Parallel multi-source search
- Retry with exponential backoff
- SSL workaround for stability
- Rate limit handling
"""

import json
import time
import random
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, Any
from functools import wraps
import urllib.request
import urllib.parse
import urllib.error
import ssl
import sys

# Try to import optional dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    retriable_errors: tuple = (
        urllib.error.URLError,
        urllib.error.HTTPError,
        ConnectionResetError,
        ssl.SSLError,
        TimeoutError,
    )
):
    """Decorator to retry function with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retriable_errors as e:
                    last_exception = e

                    if attempt < max_retries:
                        # Calculate delay with jitter
                        delay = min(base_delay * (exponential_base ** attempt), max_delay)
                        jitter = random.uniform(0, 0.3 * delay)
                        total_delay = delay + jitter

                        print(f"  [Retry {attempt + 1}/{max_retries}] {func.__name__}: {e}. Waiting {total_delay:.1f}s...", file=sys.stderr)
                        time.sleep(total_delay)
                    else:
                        print(f"  [Failed] {func.__name__}: {e}", file=sys.stderr)

            return last_exception

        return wrapper
    return decorator


@dataclass
class Paper:
    """Unified paper representation across all sources"""
    title: str
    year: int
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    url: str = ""
    source: str = ""  # arxiv, semantic_scholar, papers_with_code, dblp, google_scholar
    venue: str = ""  # conference/journal name
    citations: int = 0
    code_url: str = ""  # for Papers with Code
    categories: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class MultiSourceSearcher:
    """Parallel multi-source academic paper searcher with retry logic"""

    def __init__(self, serper_api_key: Optional[str] = None):
        self.serper_api_key = serper_api_key
        self.results: list[Paper] = []
        self._session = None

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with fallback options"""
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        # Enable TLS 1.3 and fallback options for better compatibility
        ctx.minimum_version = ssl.TLSVersion.TLSv1
        return ctx

    def _get_request_headers(self) -> dict:
        """Return browser-like headers to avoid blocks"""
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, application/xml, text/html, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

    def _urlopen_with_retry(self, url: str, timeout: int = 60, max_retries: int = 3) -> Any:
        """URL open with retry and exponential backoff"""
        ctx = self._create_ssl_context()
        headers = self._get_request_headers()

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                req = urllib.request.Request(url, headers=headers)
                return urllib.request.urlopen(req, timeout=timeout, context=ctx)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = min(2 ** attempt + random.uniform(0, 1), 30)
                    print(f"    Retry {attempt + 1}/{max_retries} after {delay:.1f}s...", file=sys.stderr)
                    time.sleep(delay)

        raise last_error

    def search_all(self, query: str, year_range: tuple[int, int] = (2024, 2026),
                   max_per_source: int = 10) -> list[Paper]:
        """
        Search all sources in parallel and merge results.

        Args:
            query: Search query string
            year_range: (start_year, end_year) - defaults to 2024-2026
            max_per_source: Max results per source

        Returns:
            Merged and deduplicated list of Paper objects
        """
        import concurrent.futures

        start_year, end_year = year_range

        # Define search tasks for each source
        search_tasks = [
            ("arxiv", lambda: self._search_arxiv(query, start_year, end_year, max_per_source)),
            ("semantic_scholar", lambda: self._search_semantic_scholar(query, start_year, end_year, max_per_source)),
            ("papers_with_code", lambda: self._search_papers_with_code(query, start_year, end_year, max_per_source)),
            ("dblp", lambda: self._search_dblp(query, start_year, end_year, max_per_source)),
            ("google_scholar", lambda: self._search_google_scholar(query, max_per_source)),
        ]

        all_results = []

        # Execute all searches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_source = {executor.submit(task[1]): task[0] for task in search_tasks}

            for future in concurrent.futures.as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    results = future.result()
                    if results and isinstance(results, list):
                        all_results.extend(results)
                        print(f"[{source}] Found {len(results)} papers", file=sys.stderr)
                    elif results and isinstance(results, Exception):
                        print(f"[{source}] Failed after retries: {results}", file=sys.stderr)
                except Exception as e:
                    print(f"[{source}] Error: {e}", file=sys.stderr)

        # Deduplicate and sort
        merged = self._deduplicate(all_results)
        merged = self._sort_papers(merged)

        self.results = merged
        return merged

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def _search_arxiv(self, query: str, start_year: int, end_year: int,
                      max_results: int) -> list[Paper]:
        """Search arXiv API with retry"""
        papers = []

        # Build arXiv query
        arxiv_query = urllib.parse.quote(query.replace(" ", "+"))
        start_date = f"{start_year}0101"
        end_date = f"{end_year}1231"

        url = (f"http://export.arxiv.org/api/query?"
               f"search_query=all:{arxiv_query}+AND+submittedDate:[{start_date}+TO+{end_date}]"
               f"&start=0&max_results={max_results}"
               f"&sortBy=submittedDate&sortOrder=descending")

        try:
            ctx = self._create_ssl_context()
            headers = self._get_request_headers()
            req = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(req, timeout=60, context=ctx) as response:
                import xml.etree.ElementTree as ET
                content = response.read().decode('utf-8')
                root = ET.fromstring(content)

                ns = {'atom': 'http://www.w3.org/2005/Atom'}

                for entry in root.findall('atom:entry', ns):
                    title = entry.find('atom:title', ns)
                    summary = entry.find('atom:summary', ns)
                    published = entry.find('atom:published', ns)
                    link = entry.find('atom:id', ns)
                    author_elements = entry.findall('atom:author/atom:name', ns)

                    year = 0
                    if published is not None and published.text:
                        year = int(published.text[:4])

                    authors = [a.text for a in author_elements if a.text]

                    paper = Paper(
                        title=title.text.strip().replace('\n', ' ') if title is not None else "",
                        year=year,
                        authors=authors,
                        abstract=summary.text.strip().replace('\n', ' ') if summary is not None else "",
                        url=link.text if link is not None else "",
                        source="arxiv",
                        categories=[]
                    )
                    papers.append(paper)

        except Exception as e:
            print(f"  arXiv error: {e}", file=sys.stderr)
            raise  # Re-raise to trigger retry

        return papers

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def _search_semantic_scholar(self, query: str, start_year: int, end_year: int,
                                  max_results: int) -> list[Paper]:
        """Search Semantic Scholar Graph API with retry"""
        papers = []

        url = "https://api.semanticscholar.org/graph/v1/paper/search"

        params = {
            "query": query,
            "year": f"{start_year}-{end_year}",
            "limit": max_results,
            "fields": "title,authors,year,abstract,url,venue,citationCount,externalIds"
        }

        try:
            headers = self._get_request_headers()
            headers['Accept'] = 'application/json'

            req = urllib.request.Request(
                f"{url}?{urllib.parse.urlencode(params)}",
                headers=headers
            )

            ctx = self._create_ssl_context()

            with urllib.request.urlopen(req, timeout=60, context=ctx) as response:
                data = json.loads(response.read().decode('utf-8'))

                for item in data.get('data', []):
                    authors = [a.get('name', '') for a in item.get('authors', [])]

                    paper = Paper(
                        title=item.get('title', ''),
                        year=item.get('year', 0),
                        authors=authors,
                        abstract=item.get('abstract', ''),
                        url=item.get('url', ''),
                        source="semantic_scholar",
                        venue=item.get('venue', ''),
                        citations=item.get('citationCount', 0)
                    )
                    papers.append(paper)

        except urllib.error.HTTPError as e:
            if e.code == 429:  # Rate limited
                print(f"  Semantic Scholar rate limited, will retry...", file=sys.stderr)
                raise  # Trigger retry
            print(f"  Semantic Scholar HTTP error: {e}", file=sys.stderr)
            raise
        except Exception as e:
            print(f"  Semantic Scholar error: {e}", file=sys.stderr)
            raise

        return papers

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def _search_papers_with_code(self, query: str, start_year: int, end_year: int,
                                  max_results: int) -> list[Paper]:
        """Search Papers with Code API with retry"""
        papers = []

        # Papers with Code search endpoint
        search_url = f"https://paperswithcode.com/api/v1/papers/?q={urllib.parse.quote(query)}&page=1&itemsPerPage={max_results}"

        try:
            headers = self._get_request_headers()
            # Don't request gzip encoding to avoid decompression issues
            headers.pop('Accept-Encoding', None)
            req = urllib.request.Request(search_url, headers=headers)

            ctx = self._create_ssl_context()

            with urllib.request.urlopen(req, timeout=60, context=ctx) as response:
                raw_data = response.read()

                # Handle gzip compression if Content-Encoding is gzip
                content_encoding = response.headers.get('Content-Encoding', '').lower()
                if 'gzip' in content_encoding:
                    import gzip
                    raw_data = gzip.decompress(raw_data)

                data = json.loads(raw_data.decode('utf-8'))

                for item in data.get('results', []):
                    year = item.get('published', '')

                    if isinstance(year, str) and year:
                        try:
                            paper_year = int(year[:4])
                            if paper_year < start_year or paper_year > end_year:
                                continue
                        except ValueError:
                            pass
                    elif isinstance(year, int):
                        if year < start_year or year > end_year:
                            continue

                    paper = Paper(
                        title=item.get('title', ''),
                        year=year if isinstance(year, int) else int(str(year)[:4]) if year else 0,
                        authors=item.get('authors', []),
                        abstract=item.get('abstract', ''),
                        url=item.get('url', ''),
                        source="papers_with_code",
                        code_url=item.get('code_url', ''),
                        categories=item.get('tasks', [])
                    )
                    papers.append(paper)

        except Exception as e:
            print(f"  Papers with Code error: {e}", file=sys.stderr)
            raise

        return papers

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def _search_dblp(self, query: str, start_year: int, end_year: int,
                      max_results: int) -> list[Paper]:
        """Search DBLP API with retry"""
        papers = []

        base_url = "https://api.dblp.org/v1/search"

        params = {
            "q": query,
            "h": max_results,
            "f": 0,
            "format": "json"
        }

        try:
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            headers = self._get_request_headers()
            req = urllib.request.Request(url, headers=headers)

            ctx = self._create_ssl_context()

            with urllib.request.urlopen(req, timeout=60, context=ctx) as response:
                data = json.loads(response.read().decode('utf-8'))

                for hit in data.get('result', {}).get('hits', {}).get('hit', []):
                    info = hit.get('info', {})

                    year_str = info.get('year', '0')
                    try:
                        year = int(year_str)
                        if year < start_year or year > end_year:
                            continue
                    except (ValueError, TypeError):
                        year = 0

                    authors_data = info.get('authors', {})
                    if isinstance(authors_data, dict):
                        authors_list = authors_data.get('author', [])
                        if isinstance(authors_list, str):
                            authors_list = [authors_list]
                        authors = authors_list
                    elif isinstance(authors_data, list):
                        authors = authors_data
                    else:
                        authors = []

                    paper = Paper(
                        title=info.get('title', ''),
                        year=year,
                        authors=authors,
                        abstract="",  # DBLP doesn't provide abstracts
                        url=info.get('url', ''),
                        source="dblp",
                        venue=info.get('venue', '')
                    )
                    papers.append(paper)

        except Exception as e:
            print(f"  DBLP error: {e}", file=sys.stderr)
            raise

        return papers

    def _search_google_scholar(self, query: str, max_results: int) -> list[Paper]:
        """Search Google Scholar via Serper.dev API"""
        papers = []

        if self.serper_api_key:
            return self._search_google_scholar_serper(query, max_results)
        else:
            print("Google Scholar: Skipped (no --serper-key provided)", file=sys.stderr)
            return papers

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def _search_google_scholar_serper(self, query: str, max_results: int) -> list[Paper]:
        """Search Google Scholar via Serper.dev API with retry"""
        papers = []

        url = "https://google.serper.dev/scholar"

        payload = {
            "q": query,
            "num": min(max_results, 20)
        }

        headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers=headers,
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode('utf-8'))

                for item in data.get('organic', []):
                    year_str = item.get('year', '')
                    year = int(year_str) if year_str.isdigit() else 0

                    paper = Paper(
                        title=item.get('title', ''),
                        year=year,
                        authors=[],
                        abstract=item.get('snippet', ''),
                        url=item.get('link', ''),
                        source="google_scholar",
                        citations=item.get('citations', 0)
                    )
                    papers.append(paper)

        except Exception as e:
            print(f"  Google Scholar error: {e}", file=sys.stderr)
            raise

        return papers

    def _deduplicate(self, papers: list[Paper]) -> list[Paper]:
        """Remove duplicate papers based on title similarity"""
        seen_titles = set()
        unique_papers = []

        for paper in papers:
            normalized = paper.title.lower().strip()

            if normalized not in seen_titles:
                seen_titles.add(normalized)
                unique_papers.append(paper)

        return unique_papers

    def _sort_papers(self, papers: list[Paper]) -> list[Paper]:
        """Sort papers by: source priority (quality indicator), year, then citations"""
        source_priority = {
            'semantic_scholar': 5,
            'dblp': 4,
            'arxiv': 3,
            'google_scholar': 2,
            'papers_with_code': 1
        }

        def sort_key(paper: Paper):
            priority = source_priority.get(paper.source, 0)
            return (priority, paper.year, paper.citations)

        return sorted(papers, key=sort_key, reverse=True)

    def export_json(self, filepath: str):
        """Export results to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([p.to_dict() for p in self.results], f, ensure_ascii=False, indent=2)

    def export_markdown(self, filepath: str, query: str):
        """Export results to Markdown report"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Paper Search Results: {query}\n\n")
            f.write(f"**Total papers found: {len(self.results)}**\n\n")

            current_source = None
            for paper in self.results:
                if paper.source != current_source:
                    f.write(f"\n## Source: {paper.source.replace('_', ' ').title()}\n\n")
                    current_source = paper.source

                f.write(f"### {paper.title} ({paper.year})\n")
                if paper.authors:
                    f.write(f"- **Authors**: {', '.join(paper.authors[:5])}")
                    if len(paper.authors) > 5:
                        f.write(f" et al.")
                    f.write("\n")
                if paper.venue:
                    f.write(f"- **Venue**: {paper.venue}\n")
                if paper.citations > 0:
                    f.write(f"- **Citations**: {paper.citations}\n")
                if paper.abstract:
                    abstract_preview = paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract
                    f.write(f"- **Abstract**: {abstract_preview}\n")
                f.write(f"- **URL**: {paper.url}\n")
                if paper.code_url:
                    f.write(f"- **Code**: {paper.code_url}\n")
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(description='Multi-source Academic Paper Search (with retry)')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--start-year', type=int, default=2024, help='Start year (default: 2024)')
    parser.add_argument('--end-year', type=int, default=2026, help='End year (default: 2026)')
    parser.add_argument('--max-per-source', type=int, default=10, help='Max results per source (default: 10)')
    parser.add_argument('--serper-key', type=str, default=None, help='Serper.dev API key for Google Scholar')
    parser.add_argument('--output', '-o', type=str, help='Output file (JSON or Markdown)')
    parser.add_argument('--format', choices=['json', 'markdown'], default='json', help='Output format')

    args = parser.parse_args()

    print(f"Searching for: {args.query}", file=sys.stderr)
    print(f"Year range: {args.start_year}-{args.end_year}", file=sys.stderr)
    print(f"Max per source: {args.max_per_source}", file=sys.stderr)
    print("-" * 50, file=sys.stderr)

    searcher = MultiSourceSearcher(serper_api_key=args.serper_key)

    start_time = time.time()

    results = searcher.search_all(
        args.query,
        year_range=(args.start_year, args.end_year),
        max_per_source=args.max_per_source
    )

    elapsed = time.time() - start_time

    print("-" * 50, file=sys.stderr)
    print(f"Total unique papers: {len(results)}", file=sys.stderr)
    print(f"Time elapsed: {elapsed:.1f}s", file=sys.stderr)

    if args.output:
        if args.format == 'json':
            searcher.export_json(args.output)
        else:
            searcher.export_markdown(args.output, args.query)
        print(f"Results saved to: {args.output}", file=sys.stderr)
    else:
        print(json.dumps([p.to_dict() for p in results], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
