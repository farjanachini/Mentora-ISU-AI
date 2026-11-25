import concurrent.futures
import http.client
import re
from html.parser import HTMLParser
from urllib.error import URLError
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from unidecode import unidecode


BASE_DOMAIN = "isu.ac.bd"
START_URLS = (
    "https://isu.ac.bd/",
    "https://www.isu.ac.bd/about-us",
    "https://www.isu.ac.bd/vision-mission",
    "https://www.isu.ac.bd/message-from-chairman-bot",
    "https://www.isu.ac.bd/message-from-vice-chairman-bot",
    "https://www.isu.ac.bd/message-from-vice-chancellor",
    "https://isu.ac.bd/faculties/department-of-computer-science-and-engineering",
    "https://isu.ac.bd/faculties/department-of-ai-and-data-science",
    "https://www.isu.ac.bd/faculties/department-of-apparel-merchandising-and-management",
    "https://www.isu.ac.bd/faculties/department-of-business-administration",
    "https://www.isu.ac.bd/faculties/department-of-english",
    "https://www.isu.ac.bd/faculties/department-of-law",
    "https://www.isu.ac.bd/faculties/department-of-public-health",
    "https://www.isu.ac.bd/faculties/department-of-textile-engineering",
    "https://isu.ac.bd/faculties/all-departments",
    "https://convocation.isu.ac.bd/",
)
SKIP_TAGS = {"script", "style", "noscript", "svg", "template"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
BLOCK_TAGS = {
    "address", "article", "br", "dd", "div", "dt", "h1", "h2", "h3",
    "h4", "h5", "h6", "li", "p", "section", "tr",
}
ROUTE_HINTS = (
    "facult", "department", "admission", "notice", "news", "academic",
    "program", "tuition", "fee", "research", "contact", "convocation",
    "ceremony", "schedule", "event",
)
STOP_WORDS = {
    "about", "can", "could", "describe", "does", "for", "give", "hello",
    "how", "information", "is", "me", "of", "please", "profile", "tell",
    "the", "what", "when", "where", "which", "who", "why", "with", "dr",
    "md", "mr", "mrs", "ms", "prof",
}
MAX_PAGES = 14
MAX_PAGES_PER_ROUND = 6
MAX_RESPONSE_BYTES = 2_000_000
REQUEST_TIMEOUT = 6
MAX_CONTEXT_PAGES = 8
MAX_CONTEXT_CHARS = 5000


class WebsiteRetrievalError(RuntimeError):
    pass


class _PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text = []
        self.links = []
        self._skip_depth = 0
        self._anchor_href = None
        self._anchor_text = []
        self._current_faculty_section = None
        self._section_heading_tag = None
        self._section_heading_text = []

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return

        if self._skip_depth:
            return

        attributes = dict(attrs)
        classes = set(attributes.get("class", "").split())

        if tag in BLOCK_TAGS:
            self.text.append("\n")

        if tag == "h1":
            self.text.append("Page heading: ")

        if tag in HEADING_TAGS and "details-page-heading" in classes:
            self._section_heading_tag = tag
            self._section_heading_text = []

        if (
            tag == "div"
            and {"col-md-10", "card"}.issubset(classes)
            and self._current_faculty_section
        ):
            self.text.append(
                f"\nFaculty category: {self._current_faculty_section}\n"
            )

        if tag == "h4" and "name" in classes:
            self.text.append("Name: ")

        if tag == "a":
            href = attributes.get("href")
            if href:
                self._anchor_href = href
                self._anchor_text = []

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return

        if self._skip_depth:
            return

        if tag == self._section_heading_tag:
            section = " ".join(self._section_heading_text).split()
            if section:
                self._current_faculty_section = " ".join(section)
            self._section_heading_tag = None
            self._section_heading_text = []

        if tag == "a" and self._anchor_href:
            label = " ".join(" ".join(self._anchor_text).split())
            self.links.append((self._anchor_href, label))
            self._anchor_href = None
            self._anchor_text = []

        if tag in BLOCK_TAGS:
            self.text.append("\n")

    def handle_data(self, data):
        if self._skip_depth:
            return

        self.text.append(data + " ")
        if self._section_heading_tag:
            self._section_heading_text.append(data)
        if self._anchor_href:
            self._anchor_text.append(data)


def _plain_terms(text):
    transliterated = unidecode(text).lower()
    return {
        term for term in re.findall(r"[a-z0-9]+", transliterated)
        if len(term) > 1 and term not in STOP_WORDS
    }

def _terms(text):
    terms = _plain_terms(text)
    for term in tuple(terms):
        consonants = re.sub(r"[aeiou]", "", term)
        if len(consonants) >= 3:
            terms.add(consonants)

    return terms


def _normalize_url(url, base_url=None):
    absolute_url = urldefrag(urljoin(base_url or "", url))[0]
    parts = urlsplit(absolute_url)
    hostname = (parts.hostname or "").lower().rstrip(".")

    if (
        parts.scheme not in {"http", "https"}
        or (
            hostname != BASE_DOMAIN
            and not hostname.endswith(f".{BASE_DOMAIN}")
        )
    ):
        return None

    return urlunsplit((
        parts.scheme,
        parts.netloc.lower(),
        parts.path or "/",
        parts.query,
        "",
    ))


def _is_official_subdomain(url):
    hostname = (urlsplit(url).hostname or "").lower().rstrip(".")
    return (
        hostname not in {BASE_DOMAIN, f"www.{BASE_DOMAIN}"}
        and hostname.endswith(f".{BASE_DOMAIN}")
    )


def _is_subdomain_root(url):
    parts = urlsplit(url)
    return (
        _is_official_subdomain(url)
        and parts.path in {"", "/"}
        and not parts.query
    )


def _fetch_page(url):
    request = Request(
        url,
        headers={"User-Agent": "Mentora-ISU-Assistant/1.0"}
    )

    with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        final_url = _normalize_url(response.geturl())
        content_type = response.headers.get_content_type()

        if not final_url or content_type not in {"text/html", "application/xhtml+xml"}:
            return None

        raw_html = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw_html) > MAX_RESPONSE_BYTES:
            raw_html = raw_html[:MAX_RESPONSE_BYTES]

        charset = response.headers.get_content_charset() or "utf-8"
        html = raw_html.decode(charset, errors="replace")

    parser = _PageParser()
    parser.feed(html)

    lines = [
        " ".join(line.split())
        for line in "".join(parser.text).splitlines()
        if line.strip()
    ]

    return {
        "url": final_url,
        "text": "\n".join(lines),
        "links": parser.links,
    }


def _fetch_batch(urls):
    fetched_pages = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_fetch_page, url): url
            for url in urls
        }

        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                page = future.result()
            except (
                OSError, URLError, TimeoutError, ValueError,
                http.client.HTTPException, LookupError
            ) as error:
                print(f"Official ISU page fetch failed for {url}: {error}")
                continue

            if page:
                fetched_pages[url] = page

    return [
        fetched_pages[url]
        for url in urls
        if url in fetched_pages
    ]


def _link_score(url, label, query_terms):
    searchable_text = f"{label} {url}"
    score = 10 * len(query_terms.intersection(_terms(searchable_text)))
    lowered = searchable_text.lower()
    score += sum(1 for hint in ROUTE_HINTS if hint in lowered)
    return score


def _page_score(page, query_terms):
    score = len(query_terms.intersection(_terms(page["text"])))
    score += _link_score(page["url"], "", query_terms) / 10
    path = page["url"].lower()
    if "/faculties/department-" in path:
        score += 1
    return score


def _relevant_excerpt(text, query_terms):
    lines = text.splitlines()
    matching_indexes = [
        index for index, line in enumerate(lines)
        if query_terms.intersection(_terms(line))
    ]

    if not matching_indexes:
        excerpt = "\n".join(lines[:40])
        return excerpt[:MAX_CONTEXT_CHARS]

    selected_indexes = set()
    page_heading_index = next(
        (
            index for index, line in enumerate(lines)
            if line.startswith("Page heading: ")
        ),
        None
    )
    if page_heading_index is not None:
        selected_indexes.add(page_heading_index)

    for index in matching_indexes:
        selected_indexes.update(range(
            max(0, index - 10),
            min(len(lines), index + 10)
        ))
        category_index = next(
            (
                category_index
                for category_index in range(index - 1, -1, -1)
                if lines[category_index].startswith("Faculty category: ")
            ),
            None
        )
        if category_index is not None:
            selected_indexes.add(category_index)

    excerpt = "\n".join(
        lines[index] for index in sorted(selected_indexes)
    )
    return excerpt[:MAX_CONTEXT_CHARS]


def matching_faculty_categories(question, context):
    question_terms = _plain_terms(question)
    page_heading = ""
    category = ""
    matches = []
    seen = set()

    for line in context.splitlines():
        if line.startswith("Source URL:"):
            page_heading = ""
            category = ""
        elif line.startswith("Page heading: "):
            page_heading = line[len("Page heading: "):].strip()
        elif line.startswith("Faculty category: "):
            category = line[len("Faculty category: "):].strip()
        elif line.startswith("Name: "):
            name = line[len("Name: "):].strip()
            name_terms = _plain_terms(name)
            shared_terms = name_terms.intersection(question_terms)
            if not shared_terms or (
                len(shared_terms) < 2
                and not any(len(term) >= 8 for term in shared_terms)
            ):
                continue

            if not category or category.casefold() in {
                "faculty members", "all departments"
            }:
                continue

            department_match = re.search(
                r"\((Department of [^)]+)\)",
                page_heading,
                flags=re.IGNORECASE
            )
            department = (
                department_match.group(1)
                if department_match
                else ""
            )
            match = (name, category, department)
            if match not in seen:
                seen.add(match)
                matches.append({
                    "name": name,
                    "category": category,
                    "department": department,
                })

    return matches


def retrieve_documents(question):
    """Fetch relevant information live from the official ISU website."""
    query_terms = _terms(question)
    pages = _fetch_batch(START_URLS)

    if not pages:
        raise WebsiteRetrievalError(
            "Could not reach the official ISU websites."
        )

    visited = {page["url"] for page in pages}

    while len(visited) < MAX_PAGES:
        candidates = {}

        for page in pages:
            for href, label in page["links"]:
                url = _normalize_url(href, page["url"])
                if not url or url in visited:
                    continue

                score = _link_score(url, label, query_terms)
                if _is_official_subdomain(url):
                    score += 100
                if score > candidates.get(url, 0):
                    candidates[url] = score

        if not candidates:
            break

        ranked_urls = sorted(
            candidates,
            key=lambda url: (
                _is_subdomain_root(url),
                candidates[url],
            ),
            reverse=True
        )
        batch = ranked_urls[:min(
            MAX_PAGES_PER_ROUND,
            MAX_PAGES - len(visited)
        )]
        visited.update(batch)

        new_pages = _fetch_batch(batch)
        if not new_pages:
            break

        pages.extend(new_pages)

    ranked_pages = sorted(
        pages,
        key=lambda page: _page_score(page, query_terms),
        reverse=True
    )
    department_pages = [
        page for page in ranked_pages
        if "/faculties/department-" in page["url"].lower()
        and len(
            _plain_terms(question).intersection(_plain_terms(page["text"]))
        ) >= 2
    ]
    relevant_pages = [
        page for page in ranked_pages
        if _page_score(page, query_terms) > 0
    ]
    if department_pages:
        department_urls = {page["url"] for page in department_pages}
        relevant_pages = department_pages + [
            page for page in relevant_pages
            if page["url"] not in department_urls
        ]
    relevant_pages = relevant_pages[:MAX_CONTEXT_PAGES]

    return "\n\n".join(
        f"Source URL: {page['url']}\n"
        f"Information:\n{_relevant_excerpt(page['text'], query_terms)}"
        for page in relevant_pages
    )
