#!/usr/bin/env python3
"""
fetch_arxiv_papers.py
Zero-dependency utility to query the arXiv API for drone detection, counter-UAS,
and anti-UAV research, parse Atom XML feeds, and optionally merge the results
into papers_database.json.

Fetched entries use the same schema as curated ones, with "verified": false.
They are candidates, not citations: tools/build_bibliography.py skips them until
someone checks them and sets "verified": true.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_API_BASE = "https://export.arxiv.org/api/query"
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers_database.json")
USER_AGENT = "drone-detection-compendium/1.2 (tools/fetch_arxiv_papers.py)"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
ARXIV_ID_RE = re.compile(r"abs/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/[0-9]{7})(v[0-9]+)?$")
# arXiv's CDN throttles with 406 as well as 429/503; all of these clear up after a pause.
RETRYABLE_HTTP = {406, 429, 500, 502, 503, 504}


def base_arxiv_id(raw: str) -> str:
    """'2609.01742v1' -> '2609.01742'. Older database entries stored the version suffix."""
    return re.sub(r"v[0-9]+$", "", raw or "")


def query_arxiv(search_query: str, max_results: int = 10, start: int = 0,
                sort_by: str = "submittedDate", sort_order: str = "descending", attempts: int = 4):
    """Query the arXiv API and return a list of paper dictionaries (empty on failure)."""
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    url = f"{ARXIV_API_BASE}?{urllib.parse.urlencode(params)}"
    print(f"[*] Querying arXiv API: {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                xml_data = response.read()
            # The API sometimes answers a rate limit with HTTP 200 and a plain-text body.
            if xml_data.lstrip().startswith(b"<"):
                return parse_arxiv_atom(xml_data)
            reason = xml_data[:80].decode("utf-8", "replace").strip()
        except urllib.error.HTTPError as e:
            if e.code not in RETRYABLE_HTTP:
                print(f"[!] arXiv API returned HTTP {e.code}", file=sys.stderr)
                return []
            reason = f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError) as e:
            reason = str(e)

        if attempt == attempts:
            break
        wait = 5 * 2 ** (attempt - 1)  # 5, 10, 20 s: arXiv asks for >= 3 s between requests
        print(f"[!] arXiv unavailable ({reason}); retry {attempt}/{attempts - 1} in {wait}s", file=sys.stderr)
        time.sleep(wait)

    print(f"[!] Giving up on the arXiv API after {attempts} attempts ({reason}).", file=sys.stderr)
    return []


def _text(entry, path):
    elem = entry.find(path, NS)
    return " ".join(elem.text.split()) if elem is not None and elem.text else None


def parse_arxiv_atom(xml_bytes: bytes):
    """Parse Atom XML returned by arXiv into entries matching papers_database.json's schema."""
    root = ET.fromstring(xml_bytes)
    papers = []

    for entry in root.findall("atom:entry", NS):
        raw_id = _text(entry, "atom:id") or ""
        # arXiv reports query errors as a feed entry whose id points at /api/errors.
        if "/api/errors" in raw_id:
            raise ValueError(f"arXiv API error: {_text(entry, 'atom:summary')}")

        m = ARXIV_ID_RE.search(raw_id)
        if not m:
            print(f"[!] Skipping entry with unrecognised id: {raw_id}", file=sys.stderr)
            continue
        arxiv_id, version = m.group(1), m.group(2)

        published = _text(entry, "atom:published")
        year = int(published[:4]) if published else None
        authors = [n.text.strip() for n in entry.findall("atom:author/atom:name", NS) if n.text]
        primary_cat = entry.find("arxiv:primary_category", NS)
        category = primary_cat.attrib.get("term") if primary_cat is not None else None
        first_author = re.sub(r"[^a-zA-Z0-9]", "", authors[0].split()[-1].lower()) if authors else "anon"

        papers.append({
            "id": f"{first_author}{year or ''}_{re.sub(r'[./]', '_', arxiv_id)}",
            "section": None,
            "bibtype": "misc",
            "title": _text(entry, "atom:title"),
            "authors": authors,
            "year": year,
            "venue": _text(entry, "arxiv:journal_ref") or "arXiv preprint",
            "volume": None,
            "number": None,
            "pages": None,
            "doi": _text(entry, "arxiv:doi"),
            "arxiv_id": arxiv_id,
            "code": None,
            "note": None,
            "modality": None,
            "ai_method": None,
            "summary": None,
            "tags": ["arxiv-auto"] + ([category.lower()] if category else []),
            "verified": False,
            "verified_on": None,
            "abstract_reviewed": False,
            # Extra fields only auto-fetched entries carry (arXiv metadata is CC0):
            "arxiv_version": version,
            "primary_category": category,
            "abstract": _text(entry, "atom:summary"),
        })

    return papers


def update_database(new_papers, db_path: str = DEFAULT_DB_PATH):
    """Merge newly fetched papers into the local JSON database without duplicates."""
    db = []
    if os.path.exists(db_path):
        with open(db_path, "r", encoding="utf-8") as f:
            try:
                db = json.load(f)
            except json.JSONDecodeError as e:
                # Never overwrite a database we could not read: that would destroy it.
                sys.exit(f"[!] {db_path} is not valid JSON ({e}); fix it before updating. Nothing was written.")

    # Compare version-less ids so v1/v2 of the same paper are not duplicated.
    existing_ids = {base_arxiv_id(p["arxiv_id"]) for p in db if p.get("arxiv_id")}
    added = [p for p in new_papers if p["arxiv_id"] not in existing_ids]
    for p in added:
        existing_ids.add(p["arxiv_id"])
    if not added:
        print(f"[=] No new papers; {db_path} left unchanged ({len(db)} papers).")
        return

    # Write to a temp file and swap it in, so an interrupted write cannot truncate the DB.
    # newline="\n" keeps line endings stable across Windows and Linux.
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(db_path)), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(db + added, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, db_path)
    except BaseException:
        os.unlink(tmp_path)
        raise

    print(f"[+] Appended {len(added)} new papers to {db_path}. Total papers in DB: {len(db) + len(added)}")


def main():
    parser = argparse.ArgumentParser(description="Query arXiv API for C-UAS and Drone Detection papers.")
    parser.add_argument("--query", type=str, default='all:"drone detection" AND all:"deep learning"',
                        help='arXiv search query, using field prefixes, e.g. \'abs:"counter-UAS" AND abs:"deep learning"\'')
    parser.add_argument("--max_results", type=int, default=5, help="Number of papers to fetch")
    parser.add_argument("--update_db", action="store_true", help="Append fetched papers to papers_database.json")
    parser.add_argument("--db_path", type=str, default=DEFAULT_DB_PATH, help="Path to papers_database.json")
    args = parser.parse_args()

    try:
        results = query_arxiv(args.query, max_results=args.max_results)
    except ValueError as e:
        sys.exit(f"[!] {e}")

    print(f"\nFound {len(results)} papers:")
    print("-" * 80)
    for i, paper in enumerate(results, 1):
        authors = paper["authors"]
        print(f"{i}. [{paper['arxiv_id']}] ({paper['year']}) {paper['title']}")
        print(f"   Authors: {', '.join(authors[:4])}{'...' if len(authors) > 4 else ''}")
        print(f"   Abstract: {(paper['abstract'] or '')[:150]}...\n")

    if args.update_db and results:
        update_database(results, args.db_path)


if __name__ == "__main__":
    main()
