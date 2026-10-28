#!/usr/bin/env python3
"""
check_docs.py
Offline consistency checks for the compendium. Prints every problem and exits 1 if there are any.

  1. Every relative Markdown link resolves, including #anchors (GitHub heading slugs).
  2. 06_bibliography.bib matches papers_database.json (see build_bibliography.py).
  3. Every verified paper in papers_database.json is cited (by DOI, arXiv ID or code URL)
     in 01_research_papers_compendium.md.
  4. Mermaid node and edge labels avoid characters that break the parser.

    python tools/check_docs.py
"""
import glob
import json
import os
import re
import sys
import unicodedata
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_bibliography import BIB_PATH, DB_PATH, ROOT, read_text, render  # noqa: E402

CODE_BLOCK = re.compile(r"```.*?```", re.S)


def github_slug(heading: str) -> str:
    """Anchor GitHub generates for a heading: lowercase, drop punctuation/emoji, spaces -> hyphens."""
    h = re.sub(r"[`*_]", "", heading.strip().lower())
    return "".join(c for c in h if c in " -" or unicodedata.category(c)[0] in "LN").replace(" ", "-")


def anchors(text: str) -> set:
    seen, out = {}, set()
    for heading in re.findall(r"^#{1,6}\s+(.*)$", CODE_BLOCK.sub("", text), re.M):
        slug = github_slug(heading)
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        out.add(slug if n == 0 else f"{slug}-{n}")
    return out


def check_links(docs: dict) -> list:
    problems, anchor_cache = [], {}
    for name, text in docs.items():
        for target in re.findall(r"\]\(([^)\s]+)\)", CODE_BLOCK.sub("", text)):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if target.startswith("file:"):
                problems.append(f"{name}: absolute file link {target}")
                continue
            path, _, frag = target.partition("#")
            path = urllib.parse.unquote(path) or name
            if not os.path.exists(os.path.join(ROOT, path)):
                problems.append(f"{name}: missing file {target}")
            elif frag and path.endswith(".md"):
                if path not in anchor_cache:
                    anchor_cache[path] = anchors(docs.get(path) or read_text(os.path.join(ROOT, path)))
                if urllib.parse.unquote(frag) not in anchor_cache[path]:
                    problems.append(f"{name}: missing anchor {target}")
    return problems


def check_references(compendium: str) -> list:
    db = json.loads(read_text(DB_PATH))
    problems = []
    if read_text(BIB_PATH) != render(db):
        problems.append("06_bibliography.bib is out of date: run  python tools/build_bibliography.py")
    for p in db:
        if not p.get("verified"):
            continue
        cited = [s for s in (p.get("doi"), p.get("arxiv_id") and f"arxiv.org/abs/{p['arxiv_id']}", p.get("code")) if s]
        if not any(s in compendium for s in cited):
            problems.append(f"01_research_papers_compendium.md: verified paper '{p['id']}' is not cited")
    return problems


def check_mermaid(docs: dict) -> list:
    problems = []
    for name, text in docs.items():
        for block in re.findall(r"```mermaid\n(.*?)```", text, re.S):
            for label in re.findall(r"(\[[^\]]*\]|\{[^}]*\}|\|[^|]*\|)", block):
                if not label[1:-1].startswith('"') and re.search(r'[();"]', label[1:-1]):
                    problems.append(f"{name}: Mermaid label {label} contains ( ) ; or \" (quote the label)")
    return problems


def main():
    docs = {os.path.basename(p): read_text(p) for p in sorted(glob.glob(os.path.join(ROOT, "*.md")))}
    problems = (check_links(docs) + check_references(docs.get("01_research_papers_compendium.md", ""))
                + check_mermaid(docs))
    for p in problems:
        print("FAIL:", p)   # ASCII only: Windows pipes/redirects use cp1252 and would crash on symbols
    links = sum(len(re.findall(r"\]\((?!https?:)[^)\s]+\)", CODE_BLOCK.sub("", t))) for t in docs.values())
    print(f"{len(docs)} Markdown files, {links} relative links checked: "
          + (f"{len(problems)} problem(s)" if problems else "all OK"))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
