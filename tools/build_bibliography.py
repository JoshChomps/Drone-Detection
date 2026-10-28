#!/usr/bin/env python3
"""
build_bibliography.py
Generate 06_bibliography.bib from papers_database.json, the single source of truth for references.
Only entries marked "verified": true are exported, so auto-fetched candidates never reach the .bib.

    python tools/build_bibliography.py           # rewrite 06_bibliography.bib
    python tools/build_bibliography.py --check   # exit 1 if the .bib is out of date
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "papers_database.json")
BIB_PATH = os.path.join(ROOT, "06_bibliography.bib")
HEADER = ("% 06_bibliography.bib -- GENERATED from papers_database.json by tools/build_bibliography.py.\n"
          "% Do not edit by hand: edit the JSON, then re-run the tool.\n\n")
VENUE_FIELD = {"article": "journal", "inproceedings": "booktitle", "misc": "howpublished"}


def read_text(path: str) -> str:
    """File contents, or "" if it does not exist."""
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def tex(s: str) -> str:
    return s.replace("&", r"\&").replace("µ", r"$\mu$")


def to_bibtex(p: dict) -> str:
    fields = [
        ("title", "{" + tex(p["title"]) + "}"),     # double braces keep capitalisation
        ("author", " and ".join(p["authors"])),
        ("year", str(p["year"])),
        (VENUE_FIELD[p["bibtype"]], tex(p["venue"])),
    ]
    fields += [(k, p[k]) for k in ("volume", "number", "pages", "doi") if p.get(k)]
    if p.get("arxiv_id"):
        fields += [("eprint", p["arxiv_id"]), ("archivePrefix", "arXiv")]
    if p.get("code"):
        fields.append(("url", p["code"]))
    if p.get("note"):
        fields.append(("note", tex(p["note"])))
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields)
    return f"@{p['bibtype']}{{{p['id']},\n{body}\n}}\n"


def render(db: list) -> str:
    return HEADER + "\n".join(to_bibtex(p) for p in db if p.get("verified"))


def main():
    parser = argparse.ArgumentParser(description="Generate 06_bibliography.bib from papers_database.json.")
    parser.add_argument("--check", action="store_true", help="exit 1 if the .bib differs from the generated output")
    args = parser.parse_args()

    with open(DB_PATH, encoding="utf-8") as f:
        bib = render(json.load(f))

    if args.check:
        if read_text(BIB_PATH) != bib:
            sys.exit("06_bibliography.bib is out of date: run  python tools/build_bibliography.py")
        print("06_bibliography.bib is up to date")
        return

    with open(BIB_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(bib)
    print(f"Wrote {bib.count(chr(10) + '@') + bib.startswith('@')} entries to {BIB_PATH}")


if __name__ == "__main__":
    main()
