#!/usr/bin/env python3
"""
Tests for the tools in this folder. Standard library only, no network.

    python -m unittest discover -s tools -v
"""
import io
import json
import math
import os
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_bibliography as bib  # noqa: E402
import check_docs  # noqa: E402
import fetch_arxiv_papers as fetch  # noqa: E402
import geometry_budget as geo  # noqa: E402

FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
 <entry><id>http://arxiv.org/abs/2609.01742v2</id><published>2026-09-01T00:00:00Z</published>
  <title>Evidential Deep
   Learning</title><summary> abstract  text </summary>
  <author><name>Dmitry Golovchits</name></author>
  <arxiv:primary_category term="cs.CV"/></entry>
 <entry><id>http://arxiv.org/abs/2610.00001v1</id><published>2026-10-01T00:00:00Z</published>
  <title>New Paper</title><summary>x</summary><author><name>Ada O'Neil</name></author></entry>
 <entry><id>http://arxiv.org/abs/hep-th/9901001v1</id><title>Old style</title></entry>
</feed>"""

ERROR_FEED = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/api/errors#incorrect_id_format_for_1234</id><title>Error</title>
<summary>incorrect id format for 1234</summary></entry></feed>"""


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code):
    return urllib.error.HTTPError("https://export.arxiv.org", code, "error", {}, io.BytesIO())


class FetchArxivTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "db.json")
        self.quiet = mock.patch("sys.stdout", io.StringIO())
        self.quiet.start()

    def tearDown(self):
        self.quiet.stop()
        self.tmp.cleanup()

    def test_parse_feed(self):
        papers = fetch.parse_arxiv_atom(FEED)
        self.assertEqual([p["arxiv_id"] for p in papers], ["2609.01742", "2610.00001", "hep-th/9901001"])
        first = papers[0]
        self.assertEqual((first["arxiv_version"], first["title"], first["abstract"]), ("v2", "Evidential Deep Learning", "abstract text"))
        self.assertEqual(first["tags"], ["arxiv-auto", "cs.cv"])
        self.assertFalse(first["verified"])
        self.assertEqual(papers[1]["id"], "oneil2026_2610_00001")
        self.assertIsNone(papers[2]["year"])

    def test_auto_entries_share_the_curated_schema(self):
        with open(bib.DB_PATH, encoding="utf-8") as f:
            curated_keys = set(json.load(f)[0])
        self.assertLessEqual(curated_keys, set(fetch.parse_arxiv_atom(FEED)[0]))

    def test_error_feed_raises(self):
        with self.assertRaisesRegex(ValueError, "incorrect id format"):
            fetch.parse_arxiv_atom(ERROR_FEED)

    def test_update_dedupes_across_versions_and_writes_lf(self):
        with open(self.db, "w", encoding="utf-8") as f:
            json.dump([{"id": "legacy", "arxiv_id": "2609.01742v1"}], f)  # old format kept the version suffix
        fetch.update_database(fetch.parse_arxiv_atom(FEED), self.db)
        with open(self.db, "rb") as f:
            raw = f.read()
        self.assertNotIn(b"\r\n", raw)
        self.assertEqual([p["arxiv_id"] for p in json.loads(raw)], ["2609.01742v1", "2610.00001", "hep-th/9901001"])

    def test_update_with_nothing_new_leaves_file_untouched(self):
        fetch.update_database(fetch.parse_arxiv_atom(FEED), self.db)
        before = os.stat(self.db).st_mtime_ns
        fetch.update_database(fetch.parse_arxiv_atom(FEED), self.db)
        self.assertEqual(os.stat(self.db).st_mtime_ns, before)

    def test_corrupt_database_is_never_overwritten(self):
        with open(self.db, "w", encoding="utf-8") as f:
            f.write('[{"id": "x",')
        with self.assertRaises(SystemExit):
            fetch.update_database(fetch.parse_arxiv_atom(FEED), self.db)
        with open(self.db, encoding="utf-8") as f:
            self.assertEqual(f.read(), '[{"id": "x",')
        self.assertEqual(os.listdir(self.tmp.name), ["db.json"])  # no leftover temp file

    def _query(self, responses):
        calls = iter(responses)

        def fake_urlopen(req, timeout):
            r = next(calls)
            if isinstance(r, Exception):
                raise r
            return _Response(r)

        with mock.patch.object(fetch.urllib.request, "urlopen", fake_urlopen), \
                mock.patch.object(fetch.time, "sleep") as sleep, \
                mock.patch("sys.stdout", io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            result = fetch.query_arxiv('abs:"x"', 1)
        return result, [c.args[0] for c in sleep.call_args_list]

    def test_throttling_is_retried_with_backoff(self):
        result, waits = self._query([_http_error(406), b"Rate exceeded.", FEED])
        self.assertEqual(len(result), 3)
        self.assertEqual(waits, [5, 10])

    def test_no_sleep_after_the_final_attempt(self):
        result, waits = self._query([_http_error(503)] * 4)
        self.assertEqual(result, [])
        self.assertEqual(waits, [5, 10, 20])  # 4 attempts, 3 waits

    def test_non_retryable_error_gives_up_at_once(self):
        result, waits = self._query([_http_error(400)])
        self.assertEqual((result, waits), ([], []))


class BuildBibliographyTests(unittest.TestCase):
    PAPER = {"id": "k2026x", "bibtype": "article", "title": "DIAT-µSAT & Friends", "authors": ["A B", "C D"],
             "year": 2026, "venue": "Radar & Sonar", "volume": "7", "number": None, "pages": "1--5",
             "doi": "10.1/x", "arxiv_id": "2601.00001", "code": None, "note": None, "verified": True}

    def test_entry_escapes_and_fields(self):
        out = bib.to_bibtex(self.PAPER)
        self.assertTrue(out.startswith("@article{k2026x,"))
        self.assertIn(r"title = {{DIAT-$\mu$SAT \& Friends}}", out)
        self.assertIn(r"journal = {Radar \& Sonar}", out)
        self.assertIn("author = {A B and C D}", out)
        self.assertIn("eprint = {2601.00001}", out)
        self.assertNotIn("number", out)

    def test_unverified_entries_are_not_exported(self):
        out = bib.render([self.PAPER, dict(self.PAPER, id="auto", verified=False)])
        self.assertIn("k2026x", out)
        self.assertNotIn("auto", out)

    def test_repository_bib_is_up_to_date(self):
        with open(bib.DB_PATH, encoding="utf-8") as f:
            expected = bib.render(json.load(f))
        with open(bib.BIB_PATH, encoding="utf-8") as f:
            self.assertEqual(f.read(), expected, "run: python tools/build_bibliography.py")


class GeometryBudgetTests(unittest.TestCase):
    def test_matches_doc_13_worked_numbers(self):
        self.assertEqual(round(geo.f_px(30) * 0.35 / 10), 251)        # 13 section 2.3
        self.assertAlmostEqual(geo.hfov_deg(10.3 * 40 * 10 / 0.35), 18.5, delta=0.05)   # 13 section 3 example
        self.assertAlmostEqual(geo.vfov_deg(60), 36.0, delta=0.1)     # 13 section 2.2 table

    def test_scenario_a(self):
        b = geo.budget(geo.SCENARIOS["A"])
        t = {r["name"]: r for r in b["threats"]}
        self.assertEqual((round(b["row_hfov_deg"], 1), b["row_cameras"]), (25.8, 4))   # sized by the FPV
        self.assertAlmostEqual(t["FPV"]["warning_s"], 5.0)            # exactly the timeline it was sized for
        self.assertTrue(t["FPV, low"]["below_skyline"])
        self.assertNotIn("warning_s", t["Mavic-class quad"])          # 100 m up: above the row until too small

    def test_skyline_caps_warning(self):
        shahed = next(r for r in geo.budget(geo.SCENARIOS["B"])["threats"] if r["name"] == "Shahed-class")
        # Farther than 95 m / tan(2 deg) it sits below the skyline, however many pixels it has.
        self.assertAlmostEqual(shahed["warning_s"], (95 / math.tan(math.radians(2)) - 150) / 50, places=6)


class CheckDocsTests(unittest.TestCase):
    def test_github_slug(self):
        self.assertEqual(check_docs.github_slug("🧭 Current Design at a Glance"), "-current-design-at-a-glance")
        self.assertEqual(check_docs.github_slug("4.1 DIAT-µSAT: Small Targets' Micro-Doppler"), "41-diat-µsat-small-targets-micro-doppler")
        self.assertEqual(check_docs.github_slug("8. Frontier Detectors & Data (2024–2026)"), "8-frontier-detectors--data-20242026")

    def test_duplicate_headings_get_suffixes(self):
        self.assertEqual(check_docs.anchors("# Notes\n## Notes\n```\n# not a heading\n```"), {"notes", "notes-1"})

    def test_repository_docs_are_consistent(self):
        docs = {}
        for name in os.listdir(bib.ROOT):
            if name.endswith(".md"):
                with open(os.path.join(bib.ROOT, name), encoding="utf-8") as f:
                    docs[name] = f.read()
        problems = (check_docs.check_links(docs) + check_docs.check_mermaid(docs)
                    + check_docs.check_references(docs["01_research_papers_compendium.md"]))
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
