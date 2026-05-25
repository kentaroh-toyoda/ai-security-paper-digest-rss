"""Tests for utils.scholars matching."""

import os
import sys
import unittest

# Make project root importable when running with `python -m unittest`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.scholars import (  # noqa: E402
    Scholar,
    _normalize,
    build_index,
    load_scholars,
    match_authors,
)


class NormalizeTests(unittest.TestCase):
    def test_lowercases(self):
        self.assertEqual(_normalize("Dawn Song"), "dawn song")

    def test_strips_accents(self):
        self.assertEqual(_normalize("Florian Tramèr"), "florian tramer")

    def test_collapses_whitespace(self):
        self.assertEqual(_normalize("  Bo   Li  "), "bo li")


class MatchAuthorsTests(unittest.TestCase):
    def setUp(self):
        self.scholars = load_scholars()
        self.index = build_index(self.scholars)

    def test_exact_match(self):
        matched = match_authors(["Daniel Kang", "Random Person"], self.index)
        names = [s.name for s in matched]
        self.assertIn("Daniel Kang", names)

    def test_accent_fold_match(self):
        # Paper might cite Florian Tramèr as "Florian Tramer" (no accent).
        matched = match_authors(["Florian Tramer"], self.index)
        names = [s.name for s in matched]
        self.assertIn("Florian Tramèr", names)

    def test_case_insensitive_match(self):
        matched = match_authors(["dawn song"], self.index)
        names = [s.name for s in matched]
        self.assertIn("Dawn Song", names)

    def test_no_match_returns_empty(self):
        self.assertEqual(
            match_authors(["Someone Totally Random", "Another Stranger"], self.index),
            [],
        )

    def test_name_variant_matches(self):
        # Niloofar Mireshghallah has variant "Fatemehsadat Mireshghallah".
        matched = match_authors(["Fatemehsadat Mireshghallah"], self.index)
        names = [s.name for s in matched]
        self.assertIn("Niloofar Mireshghallah", names)

    def test_deduplicates_when_one_scholar_matches_twice(self):
        # If an author list redundantly contains the same scholar name, only one entry returns.
        matched = match_authors(["Dawn Song", "Dawn Song"], self.index)
        self.assertEqual(sum(1 for s in matched if s.name == "Dawn Song"), 1)


class LoadScholarsTests(unittest.TestCase):
    def test_loads_expected_count(self):
        scholars = load_scholars()
        # scholars.yaml currently has 28 entries — assert a sane minimum so this
        # test stays useful if the list grows.
        self.assertGreaterEqual(len(scholars), 20)
        self.assertTrue(all(isinstance(s, Scholar) for s in scholars))

    def test_every_scholar_has_required_fields(self):
        for s in load_scholars():
            self.assertTrue(s.name)
            self.assertTrue(s.affiliation)
            self.assertTrue(s.sub_area)
            self.assertTrue(s.arxiv_authors)


if __name__ == "__main__":
    unittest.main()
