"""Regression checks for DRatings NCAAM final-score upsert reconciliation.

Run from the repository root:
    python -m unittest discover -s docs/win/basketball/tests -p 'test_transform_basketball_final_upsert.py'
"""
import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/00_intake/transform_basketball.py"
spec = importlib.util.spec_from_file_location("transform_basketball_final_upsert", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def final(game_id, date, home, away, home_score, away_score):
    return {
        "sport": "Basketball",
        "league": "NCAAM",
        "game_id": game_id,
        "game_date": date,
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
        "total": home_score + away_score,
        "home_spread": home_score - away_score,
        "away_spread": away_score - home_score,
    }


class FinalScoreUpsertRegression(unittest.TestCase):
    def setUp(self):
        self.official = [
            final("401911472", "2026_11_15", "San José State Spartans", "Alabama State Hornets", 80, 68),
            final("401926772", "2026_11_15", "USC Trojans", "Arkansas-Pine Bluff Golden Lions", 122, 46),
            final("401917528", "2026_11_15", "Seattle U Redhawks", "Idaho State Bengals", 77, 70),
            final("401926068", "2026_11_15", "Eastern Illinois Panthers", "Purdue Fort Wayne Mastodons", 74, 86),
            final("401909525", "2026_11_15", "Missouri Tigers", "Marquette Golden Eagles", 69, 64),
            final("401921287", "2026_11_15", "Holy Cross Crusaders", "UAlbany Great Danes", 60, 62),
            final("401920136", "2026_11_15", "North Dakota State Bison", "Cal State Bakersfield Roadrunners", 69, 73),
            final("401921804", "2026_11_16", "Kentucky Wildcats", "Grambling Tigers", 88, 59),
            final("401924265", "2026_11_16", "UCLA Bruins", "Long Beach State Beach", 105, 66),
        ]
        self.dratings = [
            final("", "2026_11_15", "San Jose State Spartans", "Alabama State Hornets", 80, 68),
            final("", "2026_11_15", "USC Trojans", "Ark Pine Bluff Golden Lions", 122, 46),
            final("", "2026_11_15", "Seattle Redhawks", "Idaho State Bengals", 77, 70),
            final("", "2026_11_15", "Eastern Illinois Panthers", "IPFW Mastodons", 74, 86),
            final("", "2026_11_15", "Marquette Golden Eagles", "Missouri Tigers", 64, 69),
            final("", "2026_11_15", "Holy Cross Crusaders", "Albany Great Danes", 60, 62),
            final("", "2026_11_15", "North Dakota State Bison", "CSU Bakersfield Roadrunners", 69, 73),
            final("", "2026_11_16", "Kentucky Wildcats", "Grambling State Tigers", 88, 59),
            final("", "2026_11_16", "UCLA Bruins", "Long Beach State 49ers", 105, 66),
        ]

    def test_nine_historical_duplicates_are_not_reintroduced(self):
        merged = module.collapse_rows(self.official + self.dratings)
        self.assertEqual(merged, self.official)
        self.assertEqual(module.collapse_rows(merged + self.dratings), self.official)

    def test_conflicting_scores_fail_closed(self):
        incorrect = dict(self.dratings[1], away_score=47)
        with self.assertRaisesRegex(ValueError, "conflicting scores"):
            module.collapse_rows(self.official + [incorrect])

    def test_distinct_unidentified_game_is_retained(self):
        other = final("", "2026_11_15", "Example University", "Different College", 80, 70)
        merged = module.collapse_rows(self.official + self.dratings + [other])
        self.assertEqual(len(merged), 10)
        self.assertEqual(merged[-1], other)


if __name__ == "__main__":
    unittest.main()
