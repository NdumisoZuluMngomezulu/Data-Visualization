import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for testing.

import pandas as pd

import sys
from pathlib import Path

# Add the directory containing visualization_libraries.py to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src" / "data_package"

sys.path.insert(0, str(SOURCE_DIR))

from visualization_libraries import (
    to_number,
    load_data,
    create_charts,
    main,
)


class TestToNumber(unittest.TestCase):
    """Tests for converting source values into numbers."""

    def test_none_returns_zero(self):
        self.assertEqual(to_number(None), 0)

    def test_empty_string_returns_zero(self):
        self.assertEqual(to_number(""), 0)

    def test_integer_is_preserved(self):
        self.assertEqual(to_number(123), 123)

    def test_float_is_preserved(self):
        self.assertEqual(to_number(12.5), 12.5)

    def test_string_with_spaces_as_thousands_separators(self):
        self.assertEqual(to_number("61 927"), 61927)

    def test_string_with_commas_as_thousands_separators(self):
        self.assertEqual(to_number("1,234"), 1234)

    def test_string_with_spaces_and_commas(self):
        self.assertEqual(to_number(" 2, 345 "), 2345)

    def test_whitespace_only_string_returns_zero(self):
        self.assertEqual(to_number("   "), 0)

    def test_invalid_numeric_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            to_number("not a number")


class TestLoadData(unittest.TestCase):
    """Tests for loading, validating, and cleaning JSON records."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.input_path = Path(self.temp_dir.name) / "test_covid.json"

    def write_json(self, data):
        with open(self.input_path, "w", encoding="utf-8") as file:
            json.dump(data, file)

    def valid_record(self, date="2020-03-05"):
        return {
            "Date": date,
            "Total Confirmed Cases": "1 234",
            "Total Deaths": "12",
            "Total Recovered": "100",
            "Active Cases": "1,122",
            "Daily Confirmed Cases": "25",
            "Daily  deaths": "2",
        }

    def test_loads_valid_json_data(self):
        self.write_json([self.valid_record()])

        df = load_data(self.input_path)

        self.assertEqual(len(df), 1)
        self.assertIn("Date", df.columns)
        self.assertIn("Daily Deaths", df.columns)

    def test_converts_numeric_columns(self):
        self.write_json([self.valid_record()])

        df = load_data(self.input_path)
        record = df.iloc[0]

        self.assertEqual(record["Total Confirmed Cases"], 1234)
        self.assertEqual(record["Total Deaths"], 12)
        self.assertEqual(record["Total Recovered"], 100)
        self.assertEqual(record["Active Cases"], 1122)
        self.assertEqual(record["Daily Confirmed Cases"], 25)
        

    def test_sorts_records_by_date(self):
        records = [
            self.valid_record("2020-03-07"),
            self.valid_record("2020-03-05"),
            self.valid_record("2020-03-06"),
        ]
        self.write_json(records)

        df = load_data(self.input_path)

        dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()
        self.assertEqual(
            dates,
            ["2020-03-05", "2020-03-06", "2020-03-07"],
        )

    def test_drops_records_with_invalid_dates(self):
        records = [
            self.valid_record("2020-03-05"),
            self.valid_record("not-a-date"),
        ]
        self.write_json(records)

        df = load_data(self.input_path)

        self.assertEqual(len(df), 1)
        self.assertEqual(
            df.iloc[0]["Date"].strftime("%Y-%m-%d"),
            "2020-03-05",
        )

    def test_rejects_empty_json_array(self):
        self.write_json([])

        with self.assertRaisesRegex(
            ValueError,
            "Expected a non-empty JSON array",
        ):
            load_data(self.input_path)

    def test_rejects_json_object_instead_of_array(self):
        self.write_json({"Date": "2020-03-05"})

        with self.assertRaisesRegex(
            ValueError,
            "Expected a non-empty JSON array",
        ):
            load_data(self.input_path)

    def test_rejects_missing_required_columns(self):
        self.write_json([
            {
                "Date": "2020-03-05",
                "Total Deaths": "1",
            }
        ])

        with self.assertRaisesRegex(
            ValueError,
            "Missing required JSON field",
        ):
            load_data(self.input_path)

    def test_handles_missing_daily_deaths_column(self):
        record = self.valid_record()
        del record["Daily  deaths"]
        self.write_json([record])

        df = load_data(self.input_path)

        self.assertEqual(df.iloc[0]["Daily Deaths"], 0)

    def test_handles_alternate_daily_deaths_spacing(self):
        record = self.valid_record()
        del record["Daily  deaths"]
        record["daily deaths"] = "7"
        self.write_json([record])

        df = load_data(self.input_path)

        self.assertEqual(df.iloc[0]["Daily Deaths"], 7)

    def test_returns_empty_dataframe_if_all_dates_are_invalid(self):
        self.write_json([self.valid_record("invalid-date")])

        df = load_data(self.input_path)

        self.assertTrue(df.empty)


class TestCreateCharts(unittest.TestCase):
    """Tests that charts and the cleaned CSV are created."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output_dir = Path(self.temp_dir.name) / "output"

        self.df = pd.DataFrame({
            "Date": pd.to_datetime([
                "2020-03-05",
                "2020-03-06",
            ]),
            "Total Confirmed Cases": [10, 20],
            "Total Deaths": [1, 2],
            "Total Recovered": [3, 5],
            "Active Cases": [6, 13],
            "Daily Confirmed Cases": [10, 10],
            "Daily Deaths": [1, 1],
        })

    def test_creates_output_directory_and_files(self):
        create_charts(self.df, self.output_dir)

        self.assertTrue(self.output_dir.is_dir())
        self.assertTrue(
            (self.output_dir / "covid_dashboard.png").is_file()
        )
        self.assertTrue(
            (self.output_dir / "daily_deaths.png").is_file()
        )
        self.assertTrue(
            (self.output_dir / "covid_cleaned.csv").is_file()
        )

    def test_exports_cleaned_csv_with_expected_columns(self):
        create_charts(self.df, self.output_dir)

        csv_path = self.output_dir / "covid_cleaned.csv"
        exported_df = pd.read_csv(csv_path)

        self.assertIn("Date", exported_df.columns)
        self.assertIn("Total Confirmed Cases", exported_df.columns)
        self.assertIn("Daily Deaths", exported_df.columns)
        self.assertEqual(len(exported_df), 2)

    def test_prints_output_file_locations(self):
        with patch("builtins.print") as mock_print:
            create_charts(self.df, self.output_dir)

        printed_text = " ".join(
            str(call.args[0]) for call in mock_print.call_args_list
        )

        self.assertIn("Read 2 daily records", printed_text)
        self.assertIn("covid_dashboard.png", printed_text)
        self.assertIn("daily_deaths.png", printed_text)
        self.assertIn("covid_cleaned.csv", printed_text)


class TestMain(unittest.TestCase):
    """Tests command-line argument handling."""

    def test_main_uses_supplied_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.json"
            output_dir = temp_path / "charts"

            with open(input_path, "w", encoding="utf-8") as file:
                json.dump(
                    [{
                        "Date": "2020-03-05",
                        "Total Confirmed Cases": "10",
                        "Total Deaths": "1",
                        "Total Recovered": "2",
                        "Active Cases": "7",
                        "Daily Confirmed Cases": "10",
                    }],
                    file,
                )

            with patch(
                "sys.argv",
                [
                    "visualise_covid.py",
                    "--input", str(input_path),
                    "--output-dir", str(output_dir),
                ],
            ):
                main()

            self.assertTrue(
                (output_dir / "covid_dashboard.png").is_file()
            )
            self.assertTrue(
                (output_dir / "daily_deaths.png").is_file()
            )
            self.assertTrue(
                (output_dir / "covid_cleaned.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
