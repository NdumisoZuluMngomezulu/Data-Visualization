import json
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd

from covid_analysis import (
    standardize_column,
    clean_value,
    standardize_date,
    load_data,
    get_values,
    star_graph,
    trend_graph,
)


class TestStandardizeColumn(unittest.TestCase):
    """Tests for column name standardization."""

    def test_lowercases_column_name(self):
        self.assertEqual(
            standardize_column("Daily Cases"),
            "daily cases"
        )

    def test_removes_extra_spaces(self):
        self.assertEqual(
            standardize_column("  Daily   Confirmed   Cases  "),
            "daily confirmed cases"
        )

    def test_converts_numbers_to_strings(self):
        self.assertEqual(standardize_column(123), "123")


class TestCleanValue(unittest.TestCase):
    """Tests for cleaning numeric values."""

    def test_integer(self):
        self.assertEqual(clean_value(100), 100)

    def test_numeric_string(self):
        self.assertEqual(clean_value("250"), 250)

    def test_removes_commas(self):
        self.assertEqual(clean_value("1,234"), 1234)

    def test_removes_spaces(self):
        self.assertEqual(clean_value("1 234"), 1234)

    def test_handles_decimal(self):
        self.assertEqual(clean_value("123.9"), 123)

    def test_handles_empty_string(self):
        self.assertEqual(clean_value(""), 0)

    def test_handles_whitespace(self):
        self.assertEqual(clean_value("   "), 0)

    def test_handles_none(self):
        self.assertEqual(clean_value(None), 0)

    def test_handles_nan(self):
        self.assertEqual(clean_value(np.nan), 0)

    def test_handles_invalid_string(self):
        self.assertEqual(clean_value("not a number"), 0)

    def test_handles_infinity(self):
        self.assertEqual(clean_value("inf"), 0)

    def test_handles_negative_number(self):
        self.assertEqual(clean_value("-50"), -50)


class TestStandardizeDate(unittest.TestCase):
    """Tests for parsing supported date formats."""

    def test_year_month_day_with_slashes(self):
        result = standardize_date("2026/10/25")

        self.assertEqual(
            result,
            datetime(2026, 10, 25)
        )

    def test_year_month_day_with_hyphens(self):
        result = standardize_date("2026-10-25")

        self.assertEqual(
            result,
            datetime(2026, 10, 25)
        )

    def test_day_month_year(self):
        result = standardize_date("25/10/2026")

        self.assertEqual(
            result,
            datetime(2026, 10, 25)
        )

    def test_handles_surrounding_spaces(self):
        result = standardize_date(" 2026-10-25 ")

        self.assertEqual(
            result,
            datetime(2026, 10, 25)
        )

    def test_invalid_date_returns_none(self):
        self.assertIsNone(
            standardize_date("2026-15-40")
        )

    def test_invalid_format_returns_none(self):
        self.assertIsNone(
            standardize_date("October 25, 2026")
        )

    def test_none_returns_none(self):
        self.assertIsNone(standardize_date(None))


class TestLoadData(unittest.TestCase):
    """Tests for loading, cleaning, and sorting JSON data."""

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.file_path = (
            Path(self.temp_dir.name) / "test_covid.json"
        )

    def write_json(self, data):
        with open(self.file_path, "w", encoding="utf-8") as file:
            json.dump(data, file)

        return self.file_path

    def test_loads_valid_json_records(self):
        data = [
            {
                "Date": "2026-10-24",
                "Daily Confirmed Cases": "1,200"
            },
            {
                "Date": "2026-10-25",
                "Daily Confirmed Cases": "1,500"
            },
        ]

        self.write_json(data)
        df = load_data(self.file_path)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(
            df["Daily Confirmed Cases"].tolist(),
            [1200, 1500]
        )

    def test_loads_json_data_wrapper(self):
        data = {
            "data": [
                {
                    "Date": "2026-10-25",
                    "Cases": "500"
                }
            ]
        }

        self.write_json(data)
        df = load_data(self.file_path)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "cases"], 500)

    def test_standardizes_column_names(self):
        data = [
            {
                "  DATE  ": "2026-10-25",
                " DAILY   CASES ": "100"
            }
        ]

        self.write_json(data)
        df = load_data(self.file_path)

        self.assertIn("Date", df.columns)
        self.assertIn("daily cases", df.columns)

    def test_sorts_records_chronologically(self):
        data = [
            {
                "Date": "2026-10-25",
                "Cases": "300"
            },
            {
                "Date": "2026-10-23",
                "Cases": "100"
            },
            {
                "Date": "2026-10-24",
                "Cases": "200"
            },
        ]

        self.write_json(data)
        df = load_data(self.file_path)

        self.assertEqual(
            df["Date"].tolist(),
            [
                datetime(2026, 10, 23),
                datetime(2026, 10, 24),
                datetime(2026, 10, 25),
            ]
        )

    def test_removes_records_with_invalid_dates(self):
        data = [
            {
                "Date": "2026-10-24",
                "Cases": "200"
            },
            {
                "Date": "invalid-date",
                "Cases": "500"
            },
        ]

        self.write_json(data)
        df = load_data(self.file_path)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "cases"], 200)

    def test_cleans_numeric_values(self):
        data = [
            {
                "Date": "2026-10-25",
                "Cases": "1,200",
                "Deaths": " 25 ",
                "Recoveries": "",
                "Other": "invalid"
            }
        ]

        self.write_json(data)
        df = load_data(self.file_path)

        self.assertEqual(df.loc[0, "cases"], 1200)
        self.assertEqual(df.loc[0, "deaths"], 25)
        self.assertEqual(df.loc[0, "recoveries"], 0)
        self.assertEqual(df.loc[0, "other"], 0)

    def test_supports_multiple_date_formats(self):
        data = [
            {
                "Date": "25/10/2026",
                "Cases": "300"
            },
            {
                "Date": "2026/10/23",
                "Cases": "100"
            },
            {
                "Date": "2026-10-24",
                "Cases": "200"
            },
        ]

        self.write_json(data)
        df = load_data(self.file_path)

        self.assertEqual(
            df["Date"].tolist(),
            [
                datetime(2026, 10, 23),
                datetime(2026, 10, 24),
                datetime(2026, 10, 25),
            ]
        )

    def test_rejects_non_list_json(self):
        self.write_json("this is not a JSON records list")

        with self.assertRaises(ValueError):
            load_data(self.file_path)

    def test_rejects_empty_data(self):
        self.write_json([])

        with self.assertRaisesRegex(
            ValueError,
            "No valid records found"
        ):
            load_data(self.file_path)

    def test_rejects_missing_date_column(self):
        self.write_json([
            {
                "Cases": "100"
            }
        ])

        with self.assertRaisesRegex(
            ValueError,
            "No valid records found"
        ):
            load_data(self.file_path)

    def test_rejects_all_invalid_dates(self):
        self.write_json([
            {
                "Date": "not-a-date",
                "Cases": "100"
            }
        ])

        with self.assertRaisesRegex(
            ValueError,
            "No valid records found"
        ):
            load_data(self.file_path)

    def test_rejects_duplicate_normalized_columns(self):
        self.write_json([
            {
                "Date": "2026-10-25",
                " Cases ": "100",
                "CASES": "200"
            }
        ])

        with self.assertRaisesRegex(
            ValueError,
            "Duplicate column names"
        ):
            load_data(self.file_path)

    def test_missing_file_raises_error(self):
        missing_path = (
            Path(self.temp_dir.name) / "missing.json"
        )

        with self.assertRaises(FileNotFoundError):
            load_data(missing_path)


class TestGetValues(unittest.TestCase):
    """Tests for retrieving numeric columns."""

    def setUp(self):
        self.df = pd.DataFrame({
            "Date": [
                datetime(2026, 10, 24),
                datetime(2026, 10, 25),
            ],
            "daily cases": [100, 200],
        })

    def test_retrieves_existing_field(self):
        result = get_values(self.df, "daily cases")

        np.testing.assert_array_equal(
            result,
            np.array([100, 200])
        )

    def test_field_matching_ignores_case_and_spaces(self):
        result = get_values(
            self.df,
            "  DAILY   CASES "
        )

        np.testing.assert_array_equal(
            result,
            np.array([100, 200])
        )

    def test_missing_field_returns_zeros(self):
        result = get_values(self.df, "Deaths")

        np.testing.assert_array_equal(
            result,
            np.array([0, 0])
        )


class TestGraphs(unittest.TestCase):
    """Tests for terminal graph output."""

    def setUp(self):
        self.df = pd.DataFrame({
            "Date": [
                datetime(2026, 10, 23),
                datetime(2026, 10, 24),
                datetime(2026, 10, 25),
            ],
            "daily cases": [100, 200, 300],
        })

    def test_star_graph_prints_title_and_values(self):
        output = StringIO()

        with redirect_stdout(output):
            star_graph(self.df, "daily cases")

        result = output.getvalue()

        self.assertIn("DAILY CASES", result)
        self.assertIn("20261023", result)
        self.assertIn("20261024", result)
        self.assertIn("20261025", result)
        self.assertIn("100", result)
        self.assertIn("200", result)
        self.assertIn("300", result)

    def test_star_graph_handles_zero_values(self):
        df = self.df.copy()
        df["daily cases"] = [0, 0, 0]

        output = StringIO()

        with redirect_stdout(output):
            star_graph(df, "daily cases")

        self.assertIn("DAILY CASES", output.getvalue())

    def test_trend_graph_prints_title_and_dates(self):
        output = StringIO()

        with redirect_stdout(output):
            trend_graph(
                self.df,
                "daily cases",
                height=5,
                width=10
            )

        result = output.getvalue()

        self.assertIn("DAILY CASES - TREND", result)
        self.assertIn("2026/10/23", result)
        self.assertIn("2026/10/25", result)
        self.assertIn("*", result)

    def test_trend_graph_handles_equal_values(self):
        df = self.df.copy()
        df["daily cases"] = [100, 100, 100]

        output = StringIO()

        with redirect_stdout(output):
            trend_graph(df, "daily cases")

        self.assertIn(
            "DAILY CASES - TREND",
            output.getvalue()
        )

    def test_trend_graph_limits_number_of_points(self):
        df = pd.DataFrame({
            "Date": pd.date_range(
                start="2026-01-01",
                periods=100
            ),
            "daily cases": np.arange(100),
        })

        output = StringIO()

        with redirect_stdout(output):
            trend_graph(
                df,
                "daily cases",
                height=5,
                width=10
            )

        result = output.getvalue()

        # Ten plotted points produce ten chart columns.
        chart_line = [
            line for line in result.splitlines()
            if line.startswith("      99 |")
        ]

        self.assertTrue(chart_line)

    def test_trend_graph_rejects_height_below_two(self):
        with self.assertRaisesRegex(
            ValueError,
            "height must be at least 2"
        ):
            trend_graph(self.df, "daily cases", height=1)

    def test_trend_graph_rejects_width_below_two(self):
        with self.assertRaisesRegex(
            ValueError,
            "width must be at least 2"
        ):
            trend_graph(self.df, "daily cases", width=1)

    def test_trend_graph_rejects_empty_dataframe(self):
        empty_df = pd.DataFrame({
            "Date": pd.Series(dtype="datetime64[ns]"),
            "daily cases": pd.Series(dtype="int64"),
        })

        with self.assertRaisesRegex(
            ValueError,
            "without data"
        ):
            trend_graph(empty_df, "daily cases")


if __name__ == "__main__":
    unittest.main(verbosity=2)