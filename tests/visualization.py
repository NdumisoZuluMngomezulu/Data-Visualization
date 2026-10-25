
import json
import os
import tempfile
import unittest
from datetime import datetime
from io import StringIO
from unittest.mock import patch

import visualization as covid


class TestStandardizeColumn(unittest.TestCase):

    def test_lowercases_column_name(self):
        self.assertEqual(
            covid.standardize_column("DATE"),
            "date"
        )

    def test_removes_extra_spaces(self):
        self.assertEqual(
            covid.standardize_column(" Daily   Confirmed  Cases "),
            "daily confirmed cases"
        )

    def test_handles_non_string_column_name(self):
        self.assertEqual(
            covid.standardize_column(123),
            "123"
        )


class TestExtractValue(unittest.TestCase):

    def test_extracts_existing_column(self):
        record = {"Date": "2020/03/05"}

        self.assertEqual(
            covid.extract_value(record, "Date"),
            "2020/03/05"
        )

    def test_ignores_case_differences(self):
        record = {"DAILY CONFIRMED CASES": 25}

        self.assertEqual(
            covid.extract_value(record, "Daily Confirmed Cases"),
            25
        )

    def test_ignores_extra_spaces(self):
        record = {"Daily  Confirmed   Cases": 25}

        self.assertEqual(
            covid.extract_value(record, "Daily Confirmed Cases"),
            25
        )

    def test_returns_none_for_missing_column(self):
        record = {"Date": "2020/03/05"}

        self.assertIsNone(
            covid.extract_value(record, "Total Deaths")
        )


class TestCleanValue(unittest.TestCase):

    def test_cleans_integer(self):
        self.assertEqual(covid.clean_value(100), 100)

    def test_cleans_number_with_commas(self):
        self.assertEqual(covid.clean_value("1,234"), 1234)

    def test_cleans_number_with_spaces(self):
        self.assertEqual(covid.clean_value(" 1 234 "), 1234)

    def test_cleans_decimal_string(self):
        self.assertEqual(covid.clean_value("12.9"), 12)

    def test_empty_string_returns_zero(self):
        self.assertEqual(covid.clean_value(""), 0)

    def test_whitespace_returns_zero(self):
        self.assertEqual(covid.clean_value("   "), 0)

    def test_none_returns_zero(self):
        self.assertEqual(covid.clean_value(None), 0)

    def test_invalid_number_returns_zero(self):
        self.assertEqual(covid.clean_value("not a number"), 0)


class TestStandardizeDate(unittest.TestCase):

    def test_parses_slash_separated_date(self):
        self.assertEqual(
            covid.standardize_date("2020/03/05"),
            datetime(2020, 3, 5)
        )

    def test_parses_hyphen_separated_date(self):
        self.assertEqual(
            covid.standardize_date("2020-03-05"),
            datetime(2020, 3, 5)
        )

    def test_parses_day_first_date(self):
        self.assertEqual(
            covid.standardize_date("05/03/2020"),
            datetime(2020, 3, 5)
        )

    def test_rejects_invalid_date(self):
        self.assertIsNone(
            covid.standardize_date("not a date")
        )

    def test_rejects_none(self):
        self.assertIsNone(
            covid.standardize_date(None)
        )


class TestGetDate(unittest.TestCase):

    def test_returns_first_tuple_element(self):
        item = (
            datetime(2020, 3, 5),
            {"daily confirmed cases": 10}
        )

        self.assertEqual(
            covid.get_date(item),
            datetime(2020, 3, 5)
        )


class TestLoadData(unittest.TestCase):

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            delete=False
        )
        self.filename = self.temp_file.name
        self.temp_file.close()

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def write_json(self, data):
        with open(self.filename, "w", encoding="utf-8") as file:
            json.dump(data, file)

    def test_loads_json_records(self):
        self.write_json([
            {
                "Date": "2020/03/05",
                "Daily Confirmed Cases": "1,000"
            }
        ])

        rows = covid.load_data(self.filename)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], datetime(2020, 3, 5))
        self.assertEqual(
            rows[0][1]["daily confirmed cases"],
            1000
        )

    def test_sorts_records_chronologically(self):
        self.write_json([
            {
                "Date": "2020/03/07",
                "Daily Confirmed Cases": 3
            },
            {
                "Date": "2020/03/05",
                "Daily Confirmed Cases": 1
            },
            {
                "Date": "2020/03/06",
                "Daily Confirmed Cases": 2
            }
        ])

        rows = covid.load_data(self.filename)

        dates = [date for date, record in rows]

        self.assertEqual(
            dates,
            [
                datetime(2020, 3, 5),
                datetime(2020, 3, 6),
                datetime(2020, 3, 7)
            ]
        )

    def test_accepts_json_data_wrapper(self):
        self.write_json({
            "data": [
                {
                    "Date": "2020/03/05",
                    "Daily Confirmed Cases": 10
                }
            ]
        })

        rows = covid.load_data(self.filename)

        self.assertEqual(len(rows), 1)

    def test_skips_records_with_invalid_dates(self):
        self.write_json([
            {
                "Date": "invalid date",
                "Daily Confirmed Cases": 10
            },
            {
                "Date": "2020/03/05",
                "Daily Confirmed Cases": 20
            }
        ])

        rows = covid.load_data(self.filename)

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0][1]["daily confirmed cases"],
            20
        )

    def test_raises_error_when_no_valid_records_exist(self):
        self.write_json([
            {
                "Date": "invalid date",
                "Daily Confirmed Cases": 10
            }
        ])

        with self.assertRaisesRegex(
            ValueError, "No valid records found"
        ):
            covid.load_data(self.filename)

    def test_raises_error_for_non_list_data(self):
        self.write_json({"data": "not a list"})

        with self.assertRaisesRegex(
            ValueError, "Expected a JSON list"
        ):
            covid.load_data(self.filename)

    def test_raises_error_for_invalid_json(self):
        with open(self.filename, "w", encoding="utf-8") as file:
            file.write("{invalid json")

        with self.assertRaises(json.JSONDecodeError):
            covid.load_data(self.filename)

    def test_raises_error_for_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            covid.load_data(self.filename + ".missing")


class TestStarGraph(unittest.TestCase):

    def setUp(self):
        self.rows = [
            (
                datetime(2020, 3, 5),
                {"daily confirmed cases": 0}
            ),
            (
                datetime(2020, 3, 6),
                {"daily confirmed cases": 5}
            ),
            (
                datetime(2020, 3, 7),
                {"daily confirmed cases": 10}
            )
        ]

    def test_prints_heading(self):
        output = StringIO()

        with patch("sys.stdout", output):
            covid.star_graph(self.rows, "Daily Confirmed Cases")

        self.assertIn("DAILY CONFIRMED CASES", output.getvalue())

    def test_prints_dates_and_values(self):
        output = StringIO()

        with patch("sys.stdout", output):
            covid.star_graph(self.rows, "Daily Confirmed Cases")

        result = output.getvalue()

        self.assertIn("20200305", result)
        self.assertIn("20200306", result)
        self.assertIn("20200307", result)
        self.assertIn("10", result)

    def test_scales_stars_relative_to_maximum(self):
        output = StringIO()

        with patch("sys.stdout", output):
            covid.star_graph(self.rows, "Daily Confirmed Cases")

        lines = output.getvalue().splitlines()

        line_for_five = next(
            line for line in lines if "20200306" in line
        )
        line_for_ten = next(
            line for line in lines if "20200307" in line
        )

        self.assertEqual(line_for_five.count("*"), 20)
        self.assertEqual(line_for_ten.count("*"), 40)

    def test_zero_values_do_not_produce_stars(self):
        output = StringIO()

        with patch("sys.stdout", output):
            covid.star_graph(self.rows, "Daily Confirmed Cases")

        line_for_zero = next(
            line for line in output.getvalue().splitlines()
            if "20200305" in line
        )

        self.assertEqual(line_for_zero.count("*"), 0)


class TestTrendGraph(unittest.TestCase):

    def setUp(self):
        self.rows = [
            (
                datetime(2020, 3, 5),
                {"daily confirmed cases": 0}
            ),
            (
                datetime(2020, 3, 6),
                {"daily confirmed cases": 5}
            ),
            (
                datetime(2020, 3, 7),
                {"daily confirmed cases": 10}
            )
        ]

    def test_prints_trend_heading(self):
        output = StringIO()

        with patch("sys.stdout", output):
            covid.trend_graph(
                self.rows, "Daily Confirmed Cases"
            )

        self.assertIn(
            "DAILY CONFIRMED CASES - TREND",
            output.getvalue()
        )

    def test_prints_date_range(self):
        output = StringIO()

        with patch("sys.stdout", output):
            covid.trend_graph(
                self.rows, "Daily Confirmed Cases"
            )

        result = output.getvalue()

        self.assertIn("2020/03/05", result)
        self.assertIn("2020/03/07", result)

    def test_plots_one_star_per_data_point(self):
        output = StringIO()

        with patch("sys.stdout", output):
            covid.trend_graph(
                self.rows, "Daily Confirmed Cases"
            )

        self.assertEqual(output.getvalue().count("*"), 3)

    def test_limits_chart_width(self):
        output = StringIO()

        with patch("sys.stdout", output):
            covid.trend_graph(
                self.rows,
                "Daily Confirmed Cases",
                height=5,
                width=2
            )

        chart_lines = [
            line for line in output.getvalue().splitlines()
            if "|" in line
        ]

        self.assertTrue(
            all(len(line.split("|", 1)[1]) <= 2
                for line in chart_lines)
        )

    def test_handles_equal_values(self):
        rows = [
            (
                datetime(2020, 3, 5),
                {"daily confirmed cases": 5}
            ),
            (
                datetime(2020, 3, 6),
                {"daily confirmed cases": 5}
            )
        ]

        output = StringIO()

        with patch("sys.stdout", output):
            covid.trend_graph(rows, "Daily Confirmed Cases")

        self.assertIn("*", output.getvalue())


class TestMain(unittest.TestCase):

    def test_main_runs_both_graphs(self):
        output = StringIO()

        rows = [
            (
                datetime(2020, 3, 5),
                {"daily confirmed cases": 10}
            )
        ]

        with patch.object(covid, "load_data", return_value=rows), \
             patch.object(covid, "star_graph") as star_mock, \
             patch.object(covid, "trend_graph") as trend_mock, \
             patch("sys.argv", ["visualise_covid.py"]), \
             patch("sys.stdout", output):

            covid.main()

        star_mock.assert_called_once_with(
            rows, "Daily Confirmed Cases"
        )
        trend_mock.assert_called_once_with(
            rows, "Daily Confirmed Cases"
        )

    def test_main_accepts_filename_and_field_arguments(self):
        rows = [
            (
                datetime(2020, 3, 5),
                {"daily deaths": 2}
            )
        ]

        with patch.object(covid, "load_data", return_value=rows) as load_mock, \
             patch.object(covid, "star_graph") as star_mock, \
             patch.object(covid, "trend_graph") as trend_mock, \
             patch("sys.argv", [
                 "visualise_covid.py",
                 "mydata.json",
                 "Daily deaths"
             ]):

            covid.main()

        load_mock.assert_called_once_with("mydata.json")
        star_mock.assert_called_once_with(rows, "Daily deaths")
        trend_mock.assert_called_once_with(rows, "Daily deaths")

    def test_main_reports_file_error(self):
        errors = StringIO()

        with patch.object(
            covid, "load_data",
            side_effect=FileNotFoundError("File not found")
        ), patch("sys.argv", ["visualise_covid.py"]), \
             patch("sys.stderr", errors):

            with self.assertRaises(SystemExit) as error:
                covid.main()

        self.assertEqual(error.exception.code, 1)
        self.assertIn("Error:", errors.getvalue())
        self.assertIn("File not found", errors.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
