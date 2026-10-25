# Data-Visualization
Data Visualization

# COVID-19 Data Visualisation in Python

## 1. Project Overview

This project reads COVID-19 time-series data from a JSON file, cleans the data, sorts it by date, and presents the information using graphs.

The project explores three implementations:

1. **NumPy/pandas and Matplotlib version** — uses third-party libraries to create graphical charts and export cleaned data to CSV.
2. **Terminal-only version** — prints star-based bar graphs and trend graphs directly in the terminal, also without third-party libraries. This is the version covered by the unit tests written for the project.

All versions aim to make COVID-19 data easier to understand, compare, and analyse over time.

## 2. Input Data

The programs read data from a JSON file, normally named `covid.json`.

Each record represents one date and contains information such as:

- `Date`
- `Total Confirmed Cases`
- `Total Deaths`
- `Total Recovered`
- `Active Cases`
- `Daily Confirmed Cases`
- `Daily deaths`

An example record is:

```json
{
    "Date": "2020/03/05",
    "Total Confirmed Cases": 1,
    "Total Deaths": 0,
    "Total Recovered": 0,
    "Active Cases": 1,
    "Daily Confirmed Cases": 0,
    "Daily  deaths": 0
}
```

The source data may contain inconsistent column names, different date formats, numbers represented as strings, commas used as thousands separators, or extra spaces. The programs clean or standardise these values before visualising them.

The exact input requirements differ slightly between implementations, as explained below.

---

# Part A: Implementation Using pandas and Matplotlib

## 3. Overview

The library-based implementation uses:

- **pandas** to organise, clean, and manipulate tabular data.
- **Matplotlib** to create charts and save them as PNG images.
- Python's built-in `json` module to read JSON data.
- Python's built-in `re` module to remove unwanted characters from numeric strings.
- Python's built-in `argparse` module to process command-line arguments.
- `pathlib.Path` to handle file paths and output directories.

NumPy is a common dependency in this scientific Python ecosystem, but the uploaded script does not directly import or call NumPy. Its main data-processing library is pandas, and its charting library is Matplotlib.

### 3.1 Imports

```python
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
```

Each import has a particular responsibility.

| Import | Purpose |
|---|---|
| `argparse` | Reads command-line options such as the input file and output folder. |
| `json` | Loads the source JSON data. |
| `re` | Removes commas and whitespace from numeric strings. |
| `Path` | Represents file and directory paths. |
| `pandas` | Stores the records in a DataFrame and performs data transformations. |
| `matplotlib.pyplot` | Creates charts and saves them as images. |

### 3.2 Cleaning numeric values

The function `to_number(value)` converts values into numbers that can be plotted.

```python
def to_number(value):
    if value is None or value == "":
        return 0

    if isinstance(value, (int, float)):
        return value

    cleaned = re.sub(r"[,\s]", "", str(value))
    return float(cleaned) if cleaned else 0
```

How it works:

1. If the value is `None` or an empty string, return `0`.
2. If the value is already an integer or float, return it unchanged.
3. Convert other values to strings.
4. Use a regular expression to remove commas and whitespace.
5. Convert the cleaned string to a floating-point number.

For example:

| Input | Result |
|---|---:|
| `1354` | `1354` |
| `"1,354"` | `1354.0` |
| `"1 354"` | `1354.0` |
| `None` | `0` |
| `""` | `0` |

This is useful because data from external sources does not always use consistent numeric formatting.

One limitation is that malformed non-empty strings, such as `"unknown"`, can raise a `ValueError`. The function does not currently catch that error.

### 3.3 Loading data into a DataFrame

The `load_data(input_path)` function reads the JSON file and converts its records into a pandas DataFrame.

```python
with open(input_path, "r", encoding="utf-8") as file:
    records = json.load(file)

df = pd.DataFrame(records)
```

A DataFrame is a two-dimensional data structure similar to a spreadsheet. Each column represents a field, and each row represents one record.

The function then checks that the required columns exist:

```python
missing = [
    column for column in required
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required JSON field(s): {', '.join(missing)}"
    )
```

This validation prevents the program from silently proceeding when important fields are absent.

The function also parses the dates, removes rows with invalid dates, sorts records chronologically, and cleans the numeric columns.

```python
df["Date"] = pd.to_datetime(
    df["Date"],
    format="mixed",
    errors="coerce"
)

df = df.dropna(subset=["Date"]).sort_values("Date").copy()
```

Here:

- `pd.to_datetime()` converts date strings into date values.
- `errors="coerce"` turns invalid dates into missing values.
- `dropna()` removes records with missing dates.
- `sort_values("Date")` orders records from earliest to latest.
- `.copy()` creates a separate DataFrame for subsequent transformations.

The program also handles the inconsistent `Daily  deaths` column name by searching for a column whose stripped, lowercase name matches `daily deaths`.

### 3.4 Creating the charts

The `create_charts(df, output_dir)` function creates the visualisations.

It first ensures that the output directory exists:

```python
output_dir.mkdir(parents=True, exist_ok=True)
```

It then creates a dashboard with three subplots:

```python
fig, axes = plt.subplots(
    3, 1, figsize=(13, 13), sharex=True
)
```

The three subplots show:

**Chart 1: Cumulative totals**

Three lines display total confirmed cases, total recoveries, and total deaths over time.

This allows the reader to compare the cumulative totals and observe how they change.

**Chart 2: Daily confirmed cases**

A bar chart displays the number of newly confirmed cases recorded each day.

Unlike cumulative totals, daily counts can rise or fall from one day to the next.

**Chart 3: Active cases**

A line chart displays the number of active cases over time.

This helps show how the active caseload changes throughout the period.

The code uses methods such as `set_title()`, `set_ylabel()`, `legend()`, and `grid()` to add titles, axis labels, legends, and grid lines.

It saves the combined dashboard as:

```text
covid_dashboard.png
```

The program then creates a separate bar chart of daily deaths:

```text
daily_deaths.png
```

This separate chart makes daily deaths easier to inspect without the larger numerical scale of the other measures affecting the display.

### 3.5 Exporting the cleaned data

The pandas implementation saves the cleaned DataFrame to a CSV file:

```python
df.to_csv(csv_path, index=False)
```

The `index=False` argument prevents pandas from adding its DataFrame row index as an extra column.

The result is:

```text
covid_cleaned.csv
```

The cleaned file can be opened in spreadsheet software or used as input for another analysis.

### 3.6 Running the library-based version

Install the required third-party packages:

```bash
python -m pip install pandas matplotlib
```

Run the script:

```bash
python visualise_covid.py
```

Or provide an input file and output directory:

```bash
python visualise_covid.py --input covid.json --output-dir covid_charts
```

Expected output files:

```text
covid_charts/
├── covid_dashboard.png
├── daily_deaths.png
└── covid_cleaned.csv
```

The input file must satisfy the script's required-field checks.

---

# Part B: Standard-Library Version Using SVG

## 4. Overview

The SVG implementation creates graphs without pandas, NumPy, or Matplotlib.

It uses only modules included with Python:

```python
import argparse
import csv
import json
import os
from datetime import datetime
from html import escape
```

The key difference is that the program constructs SVG markup itself rather than relying on a charting library.

SVG stands for Scalable Vector Graphics. It describes graphical elements such as lines, text, and shapes in a text-based XML format. Modern web browsers can display SVG files directly.

### 4.1 Cleaning numeric values

The `clean_number(value)` function converts values such as `"61 927"` and `"61,927"` into integers.

```python
def clean_number(value):
    if value is None or str(value).strip() == "":
        return 0

    cleaned = str(value).replace(",", "")
    cleaned = "".join(cleaned.split())

    try:
        return int(float(cleaned))
    except ValueError:
        return 0
```

The function:

1. Returns zero for missing or empty values.
2. Removes commas.
3. Removes whitespace, including internal spaces.
4. Converts the value to a number.
5. Returns zero if the value cannot be converted.

Unlike the pandas implementation's `to_number()`, this function catches `ValueError` and returns zero for invalid non-empty numeric strings.

### 4.2 Finding column names

The `get_field(record, wanted_name)` function searches a dictionary for a requested field.

```python
def get_field(record, wanted_name):
    wanted = wanted_name.strip().lower()

    for key, value in record.items():
        if key.strip().lower() == wanted:
            return value

    return None
```

It ignores leading and trailing whitespace and differences in capitalisation.

For example, `"Date"` and `" date "` will match.

However, this implementation does not collapse repeated spaces inside a column name. The explicit lookup for `"Daily deaths"` handles the source's known extra-space inconsistency.

### 4.3 Parsing dates

The `parse_date(value)` function tries several date formats:

```python
for fmt in (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y"
):
    try:
        return datetime.strptime(
            str(value).strip(), fmt
        ).date()
    except ValueError:
        continue
```

If a format matches, the function returns a date object.

If none matches, it raises a `ValueError`.

The supported formats in this uploaded version do not include `"%Y/%m/%d"`. If the JSON file uses dates such as `"2020/03/05"`, that format must be added to the list.

### 4.4 Loading and sorting records

The `load_data(filename)` function:

1. Opens and reads the JSON file.
2. Accepts a list of records or a dictionary containing a `data` list.
3. Reads the date from each record.
4. Cleans the numeric fields.
5. Creates a consistent dictionary for each usable record.
6. Sorts the records by date.

The sorting operation is:

```python
rows.sort(key=lambda row: row["Date"])
```

The `key` argument tells Python to compare the `Date` value in each dictionary.

This is chronological sorting because the dates have already been converted into date objects.

### 4.5 Writing a cleaned CSV

The `write_cleaned_csv(rows, filename)` function uses Python's built-in CSV writer.

```python
writer = csv.DictWriter(
    file,
    fieldnames=fields
)
writer.writeheader()
```

It writes the field names first, then writes each record.

The date is converted to ISO format before writing:

```python
output["Date"] = row["Date"].isoformat()
```

For example, a date may be stored as `2020-03-05` in the output file.


# Part B: Terminal-Only Version

## 5. Overview

The terminal-only implementation is the version for which the accompanying unit tests were written.

It uses Python's standard library and prints the visualisations directly to the terminal rather than creating PNG or SVG files.

Its main imports are:

```python
import json
import sys
from datetime import datetime
```

It uses:

- `json` to read the data.
- `datetime` to parse and sort dates.
- `sys` to access command-line arguments and print errors to standard error.

The graphs are made from ordinary text characters, particularly the asterisk `*`.

### 5.1 Standardising column names

The `standardize_column(name)` function normalises a column name:

```python
def standardize_column(name):
    return " ".join(str(name).lower().split())
```

It converts the name to lowercase and collapses repeated whitespace.

For example:

```text
"Daily  Confirmed   Cases"
```

becomes:

```text
"daily confirmed cases"
```

This allows the program to find fields even if the source has inconsistent capitalisation or extra spaces.

### 5.2 Extracting values

The `extract_value(record, wanted_column)` function searches a record for a column using the normalised name.

```python
def extract_value(record, wanted_column):
    wanted = standardize_column(wanted_column)

    for column, value in record.items():
        if standardize_column(column) == wanted:
            return value

    return None
```

If a matching field is found, its value is returned. If it cannot be found, the function returns `None`.

### 5.3 Cleaning values

The `clean_value(value)` function handles numeric strings and missing values.

It removes commas and whitespace, converts valid values into integers, and returns zero for empty or invalid values.

For example:

| Input | Output |
|---|---:|
| `"1,234"` | `1234` |
| `"1 234"` | `1234` |
| `" 25 "` | `25` |
| `None` | `0` |
| `"invalid"` | `0` |

The function ensures that the graph calculations work with numeric values rather than a mixture of strings and numbers.

### 5.4 Standardising and sorting dates

The `standardize_date(value)` function attempts to parse a date using the supported formats:

```python
("%Y/%m/%d", "%Y-%m-%d", "%d/%m/%Y")
```

It returns a `datetime` object when a format matches, or `None` when no format matches.

The `get_date(item)` helper returns the first element of a tuple:

```python
def get_date(item):
    return item[0]
```

Each row contains a date and a record, so sorting can be performed without using a lambda function:

```python
rows.sort(key=get_date)
```

Python sorts the records from the earliest date to the latest date.

### 5.5 Loading the JSON data

The `load_data(filename)` function opens the input file and loads its contents.

It checks whether the JSON is a list or a dictionary containing a `data` list. It then processes the records, standardises dates, cleans numeric fields, skips records with invalid dates, and sorts the usable rows.

If the JSON structure is invalid or no usable records remain, the function raises an error.

This is important because the graph functions require valid, sorted data to work correctly.

### 5.6 The star bar graph

The `star_graph(rows, field)` function prints a horizontal bar for each date.

For each row, it extracts the selected field and calculates how many stars to print.

```python
stars = round(value / maximum * 40) if maximum else 0
```

The formula scales the values relative to the largest value in the dataset. The maximum value receives approximately 40 stars, while smaller values receive proportionally fewer stars.

For example:

```text
DAILY CONFIRMED CASES
-----------------------------------------------------------------
20200305 |  0
20200306 |  0
20200307 |  1
20200320 | * 52
20200326 | *** 218
20200612 | **************************************** 3,359
```

The date identifies the observation, the stars show its relative size, and the printed number provides the exact value.

Small positive values may receive no stars because the scaled result rounds to zero. This is a consequence of the chosen scale, not necessarily missing data.

### 5.7 The trend graph

The `trend_graph(rows, field, height=10, width=60)` function creates a text-based graph.

It:

1. Extracts the dates and values.
2. Reduces the number of plotted points if there are more than the specified width.
3. Finds the minimum and maximum values.
4. Creates a two-dimensional grid of spaces.
5. Calculates a vertical position for each value.
6. Places a star at the corresponding position.
7. Prints the grid with numeric labels and a date range.

The vertical axis represents the selected data values. Higher stars indicate larger values, while lower stars indicate smaller values.

Unlike the bar graph, the trend graph emphasises the shape of the data over time.

The current implementation displays stars as separate points rather than connecting them with lines.

### 5.8 Command-line arguments and error handling

The `main()` function accepts two optional positional arguments:

```python
filename = sys.argv[1] if len(sys.argv) > 1 else "covid.json"
field = sys.argv[2] if len(sys.argv) > 2 else "Daily Confirmed Cases"
```

The first argument selects the JSON file. The second selects the column to visualise.

For example:

```bash
python visualise_covid.py
```

uses `covid.json` and `Daily Confirmed Cases`.

To select another field:

```bash
python visualise_covid.py covid.json "Total Deaths"
```

The `try`/`except` block catches file and data errors, prints an error message to standard error, and exits with a non-zero status.

This makes failures easier to diagnose when running the program from a terminal.

---
