import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd


def standardize_column(name):
    """Normalize column names by removing extra spaces and lowercasing."""
    return " ".join(str(name).lower().split())


def clean_value(value):
    """
    Convert a value into an integer.

    Handles:
    - Missing values
    - Empty strings
    - Commas in numbers
    - Spaces within numbers
    - Decimal numbers
    - Invalid numeric strings
    """
    if value is None or pd.isna(value):
        return 0

    cleaned = "".join(str(value).replace(",", "").split())

    if not cleaned:
        return 0

    try:
        number = pd.to_numeric(cleaned, errors="coerce")

        if pd.isna(number) or not np.isfinite(number):
            return 0

        return int(float(number))

    except (ValueError, TypeError, OverflowError):
        return 0


def standardize_date(value):
    """
    Parse supported date formats into Python datetime objects.

    Supported formats:
    - YYYY/MM/DD
    - YYYY-MM-DD
    - DD/MM/YYYY
    """
    if not isinstance(value, str):
        return None

    value = value.strip()

    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt)

        except ValueError:
            continue

    return None


def load_data(filename):
    """
    Load JSON records into a cleaned, chronologically sorted DataFrame.

    The JSON file can contain either:
        [...]
    or:
        {"data": [...]}

    Returns:
        A pandas DataFrame with a standardized Date column.

    Raises:
        ValueError if the JSON structure is invalid or no valid
        records remain after date validation.
    """
    with open(filename, encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = data.get("data", [])

    if not isinstance(data, list):
        raise ValueError("Expected a JSON list.")

    if not data:
        raise ValueError("No valid records found.")

    # Convert JSON records into a DataFrame.
    df = pd.DataFrame(data)

    if df.empty:
        raise ValueError("No valid records found.")

    # Normalize all column names.
    df.columns = [standardize_column(column) for column in df.columns]

    # Reject ambiguous duplicate columns after normalization.
    if df.columns.duplicated().any():
        raise ValueError(
            "Duplicate column names found after standardization."
        )

    if "date" not in df.columns:
        raise ValueError("No valid records found.")

    # Parse dates, discarding records with invalid dates.
    df["date"] = df["date"].apply(standardize_date)

    df = df.dropna(subset=["date"]).copy()

    if df.empty:
        raise ValueError("No valid records found.")

    # Sort records by date.
    df = df.sort_values("date").reset_index(drop=True)

    # Clean numeric columns while preserving the date column.
    for column in df.columns:
        if column != "date":
            df[column] = df[column].apply(clean_value).astype("int64")

    # Keep the original program's public-facing Date column name.
    df = df.rename(columns={"date": "Date"})

    return df


def get_values(df, field):
    """
    Retrieve numeric values for a field.

    Column names are matched without regard to case or extra spaces.
    Missing fields produce zeros, matching the original behavior.
    """
    wanted = standardize_column(field)

    matching_columns = [
        column
        for column in df.columns
        if standardize_column(column) == wanted
    ]

    if not matching_columns:
        return np.zeros(len(df), dtype=int)

    return (
        pd.to_numeric(df[matching_columns[0]], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=int)
    )


def star_graph(df, field):
    """Print a bar graph using stars to represent numeric values."""
    values = get_values(df, field)
    maximum = values.max() if len(values) else 0

    print(f"\n{field.upper()}")
    print("-" * 65)

    for date, value in zip(df["Date"], values):
        stars = round(value / maximum * 40) if maximum else 0

        print(f"{date:%Y%m%d} | {'*' * stars} {value:,}")


def trend_graph(df, field, height=10, width=60):
    """Print a simple terminal-based trend graph."""
    if height < 2:
        raise ValueError("Chart height must be at least 2.")

    if width < 2:
        raise ValueError("Chart width must be at least 2.")

    if df.empty:
        raise ValueError("Cannot draw a trend graph without data.")

    values = get_values(df, field)
    dates = df["Date"].tolist()

    # Select evenly spaced points if there are too many records.
    if len(values) > width:
        indices = np.rint(
            np.linspace(0, len(values) - 1, num=width)
        ).astype(int)

        values = values[indices]
        dates = [dates[index] for index in indices]

    low = int(values.min())
    high = int(values.max())

    # Avoid division by zero when all values are equal.
    if low == high:
        high = low + 1

    chart = np.full((height, len(values)), " ", dtype="<U1")

    # Calculate the vertical position of every point.
    positions = np.rint(
        (high - values) / (high - low) * (height - 1)
    ).astype(int)

    for x, y in enumerate(positions):
        chart[y, x] = "*"

    print(f"\n{field.upper()} - TREND")

    for y, line in enumerate(chart):
        label = round(
            high - (high - low) * y / (height - 1)
        )

        print(f"{label:>7} | {''.join(line)}")

    print("          +" + "-" * len(values))

    print(
        f"           {dates[0]:%Y/%m/%d}"
        f"  to  {dates[-1]:%Y/%m/%d}"
    )


def main():
    """Run the analysis from the command line."""
    filename = (
        sys.argv[1] if len(sys.argv) > 1 else "covid.json"
    )

    field = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "Daily Confirmed Cases"
    )

    try:
        df = load_data(filename)

        star_graph(df, field)
        trend_graph(df, field)

    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()