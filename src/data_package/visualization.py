# Data cleaning and visualization programme without the 
# use of libraries

import json
import os
import sys
from datetime import datetime

def standardize_column(name):
    #to handle double spaces and different cases in column names
    return " ".join(str(name).lower().split())

def extract_value(record, wanted_column):
    #returns the value based on the column name
    wanted = standardize_column(wanted_column)

    for column, value in record.items():
        if (standardize_column(column) == wanted):
            return value

    return None

def clean_value(value):
    #cleans up edge cases in values from strings, spaces, commas and null values
    if value is None or str(value).strip() == "":
        return 0

    first_clean = str(value).replace(",","")
    cleaned = "".join(first_clean.split())

    try:
        return int(float(cleaned))
    except ValueError:
        return 0

def standardize_date(value):
    #handling different date formates and standardizing them
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            pass

    return None

def get_date(item):
    #helper method to sort by date
    return item[0]

def load_data(filename):
    #opens file records
    with open(filename, encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = data.get("data", [])

    if not isinstance(data, list):
        raise ValueError("Expected a JSON list.")

    rows = []

    for record in data:
        date = standardize_date(extract_value(record, "Date"))

        if date:
            cleaned_record = {}

            for column, value in record.items():
                clean_column = standardize_column(column)

                if clean_column == "date":
                    cleaned_record["Date"] = value
                else:
                    cleaned_record[clean_column] = clean_value(value)

            rows.append((date, cleaned_record))

    rows.sort(key=get_date)

    if not rows:
        raise ValueError("No valid records found.")

    return rows

def star_graph(rows, field):
    #simple bar graph visualization making use of stars as the pattern
    values = [
        clean_value(extract_value(record, field)) for _, record in rows
    ]
    maximum = max(values, default=0)

    print("\n" + field.upper())
    print("-" * 65)

    for (date, _), value in zip(rows, values):
        stars = round(value / maximum * 40) if maximum else 0
        print(f"{date:%Y%m%d} | {'*' * stars} {value:,}")


def trend_graph(rows, field, height=10, width=60):
    #a graph to demonstrate the trends or occurences
    points = [
        (date, clean_value(extract_value(record, field)))
        for date, record in rows
    ]

    #keeps the chart within a readable terminal width
    if len(points) > width:
        index = [
            round(i * (len(points) - 1) / (width - 1))
            for i in range(width)
        ]

        points = [points[i] for i in index]

    values = [value for _, value in points]
    low = min(values)
    high = max(values)

    if (low == high):
        high = low + 1

    chart = [[" " for _ in points] for _ in range(height)]

    for x, value in enumerate(values):
        y = round((high - value) / (high - low) * (height - 1))
        chart[y][x] = "*"

    print(f"\n{field.upper()} - TREND")
    for y, line in enumerate(chart):
        label = round(high - (high - low) * y / (height - 1))
        print(f"{label:>7} | {''.join(line)}")

    print("          +" + "-" * len(points))
    print(f"           {points[0][0]:%Y/%m/%d}  to  {points[-1][0]:%Y/%m/%d}")
    print

def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "covid.json"
    field = sys.argv[2] if len(sys.argv) > 2 else "Daily Confirmed Cases"

    # Fix: If the file path is relative (like 'covid.json'), 
    # force it to look inside the script's directory.
    if not os.path.isabs(filename):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(script_dir, filename)

    try:
        rows = load_data(filename)
        star_graph(rows, field)
        trend_graph(rows, field)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()