#!/usr/bin/env python3
"""
Usage:
    cd into src, then cn into data-package
    python visualise_covid.py
    python visualise_covid.py --input covid.json --output-dir charts

Install dependencies:
    python -m pip install pandas matplotlib
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def to_number(value):
    """Convert values such as 61927, '61 927', or '1,354' into numbers."""
    if value is None or value == "":
        return 0
    if isinstance(value, (int, float)):
        return value
    cleaned = re.sub(r"[,\s]", "", str(value))
    return float(cleaned) if cleaned else 0


def load_data(input_path):
    with open(input_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list) or not records:
        raise ValueError("Expected a non-empty JSON array of daily records.")

    df = pd.DataFrame(records)

    required = [
        "Date",
        "Total Confirmed Cases",
        "Total Deaths",
        "Total Recovered",
        "Active Cases",
        "Daily Confirmed Cases",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required JSON field(s): {', '.join(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").copy()

    numeric_columns = [
        "Total Confirmed Cases",
        "Total Deaths",
        "Total Recovered",
        "Active Cases",
        "Daily Confirmed Cases",
    ]
    # The source uses spaces as thousands separators in some string values.
    for column in numeric_columns:
        df[column] = df[column].map(to_number)

    # Handle the inconsistent extra space in the source field name.
    daily_deaths_column = next(
        (c for c in df.columns if c.strip().lower() == "daily deaths"), None
    )
    if daily_deaths_column:
        df["Daily Deaths"] = df[daily_deaths_column].map(to_number)
    else:
        df["Daily Deaths"] = 0

    return df


def create_charts(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = df["Date"]

    # One dashboard with three useful views.
    fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)
    fig.suptitle("COVID-19 Data Overview", fontsize=18, fontweight="bold")

    axes[0].plot(dates, df["Total Confirmed Cases"], label="Confirmed cases", linewidth=2)
    axes[0].plot(dates, df["Total Recovered"], label="Recovered", linewidth=2)
    axes[0].plot(dates, df["Total Deaths"], label="Deaths", linewidth=2)
    axes[0].set_title("Cumulative totals")
    axes[0].set_ylabel("People")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(dates, df["Daily Confirmed Cases"], width=0.8)
    axes[1].set_title("Daily confirmed cases")
    axes[1].set_ylabel("New cases")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].plot(dates, df["Active Cases"], label="Active cases", linewidth=2)
    axes[2].set_title("Active cases over time")
    axes[2].set_ylabel("People")
    axes[2].set_xlabel("Date")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    dashboard_path = output_dir / "covid_dashboard.png"
    fig.savefig(dashboard_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Separate daily-deaths chart, useful because deaths have a different scale.
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(dates, df["Daily Deaths"], width=0.8)
    ax.set_title("Daily reported deaths")
    ax.set_xlabel("Date")
    ax.set_ylabel("Deaths")
    ax.grid(True, axis="y", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    deaths_path = output_dir / "daily_deaths.png"
    fig.savefig(deaths_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Export the cleaned data for reuse in other tools.
    csv_path = output_dir / "covid_cleaned.csv"
    df.to_csv(csv_path, index=False)

    print(f"Read {len(df)} daily records from {dates.min().date()} to {dates.max().date()}.")
    print(f"Dashboard saved to: {dashboard_path}")
    print(f"Daily deaths chart saved to: {deaths_path}")
    print(f"Cleaned CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot COVID-19 data from a JSON file.")
    parser.add_argument("--input", default="covid.json", help="Path to the input JSON file")
    parser.add_argument("--output-dir", default="covid_charts", help="Folder for charts and cleaned CSV")
    args = parser.parse_args()

    df = load_data(Path(args.input))
    create_charts(df, Path(args.output_dir))


if __name__ == "__main__":
    main()
