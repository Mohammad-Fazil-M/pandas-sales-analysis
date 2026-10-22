
import pandas as pd
import numpy as np
from pathlib import Path

def load_data():
    """Create the initial retail sales dataset."""
    data = {
        "order_id": [101, 102, 103, 104, 104, 106, 107, 108],
        "product": [
            "Laptop", " Mouse ", "keyboard", "Laptop",
            "Laptop", None, "Mouse", "Monitor"
        ],
        "category": [
            "Electronics", "Accessories", "Accessories", "Electronics",
            "Electronics", "Accessories", None, "Electronics"
        ],
        "quantity": [2, 5, 3, 1, 1, 4, np.nan, 2],
        "unit_price": [800, 20, 35, 800, 800, 20, 20, np.nan],
        "date": [
            "2026-09-01", "2026-09-02", "2026-09-02",
            "2026-09-03", "2026-09-03", "2026-09-04",
            "invalid_date", "2026-09-05"
        ],
        "city": [
            "Tehran", "tehran", "Shiraz", "Tehran",
            "Tehran", "Tabriz", "Shiraz", " Tehran "
        ],
    }

    return pd.DataFrame(data)


def prepare_data(df_raw):
    """Clean text fields and prepare dates and revenue."""
    df = df_raw.copy()

    # Standardize text values while preserving missing values.
    df["product"] = df["product"].str.strip().str.title()
    df["city"] = df["city"].str.strip().str.title()

    # Convert invalid date strings to NaT.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Calculate revenue for each row.
    df["total_price"] = df["quantity"] * df["unit_price"]

    return df


def build_quality_report(df_raw, df):
    """Summarize important data quality issues."""
    report = pd.Series({
        "raw_rows": len(df_raw),
        "missing_product_rows": df["product"].isna().sum(),
        "missing_category_rows": df["category"].isna().sum(),
        "missing_quantity_rows": df["quantity"].isna().sum(),
        "missing_unit_price_rows": df["unit_price"].isna().sum(),
        "invalid_date_rows": df["date"].isna().sum(),
        "missing_total_price_rows": df["total_price"].isna().sum(),
        "duplicate_order_id_rows": df.duplicated(
            subset=["order_id"], keep=False
        ).sum(),
        "recorded_revenue": df["total_price"].sum(min_count=1),
    })

    return report


def analyze_sales(df):
    """Summarize sales by product and flag incomplete revenue."""
    # Use a reporting label for missing product names.
    df_report = df.assign(
        product_report=df["product"].fillna("Unknown product")
    )

    result = df_report.groupby("product_report").agg(
        row_count=("product_report", "size"),
        total_quantity=("quantity", "sum"),
        total_revenue=("total_price", "sum"),
        valid_revenue_count=("total_price", "count"),
    )

    # Do not report zero revenue when no row has a valid revenue.
    result.loc[
        result["valid_revenue_count"] == 0,
        "total_revenue"
    ] = np.nan

    # Label each product's revenue completeness.
    result["revenue_completeness"] = np.select(
        [
            result["valid_revenue_count"] == result["row_count"],
            result["valid_revenue_count"] == 0,
        ],
        [
            "complete",
            "missing",
        ],
        default="partial",
    )

    result = result.sort_values(
        "total_revenue",
        ascending=False,
        na_position="last",
    )

    return result
    
def validate_results(df, sales_report):
    """Validate row-level revenue calculations and report totals."""
    expected_revenue = df["quantity"] * df["unit_price"]

    row_check = (
        df["total_price"].eq(expected_revenue)
        | (df["total_price"].isna() & expected_revenue.isna())
    )

    raw_revenue = df["total_price"].sum(min_count=1)
    report_revenue = sales_report["total_revenue"].sum(min_count=1)

    revenue_matches = np.isclose(
        raw_revenue,
        report_revenue,
        equal_nan=True,
    )

    validation_report = pd.Series({
        "row_calculations_valid": row_check.all(),
        "rows_with_calculation_errors": (~row_check).sum(),
        "recorded_revenue": raw_revenue,
        "product_report_revenue": report_revenue,
        "revenue_totals_match": revenue_matches,
    })

    return validation_report

def save_reports(df, sales_report, quality_report):
    """Save analysis reports as CSV files."""
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    sales_report.to_csv(
        output_dir / "sales_by_product.csv",
        index=True,
    )

    quality_report.to_csv(
        output_dir / "data_quality_report.csv",
        header=["value"],
    )

    print(f"\nReports saved to: {output_dir.resolve()}")

def main():
    df_raw = load_data()
    df = prepare_data(df_raw)

    print("Initial dataset shape:", df_raw.shape)

    print("\nMissing values in raw data:")
    print(df_raw.isna().sum())

    print("\nPrepared data:")
    print(df.to_string(index=False))

    print("\nMissing values after preparation:")
    print(df.isna().sum())
    quality_report = build_quality_report(df_raw, df)

    print("\nData quality report:")
    print(quality_report.to_string()) 

    sales_report = analyze_sales(df)

    print("\nSales analysis by product:")
    print(sales_report.to_string())

    print("\nValidation report:")
    print(validate_results(df, sales_report).to_string())
    save_reports(df, sales_report, quality_report)
    


if __name__ == "__main__":
    main()
