
import pandas as pd
import pytest
from sales_analysis import load_data, prepare_data, analyze_sales, validate_results, build_quality_report


@pytest.fixture
def prepare_df():
    df_raw = load_data()
    df = prepare_data(df_raw)
    return df

def test_prepare_data_does_not_modify_raw_data():
    # Arrange: create the original dataset
    df_raw = load_data()
    original_df = df_raw.copy(deep=True)

    # Act: prepare the data
    prepare_data(df_raw)

    # Assert: the raw dataset remains unchanged
    pd.testing.assert_frame_equal(df_raw, original_df)


def test_prepare_data_converts_invalid_dates_to_nat(prepare_df):
    df = prepare_df

    assert pd.isna(df.loc[6, "date"])
def test_prepare_data_total_price(prepare_df):
    df = prepare_df
    assert df.loc[0,"total_price"] == 1600
def test_prepare_data_invalid_total_price(prepare_df):
    df = prepare_df
    assert df.loc[df["product"] == "Monitor", "total_price"].isna().any()
def test_analyze_sales_invalid_total_revenue(prepare_df):
    df = prepare_df
    result = analyze_sales(df)
    assert pd.isna(result.loc["Monitor","total_revenue"])
def test_analyze_sales_revenue_completeness(prepare_df):
    df = prepare_df
    result = analyze_sales(df)
    assert result.loc["Mouse","revenue_completeness"]=="partial"
def test_analyze_sales_revenue_completeness_missing(prepare_df):
    result = analyze_sales(prepare_df)
    assert result.loc["Monitor","revenue_completeness"]=="missing"
def test_analyze_sales_row_count(prepare_df):
    result = analyze_sales(prepare_df)
    assert result.loc["Laptop","row_count"]==3
def test_analyze_sales_total_revenue(prepare_df):
    result = analyze_sales(prepare_df)
    assert result.loc["Laptop","total_revenue"]==3200
def test_validate_result_revenue_totals_match(prepare_df):
    result = analyze_sales(prepare_df)
    validation_report = validate_results(prepare_df,result)
    assert validation_report["revenue_totals_match"]
def test_validate_result_detects_revenue_mismatch(prepare_df):
    result = analyze_sales(prepare_df)
    result.loc["Laptop","total_revenue"]=3300
    validation_report = validate_results(prepare_df,result)
    assert not(validation_report["revenue_totals_match"])
def test_build_quality_report():
    df_row=load_data()
    my_df=df_row.copy()
    my_df.loc[0,"date"]=None
    df=prepare_data(my_df)
    report=build_quality_report(my_df,df)
    assert report["missing_date_rows"]==1 
    assert report["invalid_date_rows"]==1