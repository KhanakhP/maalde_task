import pandas as pd

from .config import DATA_DIR, PROCESSED_DATA_PATH, RAW_EXCEL_PATH, SHEET_NAME


def prepare_data():
    df = pd.read_excel(RAW_EXCEL_PATH, sheet_name=SHEET_NAME)
    df.columns = df.columns.str.strip().str.lower()

    required_columns = {"code", "qty", "rate"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in sales sheet: {sorted(missing_columns)}")

    df = df.dropna(subset=["code", "qty", "rate"]).copy()
    df["code"] = df["code"].astype(int).astype(str).str.zfill(8)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna(subset=["qty", "rate"])

    df["sales_value"] = df["qty"] * df["rate"]

    df_grouped = (
        df.groupby("code", as_index=False)
        .agg(total_qty=("qty", "sum"), total_sales_value=("sales_value", "sum"))
    )
    df_grouped["avg_rate"] = (
        df_grouped["total_sales_value"] / df_grouped["total_qty"]
    ).round(2)
    df_grouped = df_grouped[["code", "total_qty", "avg_rate"]]

    DATA_DIR.mkdir(exist_ok=True)
    df_grouped.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Prepared {len(df_grouped)} products at {PROCESSED_DATA_PATH}")
