from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = "ecommerce_sales_clean.csv"
OUTPUT_DIR = Path("output_etl")
OUTPUT_FILE = OUTPUT_DIR / "sales_etl_output.csv"
OUTPUT_DIR.mkdir(exist_ok=True)


def resolve_input_file(file_name: str) -> Path:
    """
    Retourne le chemin du CSV. Si le nom exact est absent, prend
    le premier fichier compatible trouve dans le dossier courant.
    """
    candidate = Path(file_name)
    if candidate.exists():
        return candidate

    matches = sorted(Path(".").glob("ecommerce_sales_clean*.csv"))
    if matches:
        print(f"Fichier introuvable: {file_name}. Utilisation de {matches[0].name}.")
        return matches[0]

    raise FileNotFoundError(f"Impossible de trouver le fichier CSV: {file_name}")


def read_csv_with_fallback(file_path: Path, **kwargs) -> pd.DataFrame:
    for encoding in ("utf-8", "latin1"):
        try:
            return pd.read_csv(file_path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError("csv", b"", 0, 1, "encodage non supporte")


def extract_data(file_path: str | Path) -> pd.DataFrame:
    """
    Lit le fichier CSV et tente de corriger le cas ou tout le contenu
    a ete charge dans une seule colonne.
    """
    resolved_path = resolve_input_file(str(file_path))
    df = read_csv_with_fallback(resolved_path, sep=",")

    if df.shape[1] == 1:
        print("Le fichier semble charge en une seule colonne. Tentative de decoupage...")
        raw = read_csv_with_fallback(resolved_path, header=None)
        split_df = raw[0].astype(str).str.split(",", expand=True)
        split_df.columns = split_df.iloc[0]
        df = split_df.iloc[1:].reset_index(drop=True)

    print("Extraction terminee")
    print(f"Dimensions initiales : {df.shape}")
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("%", "Percent", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Order_Date" in df.columns:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

    numeric_columns = [
        "Year",
        "Month",
        "Unit_Price",
        "Quantity",
        "Discount",
        "Revenue",
        "Cost",
        "Profit",
        "Profit_Margin_Percent",
        "Shipping_Cost",
        "Shipping_Days",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    object_cols = df.select_dtypes(include="object").columns

    for col in object_cols:
        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    essential_cols = [col for col in ["Order_ID", "Order_Date"] if col in df.columns]
    if essential_cols:
        df = df.dropna(subset=essential_cols)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(0)

    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].fillna("Unknown")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)

    if "Order_ID" in df.columns:
        df = df.drop_duplicates(subset=["Order_ID"])
    else:
        df = df.drop_duplicates()

    after = len(df)
    print(f"Doublons supprimes : {before - after}")
    return df


def validate_business_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Is_Valid_Quantity"] = True
    df["Is_Valid_Unit_Price"] = True
    df["Is_Valid_Discount"] = True
    df["Is_Valid_Revenue"] = True
    df["Is_Valid_Profit"] = True

    if "Quantity" in df.columns:
        df["Is_Valid_Quantity"] = df["Quantity"] >= 0

    if "Unit_Price" in df.columns:
        df["Is_Valid_Unit_Price"] = df["Unit_Price"] >= 0

    if "Discount" in df.columns:
        df["Is_Valid_Discount"] = (df["Discount"] >= 0) & (df["Discount"] <= 1)

    if {"Revenue", "Unit_Price", "Quantity"}.issubset(df.columns):
        discount = df["Discount"] if "Discount" in df.columns else 0
        df["Expected_Revenue"] = (df["Unit_Price"] * df["Quantity"] * (1 - discount)).round(2)
        df["Revenue_Diff"] = (df["Revenue"] - df["Expected_Revenue"]).round(2)
        df["Is_Valid_Revenue"] = (df["Revenue"] >= 0) & np.isclose(
            df["Revenue"],
            df["Expected_Revenue"],
            atol=0.05,
        )

    if {"Revenue", "Cost", "Profit"}.issubset(df.columns):
        df["Expected_Profit"] = (df["Revenue"] - df["Cost"]).round(2)
        df["Profit_Diff"] = (df["Profit"] - df["Expected_Profit"]).round(2)
        df["Is_Valid_Profit"] = np.isclose(
            df["Profit"],
            df["Expected_Profit"],
            atol=0.05,
        )

    validation_cols = [col for col in df.columns if col.startswith("Is_Valid_")]
    df["Row_Valid"] = df[validation_cols].all(axis=1)

    return df


def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Order_Date" in df.columns:
        df["Order_Year"] = df["Order_Date"].dt.year
        df["Order_Month"] = df["Order_Date"].dt.month
        df["Order_Month_Name"] = df["Order_Date"].dt.month_name()
        df["Order_Day"] = df["Order_Date"].dt.day
        df["Order_Weekday"] = df["Order_Date"].dt.day_name()

    if {"Revenue", "Quantity"}.issubset(df.columns):
        df["Average_Revenue_Per_Unit"] = np.where(
            df["Quantity"] > 0,
            df["Revenue"] / df["Quantity"],
            0,
        )

    if {"Profit", "Revenue"}.issubset(df.columns):
        df["Calculated_Margin_Percent"] = np.where(
            df["Revenue"] > 0,
            (df["Profit"] / df["Revenue"]) * 100,
            0,
        ).round(2)

    if "Profit" in df.columns:
        df["Profit_Category"] = pd.cut(
            df["Profit"],
            bins=[-np.inf, 0, 50, 200, np.inf],
            labels=["Loss", "Low", "Medium", "High"],
        )

    if "Shipping_Days" in df.columns:
        df["Fast_Shipping"] = df["Shipping_Days"] <= 3

    return df


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Revenue" in df.columns:
        q1 = df["Revenue"].quantile(0.25)
        q3 = df["Revenue"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df["Revenue_Anomaly"] = (df["Revenue"] < lower) | (df["Revenue"] > upper)

    if "Profit" in df.columns:
        q1 = df["Profit"].quantile(0.25)
        q3 = df["Profit"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df["Profit_Anomaly"] = (df["Profit"] < lower) | (df["Profit"] > upper)

    return df


def create_aggregations(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    aggregations: dict[str, pd.DataFrame] = {}

    if {"Order_Year", "Order_Month", "Revenue", "Profit", "Quantity"}.issubset(df.columns):
        monthly_sales = (
            df.groupby(["Order_Year", "Order_Month"], as_index=False)
            .agg(
                Total_Revenue=("Revenue", "sum"),
                Total_Profit=("Profit", "sum"),
                Total_Quantity=("Quantity", "sum"),
                Orders=("Order_ID", "count") if "Order_ID" in df.columns else ("Revenue", "count"),
            )
            .sort_values(["Order_Year", "Order_Month"])
        )
        aggregations["monthly_sales"] = monthly_sales

    if {"Region", "Revenue", "Profit"}.issubset(df.columns):
        region_perf = (
            df.groupby("Region", as_index=False)
            .agg(
                Total_Revenue=("Revenue", "sum"),
                Total_Profit=("Profit", "sum"),
                Avg_Margin=(
                    ("Calculated_Margin_Percent", "mean")
                    if "Calculated_Margin_Percent" in df.columns
                    else ("Profit", "mean")
                ),
            )
            .sort_values("Total_Revenue", ascending=False)
        )
        aggregations["region_performance"] = region_perf

    if {"Category", "Revenue", "Profit"}.issubset(df.columns):
        category_perf = (
            df.groupby("Category", as_index=False)
            .agg(
                Total_Revenue=("Revenue", "sum"),
                Total_Profit=("Profit", "sum"),
                Total_Quantity=("Quantity", "sum") if "Quantity" in df.columns else ("Revenue", "count"),
            )
            .sort_values("Total_Revenue", ascending=False)
        )
        aggregations["category_performance"] = category_perf

    if {"Country", "Revenue"}.issubset(df.columns):
        country_sales = (
            df.groupby("Country", as_index=False)
            .agg(
                Total_Revenue=("Revenue", "sum"),
                Total_Profit=("Profit", "sum") if "Profit" in df.columns else ("Revenue", "sum"),
            )
            .sort_values("Total_Revenue", ascending=False)
        )
        aggregations["country_sales"] = country_sales

    return aggregations


def load_outputs(df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Fichier exporte : {OUTPUT_FILE.resolve()}")


def generate_quality_report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("RAPPORT QUALITE")
    print("=" * 60)
    print(f"Nombre de lignes : {len(df)}")
    print(f"Nombre de colonnes : {len(df.columns)}")

    if "Row_Valid" in df.columns:
        print(f"Lignes valides : {df['Row_Valid'].sum()}")
        print(f"Lignes invalides : {(~df['Row_Valid']).sum()}")

    missing = df.isnull().sum()
    print("\nValeurs manquantes par colonne :")
    print(missing[missing > 0])

    if "Revenue_Anomaly" in df.columns:
        print(f"\nAnomalies Revenue : {df['Revenue_Anomaly'].sum()}")

    if "Profit_Anomaly" in df.columns:
        print(f"Anomalies Profit : {df['Profit_Anomaly'].sum()}")


def run_etl() -> None:
    df = extract_data(INPUT_FILE)
    df = standardize_column_names(df)
    df = convert_data_types(df)
    df = clean_text_columns(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = validate_business_rules(df)
    df = enrich_data(df)
    df = detect_anomalies(df)

    load_outputs(df)
    generate_quality_report(df)

    print("\nETL termine avec succes")


if __name__ == "__main__":
    run_etl()
