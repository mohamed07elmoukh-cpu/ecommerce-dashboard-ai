from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.runtime
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("output_etl/sales_etl_output.csv")
PLOTLY_TEMPLATE = "plotly_dark"
CHART_HEIGHT = 320
MAP_HEIGHT = 420
COUNTRY_TO_ISO3 = {
    "Canada": "CAN",
    "China": "CHN",
    "Egypt": "EGY",
    "France": "FRA",
    "Germany": "DEU",
    "India": "IND",
    "Italy": "ITA",
    "Japan": "JPN",
    "Jordan": "JOR",
    "Kuwait": "KWT",
    "Mexico": "MEX",
    "Saudi Arabia": "SAU",
    "Singapore": "SGP",
    "South Korea": "KOR",
    "Spain": "ESP",
    "UAE": "ARE",
    "UK": "GBR",
    "USA": "USA",
}


if not streamlit.runtime.exists():
    def _identity_cache(func=None, **kwargs):
        if func is None:
            return lambda wrapped: wrapped
        return func

    st.cache_data = _identity_cache


def has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return all(column in df.columns for column in columns)


def get_margin_column(df: pd.DataFrame) -> str | None:
    for column in ["Calculated_Margin_Percent", "Profit_Margin_Percent"]:
        if column in df.columns:
            return column
    return None


def format_currency(value: float) -> str:
    return f"{value:,.2f}"


def format_number(value: float) -> str:
    return f"{value:,.0f}"


def format_percent(value: float) -> str:
    return f"{value:,.2f}%"


@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    if "Order_Date" in df.columns:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

    numeric_candidates = [
        "Revenue",
        "Profit",
        "Quantity",
        "Unit_Price",
        "Discount",
        "Cost",
        "Shipping_Cost",
        "Shipping_Days",
        "Year",
        "Month",
        "Order_Year",
        "Order_Month",
        "Profit_Margin_Percent",
        "Calculated_Margin_Percent",
    ]

    for column in numeric_candidates:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
        :root {
            --bg: #0b0f14;
            --panel: #101826;
            --panel-2: #0c1524;
            --border: rgba(148,163,184,0.16);
            --text: #e5e7eb;
            --muted: #94a3b8;
            --positive: #34d399;
            --negative: #fb7185;
            --neutral: #22d3ee;
            --accent: #14b8a6;
        }

        html, body, [class*="css"] {
            font-family: "Montserrat", system-ui, -apple-system, Segoe UI, sans-serif;
        }

        .stApp {
            background:
                radial-gradient(1200px 600px at 10% -10%, rgba(20,184,166,0.14), transparent 60%),
                radial-gradient(800px 600px at 110% 10%, rgba(34,211,238,0.08), transparent 60%),
                var(--bg);
            color: var(--text);
        }

        section[data-testid="stSidebar"] > div {
            background: linear-gradient(180deg, #0b1220 0%, #0a101c 100%);
            border-right: 1px solid var(--border);
            padding-top: 18px;
        }

        .dashboard-title {
            font-size: 32px;
            font-weight: 700;
            letter-spacing: 0.2px;
            margin-bottom: 2px;
        }

        .dashboard-subtitle {
            color: var(--muted);
            margin-top: 0;
            margin-bottom: 12px;
        }

        .header-card {
            background: linear-gradient(135deg, rgba(20,184,166,0.18), rgba(17,24,39,0.9));
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 14px 34px rgba(0,0,0,0.35);
        }

        .section-title {
            font-size: 18px;
            font-weight: 600;
            margin: 22px 0 10px 0;
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }

        .kpi-grid.small {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .kpi-card {
            background: linear-gradient(160deg, rgba(17,24,39,0.98), rgba(12,21,36,0.98));
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 12px 14px 12px 14px;
            box-shadow: 0 14px 32px rgba(0,0,0,0.32);
            min-height: 118px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .kpi-label {
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 6px;
        }

        .kpi-value {
            font-size: 26px;
            font-weight: 700;
            margin: 0;
        }

        .kpi-delta {
            font-size: 12px;
            margin-top: 6px;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(34, 211, 238, 0.14);
            color: var(--neutral);
            border-radius: 999px;
            padding: 2px 10px;
            font-size: 11px;
        }

        .pill.positive {
            background: rgba(52, 211, 153, 0.14);
            color: var(--positive);
        }

        .pill.negative {
            background: rgba(251, 113, 133, 0.14);
            color: var(--negative);
        }

        .metric-note {
            color: var(--muted);
            font-size: 12px;
        }

        .insight-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }

        .insight-card {
            background: linear-gradient(160deg, rgba(15,23,42,0.98), rgba(12,21,36,0.98));
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 12px 14px;
            box-shadow: 0 12px 26px rgba(0,0,0,0.25);
            min-height: 128px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .insight-title {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 8px;
        }

        .insight-item {
            font-size: 14px;
            margin: 4px 0;
        }

        .insight-value {
            font-weight: 600;
        }

        div[data-testid="stDataFrame"] {
            background: var(--panel);
            border-radius: 14px;
            border: 1px solid var(--border);
            padding: 6px;
        }

        button[kind="secondary"] {
            border-radius: 10px;
            border: 1px solid rgba(34,211,238,0.35);
            background: rgba(34,211,238,0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("## Filtres")
    st.sidebar.caption("Affinez l'analyse sans surcharger la vue principale.")

    filtered_df = df.copy()

    def apply_filter(label: str, candidates: list[str]) -> None:
        nonlocal filtered_df
        column = next((candidate for candidate in candidates if candidate in filtered_df.columns), None)
        if not column:
            return

        values = filtered_df[column].dropna().unique().tolist()
        values = sorted(values, key=lambda value: str(value))

        selected_values = st.multiselect(
            label,
            options=values,
            default=values,
            key=f"filter_{column}",
        )

        if selected_values:
            filtered_df = filtered_df[filtered_df[column].isin(selected_values)]

    with st.sidebar.expander("Temps", expanded=True):
        apply_filter("Année", ["Order_Year", "Year"])

    with st.sidebar.expander("Geographie", expanded=False):
        apply_filter("Region", ["Region"])
        apply_filter("Pays", ["Country"])

    with st.sidebar.expander("Produit", expanded=False):
        apply_filter("Categorie", ["Category"])
        apply_filter("Sous-categorie", ["Sub_Category"])

    with st.sidebar.expander("Transaction", expanded=False):
        apply_filter("Methode de paiement", ["Payment_Method"])
        apply_filter("Statut de commande", ["Order_Status"])

    return filtered_df


def build_targets() -> dict[str, float | None]:
    with st.sidebar.expander("Objectifs", expanded=False):
        st.caption("0 pour desactiver un objectif.")
        revenue_target = st.number_input(
            "Objectif chiffre d'affaires",
            min_value=0.0,
            value=0.0,
            step=10000.0,
            format="%.0f",
        )
        profit_target = st.number_input(
            "Objectif profit",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            format="%.0f",
        )
        margin_target = st.number_input(
            "Objectif marge (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
        )
        delivered_target = st.number_input(
            "Objectif livraison reussie (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
        )
        returned_target = st.number_input(
            "Seuil max retour (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
        )

    def normalize(value: float) -> float | None:
        return value if value and value > 0 else None

    return {
        "revenue": normalize(revenue_target),
        "profit": normalize(profit_target),
        "margin": normalize(margin_target),
        "delivered": normalize(delivered_target),
        "returned": normalize(returned_target),
    }


def build_period_series(df: pd.DataFrame) -> pd.Series | None:
    if has_columns(df, ["Order_Year", "Order_Month"]):
        return pd.to_datetime(
            df["Order_Year"].astype("Int64").astype(str)
            + "-"
            + df["Order_Month"].astype("Int64").astype(str).str.zfill(2)
            + "-01",
            errors="coerce",
        ).dt.to_period("M")

    if "Order_Date" in df.columns:
        return df["Order_Date"].dt.to_period("M")

    return None


def compute_kpi_snapshot(df: pd.DataFrame) -> dict[str, float]:
    margin_column = get_margin_column(df)
    revenue = df["Revenue"].sum() if "Revenue" in df.columns else 0.0
    profit = df["Profit"].sum() if "Profit" in df.columns else 0.0
    orders = df["Order_ID"].nunique() if "Order_ID" in df.columns else float(len(df))
    avg_basket = revenue / orders if orders else 0.0
    margin = df[margin_column].mean() if margin_column else 0.0

    delivered_rate = 0.0
    returned_rate = 0.0
    if "Order_Status" in df.columns:
        status_series = df["Order_Status"].astype(str).str.lower()
        delivered_rate = status_series.str.contains("delivered").mean() * 100
        returned_rate = status_series.str.contains("returned").mean() * 100

    return {
        "revenue": revenue,
        "profit": profit,
        "margin": margin,
        "orders": orders,
        "avg_basket": avg_basket,
        "delivered_rate": delivered_rate,
        "returned_rate": returned_rate,
    }


def compute_kpi_deltas(df: pd.DataFrame) -> dict[str, float | None]:
    period = build_period_series(df)
    if period is None:
        return {key: None for key in compute_kpi_snapshot(df)}

    working = df.copy()
    working["Period"] = period
    latest_period = working["Period"].dropna().max()
    if latest_period is pd.NaT:
        return {key: None for key in compute_kpi_snapshot(df)}

    current = working[working["Period"] == latest_period]
    previous = working[working["Period"] == (latest_period - 1)]

    if previous.empty:
        return {key: None for key in compute_kpi_snapshot(df)}

    current_snap = compute_kpi_snapshot(current)
    previous_snap = compute_kpi_snapshot(previous)
    deltas = {}
    for key, value in current_snap.items():
        deltas[key] = value - previous_snap.get(key, 0)
    return deltas


def get_kpi_definitions() -> list[dict[str, str | bool]]:
    return [
        {
            "key": "revenue",
            "label": "Chiffre d'affaires",
            "icon": "REV",
            "is_percent": False,
            "direction": "up",
        },
        {
            "key": "profit",
            "label": "Profit",
            "icon": "PROF",
            "is_percent": False,
            "direction": "up",
        },
        {
            "key": "margin",
            "label": "Marge moyenne",
            "icon": "MRG",
            "is_percent": True,
            "direction": "up",
        },
        {
            "key": "orders",
            "label": "Commandes",
            "icon": "ORD",
            "is_percent": False,
            "direction": "up",
        },
        {
            "key": "avg_basket",
            "label": "Panier moyen",
            "icon": "AVG",
            "is_percent": False,
            "direction": "up",
        },
        {
            "key": "delivered_rate",
            "label": "Livraison reussie",
            "icon": "OK",
            "is_percent": True,
            "direction": "up",
        },
        {
            "key": "returned_rate",
            "label": "Taux de retour",
            "icon": "RET",
            "is_percent": True,
            "direction": "down",
        },
    ]


def render_kpi_card(
    label: str,
    value: str,
    delta: str | None,
    delta_tone: str | None,
    icon: str,
) -> None:
    color = {
        "positive": "var(--positive)",
        "negative": "var(--negative)",
        "neutral": "var(--neutral)",
    }.get(delta_tone or "neutral", "var(--neutral)")

    delta_html = ""
    if delta:
        delta_html = f"<div class='kpi-delta' style='color:{color}'>{delta}</div>"

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{icon} {label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_kpis(df: pd.DataFrame) -> None:
    snapshot = compute_kpi_snapshot(df)
    deltas = compute_kpi_deltas(df)

    definitions = get_kpi_definitions()
    metrics = []
    for definition in definitions:
        key = definition["key"]
        value = snapshot.get(key, 0.0)
        delta = deltas.get(key)
        formatter = format_percent if definition["is_percent"] else format_currency
        if key == "orders":
            formatter = format_number
        metrics.append(
            {
                "label": definition["label"],
                "value": formatter(value),
                "delta": delta,
                "direction": definition["direction"],
                "icon": definition["icon"],
                "is_percent": definition["is_percent"],
            }
        )

    st.markdown("<div class='section-title'>KPI principaux</div>", unsafe_allow_html=True)

    for idx in range(0, len(metrics), 2):
        row = metrics[idx:idx + 2]
        columns = st.columns(2)
        for col, metric in zip(columns, row):
            delta_value = metric["delta"]
            delta_display = None
            delta_tone = None
            if delta_value is not None:
                delta_display = (
                    f"{delta_value:+.2f} pts vs période précédent"
                    if metric["is_percent"]
                    else f"{delta_value:+,.2f} vs période précédent"
                )
                if metric["direction"] == "down":
                    delta_tone = "positive" if delta_value < 0 else "negative"
                else:
                    delta_tone = "positive" if delta_value >= 0 else "negative"
            with col:
                render_kpi_card(
                    metric["label"],
                    metric["value"],
                    delta_display,
                    delta_tone,
                    metric["icon"],
                )


def build_insights(df: pd.DataFrame) -> str:
    insights = []

    if has_columns(df, ["Region", "Profit"]) and df["Region"].notna().any():
        region_sum = df.groupby("Region")["Profit"].sum()
        if not region_sum.empty:
            region = region_sum.idxmax()
            insights.append(f"Region la plus rentable: {region}")

    if has_columns(df, ["Category", "Revenue"]) and df["Category"].notna().any():
        category_sum = df.groupby("Category")["Revenue"].sum()
        if not category_sum.empty:
            category = category_sum.idxmax()
            insights.append(f"Categorie leader: {category}")

    if has_columns(df, ["Country", "Revenue"]) and df["Country"].notna().any():
        country_sum = df.groupby("Country")["Revenue"].sum()
        if not country_sum.empty:
            country = country_sum.idxmax()
            insights.append(f"Pays leader: {country}")

    if "Order_Status" in df.columns:
        delivered_rate = df["Order_Status"].astype(str).str.lower().str.contains("delivered").mean() * 100
        insights.append(f"Taux de livraison: {delivered_rate:.1f}%")

    return " | ".join(insights)


def ensure_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    if all(col in df.columns for col in ["Order_Year", "Order_Month", "Order_Day"]):
        return df

    if "Order_Date" not in df.columns:
        return df

    enriched = df.copy()
    enriched["Order_Year"] = enriched["Order_Date"].dt.year
    enriched["Order_Month"] = enriched["Order_Date"].dt.month
    enriched["Order_Day"] = enriched["Order_Date"].dt.day
    return enriched


def aggregate_metric(df: pd.DataFrame, group_cols: list[str], metric_key: str) -> pd.DataFrame:
    if not group_cols:
        return pd.DataFrame()

    if metric_key in {"revenue", "profit"}:
        value_col = "Revenue" if metric_key == "revenue" else "Profit"
        if value_col not in df.columns:
            return pd.DataFrame()
        agg = df.groupby(group_cols, as_index=False)[value_col].sum()
        return agg.rename(columns={value_col: "Metric"})

    if metric_key == "orders":
        if "Order_ID" not in df.columns:
            return pd.DataFrame()
        agg = df.groupby(group_cols, as_index=False)["Order_ID"].nunique()
        return agg.rename(columns={"Order_ID": "Metric"})

    if metric_key == "avg_basket":
        if not has_columns(df, ["Revenue", "Order_ID"]):
            return pd.DataFrame()
        revenue = df.groupby(group_cols)["Revenue"].sum()
        orders = df.groupby(group_cols)["Order_ID"].nunique()
        avg = (revenue / orders.replace(0, pd.NA)).fillna(0)
        return avg.reset_index().rename(columns={0: "Metric"})

    if metric_key == "margin":
        margin_col = get_margin_column(df)
        if not margin_col:
            return pd.DataFrame()
        agg = df.groupby(group_cols, as_index=False)[margin_col].mean()
        return agg.rename(columns={margin_col: "Metric"})

    if metric_key in {"delivered_rate", "returned_rate"}:
        if "Order_Status" not in df.columns:
            return pd.DataFrame()
        status = df["Order_Status"].astype(str).str.lower()
        if metric_key == "delivered_rate":
            flag = status.str.contains("delivered")
        else:
            flag = status.str.contains("returned")
        agg = df.assign(_flag=flag).groupby(group_cols, as_index=False)["_flag"].mean()
        agg["Metric"] = agg["_flag"] * 100
        return agg.drop(columns=["_flag"])

    return pd.DataFrame()


def show_drilldown(df: pd.DataFrame) -> None:
    st.markdown("<div class='section-title'>Drill-down KPI (Année -> Mois -> Jour)</div>", unsafe_allow_html=True)

    working = ensure_date_parts(df)
    if "Order_Year" not in working.columns:
        st.info("Drill-down indisponible: colonne de date manquante.")
        return

    kpi_defs = get_kpi_definitions()
    labels = [definition["label"] for definition in kpi_defs]
    default_key = st.session_state.get("drill_kpi", kpi_defs[0]["key"])
    default_index = next((i for i, d in enumerate(kpi_defs) if d["key"] == default_key), 0)
    selected_label = st.radio(
        "KPI a explorer",
        options=labels,
        index=default_index,
        horizontal=True,
    )
    selected_def = next(defn for defn in kpi_defs if defn["label"] == selected_label)
    st.session_state["drill_kpi"] = selected_def["key"]

    year_options = sorted(working["Order_Year"].dropna().unique().tolist())
    if not year_options:
        st.info("Aucune Année disponible pour le drill-down.")
        return

    col_year, col_month, col_day = st.columns(3)
    with col_year:
        year_value = st.selectbox("Année", options=year_options, index=len(year_options) - 1)
    months_df = working[working["Order_Year"] == year_value]
    month_options = sorted(months_df["Order_Month"].dropna().unique().tolist())
    month_value = None
    if month_options:
        with col_month:
            month_value = st.selectbox("Mois", options=month_options, index=len(month_options) - 1)
    days_df = months_df
    if month_value is not None:
        days_df = days_df[days_df["Order_Month"] == month_value]
    day_options = sorted(days_df["Order_Day"].dropna().unique().tolist()) if "Order_Day" in days_df.columns else []
    day_value = None
    if day_options:
        with col_day:
            day_value = st.selectbox("Jour", options=day_options, index=len(day_options) - 1)

    year_metric = aggregate_metric(working, ["Order_Year"], selected_def["key"])
    month_metric = aggregate_metric(months_df, ["Order_Month"], selected_def["key"])
    day_metric = aggregate_metric(days_df, ["Order_Day"], selected_def["key"]) if day_options else pd.DataFrame()

    def metric_title(label: str) -> str:
        return f"{selected_def['label']} ({label})"

    col_a, col_b = st.columns(2)
    with col_a:
        if not year_metric.empty:
            fig_year = px.line(
                year_metric,
                x="Order_Year",
                y="Metric",
                markers=True,
                title=metric_title("Année"),
                template=PLOTLY_TEMPLATE,
            )
            fig_year.update_layout(height=CHART_HEIGHT, xaxis_title="Année", yaxis_title="Valeur")
            st.plotly_chart(fig_year, width="stretch")
    with col_b:
        if not month_metric.empty:
            fig_month = px.bar(
                month_metric,
                x="Order_Month",
                y="Metric",
                title=metric_title("Mois"),
                template=PLOTLY_TEMPLATE,
            )
            fig_month.update_layout(height=CHART_HEIGHT, xaxis_title="Mois", yaxis_title="Valeur")
            st.plotly_chart(fig_month, width="stretch")

    if not day_metric.empty:
        fig_day = px.bar(
            day_metric,
            x="Order_Day",
            y="Metric",
            title=metric_title("Jour"),
            template=PLOTLY_TEMPLATE,
        )
        fig_day.update_layout(height=CHART_HEIGHT, xaxis_title="Jour", yaxis_title="Valeur")
        st.plotly_chart(fig_day, width="stretch")


def compute_period_summary(df: pd.DataFrame) -> dict[str, dict[str, float] | str] | None:
    period = build_period_series(df)
    if period is None:
        return None

    working = df.copy()
    working["Period"] = period
    working = working.dropna(subset=["Period"])
    if working.empty:
        return None

    periods = sorted(working["Period"].unique())
    if len(periods) < 2:
        return None

    current_period = periods[-1]
    previous_period = periods[-2]

    current = working[working["Period"] == current_period]
    previous = working[working["Period"] == previous_period]

    def agg_snapshot(frame: pd.DataFrame) -> dict[str, float]:
        margin_column = get_margin_column(frame)
        snapshot = {
            "revenue": frame["Revenue"].sum() if "Revenue" in frame.columns else 0.0,
            "profit": frame["Profit"].sum() if "Profit" in frame.columns else 0.0,
            "orders": frame["Order_ID"].nunique() if "Order_ID" in frame.columns else float(len(frame)),
        }
        snapshot["margin"] = frame[margin_column].mean() if margin_column else 0.0
        return snapshot

    return {
        "current_period": str(current_period),
        "previous_period": str(previous_period),
        "current": agg_snapshot(current),
        "previous": agg_snapshot(previous),
    }


def compute_growth(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return (current - previous) / previous * 100


def compute_declines(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    top_n: int = 3,
) -> list[tuple[str, float]]:
    if not has_columns(df, [group_col, value_col]):
        return []

    period = build_period_series(df)
    if period is None:
        return []

    working = df.copy()
    working["Period"] = period
    working = working.dropna(subset=["Period"])
    periods = sorted(working["Period"].unique())
    if len(periods) < 2:
        return []

    current_period = periods[-1]
    previous_period = periods[-2]

    current = (
        working[working["Period"] == current_period]
        .groupby(group_col)[value_col]
        .sum()
    )
    previous = (
        working[working["Period"] == previous_period]
        .groupby(group_col)[value_col]
        .sum()
    )

    combined = pd.concat([current, previous], axis=1, keys=["current", "previous"]).fillna(0)
    combined["diff"] = combined["current"] - combined["previous"]
    declines = combined[combined["diff"] < 0].sort_values("diff").head(top_n)

    return [(index, float(value)) for index, value in declines["diff"].items()]


def compute_top_profit(df: pd.DataFrame, group_col: str, top_n: int = 3) -> list[tuple[str, float]]:
    if not has_columns(df, [group_col, "Profit"]):
        return []

    grouped = df.groupby(group_col)["Profit"].sum().sort_values(ascending=False).head(top_n)
    return [(index, float(value)) for index, value in grouped.items()]


def render_insight_card(title: str, lines: list[str]) -> None:
    lines_html = "".join(f"<div class='insight-item'>{line}</div>" for line in lines)
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            {lines_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_strategic_section(df: pd.DataFrame, targets: dict[str, float | None]) -> None:
    st.markdown("<div class='section-title'>Questions strategiques</div>", unsafe_allow_html=True)

    summary = compute_period_summary(df)
    growth_lines: list[str] = []
    if summary is None:
        growth_lines.append("Croissance: donnees insuffisantes pour comparer.")
    else:
        revenue_growth = compute_growth(summary["current"]["revenue"], summary["previous"]["revenue"])
        profit_growth = compute_growth(summary["current"]["profit"], summary["previous"]["profit"])
        status = "Stable"
        tag_class = "neutral"
        if revenue_growth is not None:
            if revenue_growth > 0:
                status = "En croissance"
                tag_class = "positive"
            elif revenue_growth < 0:
                status = "En baisse"
                tag_class = "negative"

        growth_lines.append(f"Entreprise: <span class='pill {tag_class}'>{status}</span>")
        growth_lines.append(f"période comparee: {summary['previous_period']} -> {summary['current_period']}")
        if revenue_growth is not None:
            growth_lines.append(f"CA: <span class='insight-value'>{revenue_growth:+.1f}%</span>")
        if profit_growth is not None:
            growth_lines.append(f"Profit: <span class='insight-value'>{profit_growth:+.1f}%</span>")

    top_regions = compute_top_profit(df, "Region")
    top_products = compute_top_profit(df, "Product_Name")
    rent_lines: list[str] = []
    if top_regions:
        region_label, region_value = top_regions[0]
        rent_lines.append(f"Region la plus rentable: <span class='insight-value'>{region_label}</span>")
        rent_lines.append(f"Profit region: {format_currency(region_value)}")
    if top_products:
        product_label, product_value = top_products[0]
        rent_lines.append(f"Produit le plus rentable: <span class='insight-value'>{product_label}</span>")
        rent_lines.append(f"Profit produit: {format_currency(product_value)}")
    if not rent_lines:
        rent_lines.append("Rentabilite: donnees insuffisantes.")

    decline_regions = compute_declines(df, "Region", "Revenue")
    decline_categories = compute_declines(df, "Category", "Revenue")
    decline_lines: list[str] = []
    if decline_regions:
        name, diff = decline_regions[0]
        decline_lines.append(f"Baisse region: {name} ({diff:,.2f})")
    if decline_categories:
        name, diff = decline_categories[0]
        decline_lines.append(f"Baisse categorie: {name} ({diff:,.2f})")
    if not decline_lines:
        decline_lines.append("Aucune baisse significative detectee.")

    snapshot = compute_kpi_snapshot(df)
    target_lines: list[str] = []
    target_checks = [
        ("Chiffre d'affaires", "revenue", format_currency, True),
        ("Profit", "profit", format_currency, True),
        ("Marge", "margin", format_percent, True),
        ("Livraison reussie", "delivered_rate", format_percent, True),
        ("Taux de retour", "returned_rate", format_percent, False),
    ]
    for label, key, formatter, higher_is_better in target_checks:
        target_value = targets.get(key)
        if target_value is None:
            continue
        actual = snapshot.get(key, 0.0)
        ok = actual >= target_value if higher_is_better else actual <= target_value
        pill_class = "positive" if ok else "negative"
        target_lines.append(
            f"{label}: {formatter(actual)} / {formatter(target_value)} "
            f"<span class='pill {pill_class}'>{'OK' if ok else 'NON'}</span>"
        )
    if not target_lines:
        target_lines.append("Aucun objectif defini dans la sidebar.")

    cards = [
        ("Croissance", growth_lines),
        ("Rentabilite", rent_lines),
        ("Baisse de performance", decline_lines),
        ("Objectifs", target_lines),
    ]

    for idx in range(0, len(cards), 2):
        row = cards[idx:idx + 2]
        columns = st.columns(2)
        for col, (title, lines) in zip(columns, row):
            with col:
                render_insight_card(title, lines)


def show_monthly_sales_chart(df: pd.DataFrame) -> None:
    if "Revenue" not in df.columns:
        return

    if has_columns(df, ["Order_Year", "Order_Month"]):
        monthly_sales = (
            df.groupby(["Order_Year", "Order_Month"], as_index=False)["Revenue"]
            .sum()
            .sort_values(["Order_Year", "Order_Month"])
        )
        monthly_sales["Period"] = (
            monthly_sales["Order_Year"].astype("Int64").astype(str)
            + "-"
            + monthly_sales["Order_Month"].astype("Int64").astype(str).str.zfill(2)
        )
    elif "Order_Date" in df.columns:
        monthly_sales = (
            df.dropna(subset=["Order_Date"])
            .assign(Period=lambda frame: frame["Order_Date"].dt.to_period("M").astype(str))
            .groupby("Period", as_index=False)["Revenue"]
            .sum()
            .sort_values("Period")
        )
    else:
        return

    fig = px.line(
        monthly_sales,
        x="Period",
        y="Revenue",
        markers=True,
        title="Ventes par mois",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=CHART_HEIGHT, xaxis_title="Mois", yaxis_title="Chiffre d'affaires")
    st.plotly_chart(fig, width="stretch")


def show_category_sales_chart(df: pd.DataFrame) -> None:
    if not has_columns(df, ["Category", "Revenue"]):
        return

    category_sales = (
        df.groupby("Category", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
    )

    fig = px.bar(
        category_sales,
        x="Category",
        y="Revenue",
        color="Category",
        title="Ventes par categorie",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=CHART_HEIGHT, showlegend=False, xaxis_title="Categorie", yaxis_title="Chiffre d'affaires")
    st.plotly_chart(fig, width="stretch")


def show_region_profit_chart(df: pd.DataFrame) -> None:
    if not has_columns(df, ["Region", "Profit"]):
        return

    region_profit = (
        df.groupby("Region", as_index=False)["Profit"]
        .sum()
        .sort_values("Profit", ascending=False)
    )

    fig = px.bar(
        region_profit,
        x="Region",
        y="Profit",
        color="Region",
        title="Profit par region",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=CHART_HEIGHT, showlegend=False, xaxis_title="Region", yaxis_title="Profit")
    st.plotly_chart(fig, width="stretch")


def show_top_products_chart(df: pd.DataFrame) -> None:
    if not has_columns(df, ["Product_Name", "Revenue"]):
        return

    top_products = (
        df.groupby("Product_Name", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
        .head(10)
    )

    fig = px.bar(
        top_products.sort_values("Revenue", ascending=True),
        x="Revenue",
        y="Product_Name",
        orientation="h",
        color="Revenue",
        title="Top 10 produits par chiffre d'affaires",
        template=PLOTLY_TEMPLATE,
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=CHART_HEIGHT, yaxis_title="Produit", xaxis_title="Chiffre d'affaires")
    st.plotly_chart(fig, width="stretch")


def show_payment_sales_chart(df: pd.DataFrame) -> None:
    if not has_columns(df, ["Payment_Method", "Revenue"]):
        return

    payment_sales = df.groupby("Payment_Method", as_index=False)["Revenue"].sum()

    fig = px.pie(
        payment_sales,
        names="Payment_Method",
        values="Revenue",
        title="Repartition des ventes par methode de paiement",
        template=PLOTLY_TEMPLATE,
        hole=0.45,
    )
    fig.update_layout(height=CHART_HEIGHT)
    st.plotly_chart(fig, width="stretch")


def show_order_status_chart(df: pd.DataFrame) -> None:
    if "Order_Status" not in df.columns:
        return

    status_counts = df.groupby("Order_Status", as_index=False).size()

    fig = px.pie(
        status_counts,
        names="Order_Status",
        values="size",
        title="Repartition des commandes par statut",
        template=PLOTLY_TEMPLATE,
        hole=0.4,
    )
    fig.update_layout(height=CHART_HEIGHT)
    st.plotly_chart(fig, width="stretch")


def show_revenue_profit_scatter(df: pd.DataFrame) -> None:
    if not has_columns(df, ["Revenue", "Profit"]):
        return

    hover_fields = [
        column
        for column in ["Order_ID", "Product_Name", "Category", "Region", "Country"]
        if column in df.columns
    ]
    color_column = "Category" if "Category" in df.columns else None
    size_column = "Quantity" if "Quantity" in df.columns else None

    fig = px.scatter(
        df,
        x="Revenue",
        y="Profit",
        color=color_column,
        size=size_column,
        hover_data=hover_fields,
        title="Revenue vs Profit",
        template=PLOTLY_TEMPLATE,
        opacity=0.7,
    )
    fig.update_layout(height=CHART_HEIGHT, xaxis_title="Revenue", yaxis_title="Profit")
    st.plotly_chart(fig, width="stretch")


def show_profit_boxplot(df: pd.DataFrame) -> None:
    if not has_columns(df, ["Category", "Profit"]):
        return

    fig = px.box(
        df,
        x="Category",
        y="Profit",
        color="Category",
        title="Distribution du profit par categorie",
        template=PLOTLY_TEMPLATE,
        points="outliers",
    )
    fig.update_layout(height=CHART_HEIGHT, showlegend=False, xaxis_title="Categorie", yaxis_title="Profit")
    st.plotly_chart(fig, width="stretch")


def show_world_map(df: pd.DataFrame) -> None:
    if not has_columns(df, ["Country", "Revenue"]):
        return

    country_sales = (
        df.groupby("Country", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
    )
    country_sales["ISO3"] = country_sales["Country"].map(COUNTRY_TO_ISO3)
    country_sales = country_sales.dropna(subset=["ISO3"])

    if country_sales.empty:
        return

    fig = px.choropleth(
        country_sales,
        locations="ISO3",
        locationmode="ISO-3",
        color="Revenue",
        hover_name="Country",
        color_continuous_scale="Blues",
        title="Carte mondiale des ventes par pays",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=MAP_HEIGHT)
    fig.update_geos(showframe=False, showcoastlines=True, projection_type="equirectangular")
    st.plotly_chart(fig, width="stretch")


@st.cache_data
def compute_sales_forecast(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    if "Revenue" not in df.columns:
        return None, None

    if "Order_Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Order_Date"]):
        monthly = (
            df.dropna(subset=["Order_Date", "Revenue"])
            .set_index("Order_Date")
            .resample("MS")["Revenue"]
            .sum()
            .reset_index()
            .rename(columns={"Order_Date": "Date"})
        )
    elif has_columns(df, ["Order_Year", "Order_Month", "Revenue"]):
        working = df.dropna(subset=["Order_Year", "Order_Month", "Revenue"]).copy()
        working["Date"] = pd.to_datetime(
            working["Order_Year"].astype("Int64").astype(str)
            + "-"
            + working["Order_Month"].astype("Int64").astype(str).str.zfill(2)
            + "-01",
            errors="coerce",
        )
        monthly = (
            working.dropna(subset=["Date"])
            .groupby("Date", as_index=False)["Revenue"]
            .sum()
            .sort_values("Date")
        )
    else:
        return None, None

    monthly = monthly.sort_values("Date").reset_index(drop=True)
    if len(monthly) < 4:
        return None, None

    x = np.arange(len(monthly)).reshape(-1, 1)
    y = monthly["Revenue"].values
    model = LinearRegression()
    model.fit(x, y)

    horizon = 3
    future_x = np.arange(len(monthly), len(monthly) + horizon).reshape(-1, 1)
    future_y = model.predict(future_x)
    start_date = monthly["Date"].max() + pd.offsets.MonthBegin(1)
    future_dates = pd.date_range(start=start_date, periods=horizon, freq="MS")
    forecast = pd.DataFrame({"Date": future_dates, "Revenue": future_y})

    return monthly, forecast


def show_sales_forecast(df: pd.DataFrame) -> None:
    st.markdown("#### Prevision des ventes (3 mois)")
    history, forecast = compute_sales_forecast(df)
    if history is None or forecast is None:
        st.info("Prevision indisponible: colonnes Date/Revenue manquantes ou historique insuffisant.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["Date"],
            y=history["Revenue"],
            mode="lines+markers",
            name="Historique",
            line=dict(color="#22d3ee", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["Date"],
            y=forecast["Revenue"],
            mode="lines+markers",
            name="Prevision",
            line=dict(color="#f59e0b", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT,
        xaxis_title="Mois",
        yaxis_title="Chiffre d'affaires",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, width="stretch")


@st.cache_data
def compute_customer_segmentation(df: pd.DataFrame, k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    if "Customer_ID" not in df.columns or "Revenue" not in df.columns or "Profit" not in df.columns:
        return None, None

    base = df.dropna(subset=["Customer_ID"]).copy()
    if base.empty:
        return None, None

    if "Order_ID" in base.columns:
        grouped = (
            base.groupby("Customer_ID", as_index=False)
            .agg(
                Revenue=("Revenue", "sum"),
                Profit=("Profit", "sum"),
                Orders=("Order_ID", "nunique"),
            )
        )
    else:
        grouped = (
            base.groupby("Customer_ID", as_index=False)
            .agg(
                Revenue=("Revenue", "sum"),
                Profit=("Profit", "sum"),
                Orders=("Revenue", "size"),
            )
        )

    grouped = grouped.dropna(subset=["Revenue", "Profit", "Orders"])
    if len(grouped) < 2:
        return None, None

    n_clusters = min(max(2, k), len(grouped))
    features = grouped[["Revenue", "Profit", "Orders"]]
    scaled = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    grouped["Cluster"] = model.fit_predict(scaled).astype(str)

    summary = (
        grouped.groupby("Cluster", as_index=False)
        .agg(
            Clients=("Customer_ID", "count"),
            Avg_Revenue=("Revenue", "mean"),
            Avg_Profit=("Profit", "mean"),
            Avg_Orders=("Orders", "mean"),
        )
        .sort_values("Cluster")
    )
    return grouped, summary


def show_customer_segmentation(df: pd.DataFrame) -> None:
    st.markdown("#### Segmentation clients (K-means)")
    segmented, summary = compute_customer_segmentation(df, k=3)
    if segmented is None or summary is None:
        st.info("Segmentation indisponible: colonnes client/revenue/profit manquantes ou volume insuffisant.")
        return

    fig = px.scatter(
        segmented,
        x="Revenue",
        y="Profit",
        color="Cluster",
        size="Orders",
        title="Segmentation clients (K-means)",
        template=PLOTLY_TEMPLATE,
        hover_data=["Customer_ID"],
    )
    fig.update_layout(height=CHART_HEIGHT, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(summary, width="stretch", hide_index=True)


@st.cache_data
def compute_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame | None:
    if not has_columns(df, ["Revenue", "Profit"]):
        return None

    working = df[["Revenue", "Profit"]].dropna().copy()
    if len(working) < 20:
        return None

    model = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        random_state=42,
    )
    labels = model.fit_predict(working[["Revenue", "Profit"]])
    working["Anomaly"] = np.where(labels == -1, 1, 0)
    return working


def show_anomaly_detection(df: pd.DataFrame) -> None:
    st.markdown("#### Detection des anomalies de ventes")
    anomalies = compute_anomaly_detection(df)
    if anomalies is None:
        st.info("Detection d'anomalies indisponible: colonnes Revenue/Profit manquantes ou volume insuffisant.")
        return

    anomalies["Type"] = np.where(anomalies["Anomaly"] == 1, "Anomalie", "Normal")
    fig = px.scatter(
        anomalies,
        x="Revenue",
        y="Profit",
        color="Type",
        color_discrete_map={"Normal": "#38bdf8", "Anomalie": "#ef4444"},
        title="Detection des anomalies de ventes",
        template=PLOTLY_TEMPLATE,
        opacity=0.75,
    )
    fig.update_layout(height=CHART_HEIGHT, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")

    anomaly_count = int((anomalies["Anomaly"] == 1).sum())
    st.caption(f"Points anormaux detectes: {anomaly_count}")


@st.cache_data
def get_segmented_customers(df: pd.DataFrame) -> pd.DataFrame | None:
    if "Customer_ID" not in df.columns:
        return None

    base = df.dropna(subset=["Customer_ID"]).copy()
    if base.empty:
        return None

    if "Order_ID" in base.columns:
        customer = (
            base.groupby("Customer_ID", as_index=False)
            .agg(
                Revenue=("Revenue", "sum") if "Revenue" in base.columns else ("Customer_ID", "size"),
                Profit=("Profit", "sum") if "Profit" in base.columns else ("Customer_ID", "size"),
                Orders=("Order_ID", "nunique"),
            )
        )
    else:
        customer = (
            base.groupby("Customer_ID", as_index=False)
            .agg(
                Revenue=("Revenue", "sum") if "Revenue" in base.columns else ("Customer_ID", "size"),
                Profit=("Profit", "sum") if "Profit" in base.columns else ("Customer_ID", "size"),
                Orders=("Customer_ID", "size"),
            )
        )

    if "Cluster" in base.columns:
        cluster_map = (
            base[["Customer_ID", "Cluster"]]
            .dropna(subset=["Cluster"])
            .assign(Cluster=lambda x: x["Cluster"].astype(str))
            .groupby("Customer_ID")["Cluster"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[-1])
            .reset_index()
        )
        customer = customer.merge(cluster_map, on="Customer_ID", how="left")
        if customer["Cluster"].notna().sum() >= 2:
            customer["Cluster"] = customer["Cluster"].fillna("Unknown")
            return customer

    segmented, _ = compute_customer_segmentation(df, k=3)
    if segmented is None:
        return None
    return segmented


@st.cache_data
def get_anomaly_frame(df: pd.DataFrame) -> pd.DataFrame | None:
    if not has_columns(df, ["Revenue", "Profit"]):
        return None

    working = df.copy()
    if "Order_Date" in working.columns and pd.api.types.is_datetime64_any_dtype(working["Order_Date"]):
        pass
    elif "Order_Date" in working.columns:
        working["Order_Date"] = pd.to_datetime(working["Order_Date"], errors="coerce")

    required_cols = ["Revenue", "Profit"]
    if "Order_Date" in working.columns:
        required_cols.append("Order_Date")
    working = working.dropna(subset=["Revenue", "Profit"]).copy()
    if working.empty:
        return None

    if "Anomaly" in working.columns:
        an = pd.to_numeric(working["Anomaly"], errors="coerce").fillna(0)
        # support both {-1,1} and {0,1}
        if an.min() < 0:
            working["Anomaly"] = (an == -1).astype(int)
        else:
            working["Anomaly"] = (an > 0).astype(int)
        return working

    if len(working) < 20:
        return None

    model = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
    labels = model.fit_predict(working[["Revenue", "Profit"]])
    working["Anomaly"] = np.where(labels == -1, 1, 0)
    return working


def show_cluster_distribution(df: pd.DataFrame) -> None:
    st.markdown("#### Distribution des clusters")
    segmented = get_segmented_customers(df)
    if segmented is None or "Cluster" not in segmented.columns:
        st.info("Distribution indisponible: colonnes Customer_ID/Cluster manquantes.")
        return

    counts = (
        segmented.groupby("Cluster", as_index=False)["Customer_ID"]
        .nunique()
        .rename(columns={"Customer_ID": "Clients"})
    )
    counts["Cluster_sort"] = pd.to_numeric(counts["Cluster"], errors="coerce")
    counts = counts.sort_values(["Cluster_sort", "Cluster"]).drop(columns=["Cluster_sort"])

    fig = px.bar(
        counts,
        x="Cluster",
        y="Clients",
        title="Distribution des clusters",
        template=PLOTLY_TEMPLATE,
        color="Cluster",
    )
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def show_cluster_profile_heatmap(df: pd.DataFrame) -> None:
    st.markdown("#### Profils moyens des clusters")
    segmented = get_segmented_customers(df)
    if segmented is None or "Cluster" not in segmented.columns:
        st.info("Heatmap indisponible: colonnes cluster insuffisantes.")
        return

    required = ["Revenue", "Profit", "Orders"]
    if not has_columns(segmented, required):
        st.info("Heatmap indisponible: Revenue/Profit/Orders manquants.")
        return

    profile = (
        segmented.groupby("Cluster", as_index=False)
        .agg(
            Avg_Revenue=("Revenue", "mean"),
            Avg_Profit=("Profit", "mean"),
            Avg_Orders=("Orders", "mean"),
        )
    )
    profile["Cluster_sort"] = pd.to_numeric(profile["Cluster"], errors="coerce")
    profile = profile.sort_values(["Cluster_sort", "Cluster"]).drop(columns=["Cluster_sort"])

    z_values = profile[["Avg_Revenue", "Avg_Profit", "Avg_Orders"]].round(2).values
    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=["Avg Revenue", "Avg Profit", "Avg Orders"],
            y=profile["Cluster"].astype(str),
            colorscale="Tealgrn",
            text=z_values,
            texttemplate="%{text}",
            hovertemplate="Cluster %{y}<br>%{x}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Profils moyens des clusters",
        template=PLOTLY_TEMPLATE,
        height=280,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis_title="",
        yaxis_title="Cluster",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_cluster_boxplot(df: pd.DataFrame) -> None:
    segmented = get_segmented_customers(df)
    if segmented is None or "Cluster" not in segmented.columns:
        st.info("Boxplot indisponible: colonnes cluster insuffisantes.")
        return

    value_col = None
    chart_title = "Distribution du revenu par cluster"
    if "Revenue" in segmented.columns:
        value_col = "Revenue"
    elif "Profit" in segmented.columns:
        value_col = "Profit"
        chart_title = "Distribution du profit par cluster"

    if value_col is None:
        st.info("Boxplot indisponible: Revenue/Profit manquants.")
        return

    st.markdown(f"#### {chart_title}")
    plot_df = segmented[["Cluster", value_col]].dropna()
    plot_df["Cluster_sort"] = pd.to_numeric(plot_df["Cluster"], errors="coerce")
    plot_df = plot_df.sort_values(["Cluster_sort", "Cluster"]).drop(columns=["Cluster_sort"])

    fig = px.box(
        plot_df,
        x="Cluster",
        y=value_col,
        color="Cluster",
        points="outliers",
        title=chart_title,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def show_anomalies_over_time(df: pd.DataFrame) -> None:
    st.markdown("#### Anomalies de ventes dans le temps")
    working = get_anomaly_frame(df)
    if working is None or "Order_Date" not in working.columns:
        st.info("Graphe indisponible: colonnes Order_Date/Anomaly manquantes.")
        return

    time_df = working.dropna(subset=["Order_Date", "Revenue"]).copy()
    if time_df.empty:
        st.info("Aucune donnee temporelle disponible.")
        return

    time_df["Type"] = np.where(time_df["Anomaly"] == 1, "Anomalie", "Normal")
    fig = px.scatter(
        time_df.sort_values("Order_Date"),
        x="Order_Date",
        y="Revenue",
        color="Type",
        color_discrete_map={"Normal": "#38bdf8", "Anomalie": "#ef4444"},
        title="Anomalies de ventes dans le temps",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True)


def show_anomalies_by_month(df: pd.DataFrame) -> None:
    st.markdown("#### Nombre d'anomalies par mois")
    working = get_anomaly_frame(df)
    if working is None or "Order_Date" not in working.columns:
        st.info("Graphe indisponible: colonnes Order_Date/Anomaly manquantes.")
        return

    month_df = (
        working.dropna(subset=["Order_Date"])
        .assign(Month=lambda x: x["Order_Date"].dt.to_period("M").astype(str))
    )
    if month_df.empty:
        st.info("Aucune donnee mensuelle disponible.")
        return

    anomalies_month = (
        month_df[month_df["Anomaly"] == 1]
        .groupby("Month", as_index=False)
        .size()
        .rename(columns={"size": "Anomalies"})
        .sort_values("Month")
    )
    if anomalies_month.empty:
        st.info("Aucune anomalie detectee sur la periode.")
        return

    fig = px.bar(
        anomalies_month,
        x="Month",
        y="Anomalies",
        title="Nombre d'anomalies par mois",
        template=PLOTLY_TEMPLATE,
        color="Anomalies",
        color_continuous_scale="Reds",
    )
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def show_ai_section(df: pd.DataFrame) -> None:
    st.markdown("<div class='section-title'>Intelligence Artificielle & Analyse Avancee</div>", unsafe_allow_html=True)

    show_sales_forecast(df)
    col_left, col_right = st.columns(2)
    with col_left:
        show_customer_segmentation(df)
    with col_right:
        show_anomaly_detection(df)

    row_1_col_1, row_1_col_2 = st.columns(2)
    with row_1_col_1:
        show_cluster_distribution(df)
    with row_1_col_2:
        show_cluster_profile_heatmap(df)

    row_2_col_1, row_2_col_2 = st.columns(2)
    with row_2_col_1:
        show_cluster_boxplot(df)
    with row_2_col_2:
        show_anomalies_over_time(df)

    show_anomalies_by_month(df)


def show_empty_state() -> None:
    fig = go.Figure()
    fig.add_annotation(
        text="Aucune donnee disponible avec les filtres actuels",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=18),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=280)
    st.plotly_chart(fig, width="stretch")


def show_download_button(df: pd.DataFrame) -> None:
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Telecharger les donnees filtrees (CSV)",
        data=csv_data,
        file_name="ventes_filtrees.csv",
        mime="text/csv",
        use_container_width=True,
    )


def build_header_badges(df: pd.DataFrame) -> str:
    rows_badge = f"<span class='pill'>Lignes filtrees: {len(df):,}</span>"
    period = build_period_series(df)
    if period is not None:
        start = period.min()
        end = period.max()
        if start is not pd.NaT and end is not pd.NaT:
            period_badge = f"<span class='pill'>période: {start} -> {end}</span>"
        else:
            period_badge = ""
    else:
        period_badge = ""
    return f"{rows_badge} {period_badge}"


def main() -> None:
    st.set_page_config(
        page_title="Tableau de bord interactif des ventes",
        layout="wide",
    )

    inject_styles()

    st.markdown("<div class='dashboard-title'>Tableau de bord interactif des ventes</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dashboard-subtitle'>Dashboard BI pour analyser les ventes e-commerce a partir d'un fichier nettoye.</div>",
        unsafe_allow_html=True,
    )

    if not DATA_PATH.exists():
        st.error(f"Fichier introuvable : {DATA_PATH}")
        st.stop()

    df = load_data(str(DATA_PATH))
    filtered_df = build_sidebar_filters(df)
    targets = build_targets()

    if filtered_df.empty:
        st.warning("Aucune donnee ne correspond aux filtres selectionnes.")
        show_empty_state()
        st.dataframe(filtered_df, width="stretch", hide_index=True)
        return

    st.markdown("<div class='header-card'>", unsafe_allow_html=True)
    st.markdown(build_header_badges(filtered_df), unsafe_allow_html=True)
    st.markdown(
        "<div class='metric-note'>Synthese rapide des resultats filtres et tendances recentes.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    show_kpis(filtered_df)

    insights = build_insights(filtered_df)
    if insights:
        st.markdown(f"<div class='dashboard-subtitle'>{insights}</div>", unsafe_allow_html=True)

    show_strategic_section(filtered_df, targets)
    show_drilldown(filtered_df)

    st.markdown("<div class='section-title'>Tendances principales</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        show_monthly_sales_chart(filtered_df)
    with col2:
        show_category_sales_chart(filtered_df)

    st.markdown("<div class='section-title'>Analyse commerciale</div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        show_region_profit_chart(filtered_df)
    with col4:
        show_top_products_chart(filtered_df)

    st.markdown("<div class='section-title'>Analyse detaillee</div>", unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        show_payment_sales_chart(filtered_df)
    with col6:
        show_order_status_chart(filtered_df)

    col7, col8 = st.columns(2)
    with col7:
        show_revenue_profit_scatter(filtered_df)
    with col8:
        show_profit_boxplot(filtered_df)

    show_ai_section(filtered_df)

    st.markdown("<div class='section-title'>Carte geographique</div>", unsafe_allow_html=True)
    show_world_map(filtered_df)

    st.markdown("<div class='section-title'>Tableau final</div>", unsafe_allow_html=True)
    show_download_button(filtered_df)
    st.dataframe(filtered_df, width="stretch", height=420, hide_index=True)


if __name__ == "__main__":
    if streamlit.runtime.exists():
        main()
    else:
        print("Lancez l'application avec : python -m streamlit run app.py")
