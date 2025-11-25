import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import List

# -------------------------
# CONFIGURACIÓN GENERAL
# -------------------------

st.set_page_config(
    page_title="Rappi – Sistema de Análisis Inteligente",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Sistema de Análisis Inteligente para Operaciones Rappi (MVP)")
st.caption("Demo en Streamlit – Explorador de datos, insights automáticos y bot conversacional.")


# 1. CONFIGURACIÓN DE COLUMNAS 


COLUMN_CONFIG = {
    "country": "COUNTRY",
    "city": "CITY",
    "zone": "ZONE",
    "zone_type": "ZONE_TYPE",
    "zone_prioritization": "ZONE_PRIORITIZATION",
    "metric_name": "METRIC",
}

WEEK_COL_PATTERN = r"^L\d+W"

# 2. HELPERS DE TRANSFORMACIÓN


def detect_week_columns(df: pd.DataFrame, pattern: str = WEEK_COL_PATTERN) -> list:
    """Detecta columnas tipo L0W_ROLL, L1W_ROLL, etc."""
    week_cols = [c for c in df.columns if re.match(pattern, c)]
    week_cols_sorted = sorted(
        week_cols,
        key=lambda x: int(re.findall(r"\d+", x)[0])
    )
    return week_cols_sorted


def melt_weeks(df: pd.DataFrame, value_col_name: str = "value") -> pd.DataFrame:
    """Convierte formato wide a long con número de semana."""
    week_cols = detect_week_columns(df)
    if not week_cols:
        st.warning("No se encontraron columnas de semanas en el dataset.")
        return df.copy()

    id_vars = [c for c in df.columns if c not in week_cols]

    long_df = df.melt(
        id_vars=id_vars,
        value_vars=week_cols,
        var_name="week_label",
        value_name=value_col_name,
    )

    long_df["week_number"] = long_df["week_label"].str.extract(r"L(\d+)W").astype(int)
    return long_df


def compute_anomalies(long_df: pd.DataFrame,
                      group_cols: list,
                      value_col: str,
                      pct_threshold: float = 0.1) -> pd.DataFrame:
    """Detecta cambios semana a semana mayores a 'pct_threshold'."""
    if long_df.empty:
        return long_df

    df_sorted = long_df.sort_values(group_cols + ["week_number"]).copy()
    df_sorted["prev_value"] = df_sorted.groupby(group_cols)[value_col].shift(1)
    df_sorted["delta_abs"] = df_sorted[value_col] - df_sorted["prev_value"]
    df_sorted["delta_pct"] = df_sorted["delta_abs"] / df_sorted["prev_value"]

    anomalies = df_sorted[
        df_sorted["prev_value"].notna()
        & (df_sorted["delta_pct"].abs() >= pct_threshold)
    ].copy()

    anomalies["tipo"] = np.where(
        anomalies["delta_pct"] > 0,
        "mejora",
        "deterioro"
    )
    return anomalies


def compute_trend(long_df: pd.DataFrame,
                  group_cols: list,
                  value_col: str) -> pd.DataFrame:
    """Calcula una pendiente simple por grupo usando regresión lineal."""
    if long_df.empty:
        return long_df

    def slope_func(g: pd.DataFrame) -> float:
        if g["week_number"].nunique() < 2:
            return np.nan
        x = g["week_number"].values
        y = g[value_col].values
        x_mean, y_mean = x.mean(), y.mean()
        num = ((x - x_mean) * (y - y_mean)).sum()
        den = ((x - x_mean) ** 2).sum()
        if den == 0:
            return np.nan
        return num / den

    trend_df = (
        long_df
        .groupby(group_cols, as_index=False)
        .apply(lambda g: pd.Series({"slope": slope_func(g)}))
    )
    return trend_df


# ============================================================
# 3. BOT CONVERSACIONAL 
# ============================================================

def guess_metric_from_question(question: str, metrics_df: pd.DataFrame) -> str:
    """Intenta identificar la métrica en la pregunta."""
    col_metric = COLUMN_CONFIG["metric_name"]
    if col_metric not in metrics_df.columns:
        return ""

    q = question.lower()
    metric_names = metrics_df[col_metric].dropna().unique().tolist()

    best_match = ""
    for m in metric_names:
        m_str = str(m)
        m_lower = m_str.lower()
        if m_lower in q:
            best_match = m_str
            break
    return best_match


def top_zones_for_metric(metrics_long: pd.DataFrame,
                         metric_name: str,
                         top_n: int = 5,
                         week_number: int = 0) -> pd.DataFrame:
    col_metric = COLUMN_CONFIG["metric_name"]
    col_country = COLUMN_CONFIG["country"]
    col_city = COLUMN_CONFIG["city"]
    col_zone = COLUMN_CONFIG["zone"]

    df = metrics_long.copy()
    df = df[
        (df[col_metric] == metric_name)
        & (df["week_number"] == week_number)
    ]

    df = df[[col_country, col_city, col_zone, "value"]].dropna(subset=["value"])
    df = df.sort_values("value", ascending=False).head(top_n)
    return df


def compare_zone_type_for_metric(
    metrics_long: pd.DataFrame,
    metric_name: str,
    week_number: int = 0
) -> pd.DataFrame:
    col_metric = COLUMN_CONFIG["metric_name"]
    col_zone_type = COLUMN_CONFIG["zone_type"]

    if col_zone_type not in metrics_long.columns:
        return pd.DataFrame()

    df = metrics_long[
        (metrics_long[col_metric] == metric_name)
        & (metrics_long["week_number"] == week_number)
    ].copy()

    comp = (
        df.groupby(col_zone_type)["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "avg_value"})
    )
    return comp


def chat_answer(question: str,
                metrics_df: pd.DataFrame,
                metrics_long: pd.DataFrame) -> str:
    """Motor de respuesta simple basado en reglas."""
    q = question.lower()
    metric = guess_metric_from_question(question, metrics_df)

    if metric:
        if "top" in q or "mayor" in q or "más altos" in q or "mas altos" in q:
            match_n = re.search(r"top\s+(\d+)", q)
            top_n = int(match_n.group(1)) if match_n else 5
            top_df = top_zones_for_metric(metrics_long, metric, top_n=top_n, week_number=0)

            if top_df.empty:
                return f"No encontré datos suficientes para la métrica **{metric}** en la semana más reciente."

            lines = [f"Estas son las {len(top_df)} zonas con mayor valor en **{metric}** (semana más reciente):\n"]
            for _, row in top_df.iterrows():
                lines.append(
                    f"- {row[COLUMN_CONFIG['country']]} – {row[COLUMN_CONFIG['city']]} – "
                    f"{row[COLUMN_CONFIG['zone']]}: {row['value']:.3f}"
                )
            return "\n".join(lines)

        if "wealthy" in q or "non wealthy" in q or "tipo de zona" in q:
            comp_df = compare_zone_type_for_metric(metrics_long, metric, week_number=0)
            if comp_df.empty:
                return "No encontré la columna de tipo de zona o no hay datos suficientes para comparar."

            lines = [f"Comparación de **{metric}** por tipo de zona (promedio, semana más reciente):\n"]
            for _, row in comp_df.iterrows():
                lines.append(f"- {row[COLUMN_CONFIG['zone_type']]}: {row['avg_value']:.3f}")
            return "\n".join(lines)

    return (
        "Aún no tengo un motor de lenguaje natural completo en este MVP, "
        "pero puedo ayudarte con:\n\n"
        "- *Top zonas* para una métrica (ej. `top 5 zonas con mayor Lead Penetration`).\n"
        "- Comparaciones por tipo de zona (ej. `comparar Perfect Orders entre zonas Wealthy y Non Wealthy`).\n\n"
        "Intenta reformular la pregunta incluyendo el nombre de la métrica tal como aparece en la columna `METRIC` 😉."
    )


# 4. CARGA DE DATOS


st.sidebar.header("⚙️ Configuración de datos")

st.sidebar.markdown("Carga el archivo CSV:")

metrics_file = st.sidebar.file_uploader(
    "Dataset de métricas (RAW_INPUT_METRICS)",
    type=["csv"],
    key="metrics_file",
)

metrics_df = None
metrics_long = None

if metrics_file is not None:
    metrics_df = pd.read_csv(metrics_file)
    metrics_long = melt_weeks(metrics_df, value_col_name="value")

if metrics_df is None:
    st.info("⬅️ Carga el CSV en la barra lateral para habilitar la aplicación.")
    st.stop()


# 5. ENTRADA DEL CHAT - FUERA DE LOS TABS


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.chat_input("Haz una pregunta sobre los datos...")

# ============================
# 6. TABS PRINCIPALES
# ============================

tab1, tab2, tab3 = st.tabs([
    "🔍 Explorador de datos",
    "📈 Insights automáticos",
    "🤖 Bot conversacional"
])

# TAB 1 - Explorador
with tab1:
    st.subheader("🔍 Explorador de datos")
    st.markdown("### Dataset de métricas (wide – RAW_INPUT_METRICS)")
    st.dataframe(metrics_df.head(50), use_container_width=True)

    st.markdown("### Dataset de métricas (long, por semana)")
    if metrics_long is not None:
        st.dataframe(metrics_long.head(50), use_container_width=True)

# TAB 2 - Insights
with tab2:
    st.subheader("📈 Insights automáticos sobre métricas")

    col1, col2 = st.columns(2)

    with col1:
        selected_metric = st.selectbox(
            "Selecciona una métrica para analizar:",
            sorted(metrics_df[COLUMN_CONFIG["metric_name"]].dropna().unique().tolist()),
        )
        pct_threshold = st.slider(
            "Umbral de anomalía (% cambio semana a semana)",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
        )
    with col2:
        country_filter = st.selectbox(
            "Filtra por país (opcional):",
            options=["(Todos)"] + sorted(metrics_df[COLUMN_CONFIG["country"]].dropna().unique().tolist()),
            index=0
        )

    m_long = metrics_long.copy()
    col_metric = COLUMN_CONFIG["metric_name"]
    col_country = COLUMN_CONFIG["country"]

    m_long = m_long[m_long[col_metric] == selected_metric]

    if country_filter != "(Todos)":
        m_long = m_long[m_long[col_country] == country_filter]

    if m_long.empty:
        st.warning("No hay datos para los filtros seleccionados.")
    else:
        group_cols = [
            COLUMN_CONFIG["country"],
            COLUMN_CONFIG["city"],
            COLUMN_CONFIG["zone"],
        ]

        anomalies_df = compute_anomalies(
            m_long,
            group_cols=group_cols + [col_metric],
            value_col="value",
            pct_threshold=pct_threshold / 100.0,
        )

        st.markdown("### 🔔 Anomalías semana a semana")
        if anomalies_df.empty:
            st.success("No se detectaron anomalías con el umbral seleccionado.")
        else:
            st.dataframe(anomalies_df.head(20), use_container_width=True)

        st.markdown("### 📉 Tendencias (slope por zona)")
        trends_df = compute_trend(
            m_long,
            group_cols=group_cols + [col_metric],
            value_col="value"
        )
        st.dataframe(trends_df.head(20), use_container_width=True)

# TAB 3 - Bot
with tab3:
    st.subheader("🤖 Bot conversacional")

    for role, content in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.markdown(content)

    if user_input:
        st.session_state["chat_history"].append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        answer = chat_answer(user_input, metrics_df, metrics_long)

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state["chat_history"].append(("assistant", answer))

    st.markdown("---")
    st.markdown(
        "💡 *Ejemplos de preguntas soportadas:*\n"
        "- `top 5 zonas con mayor Lead Penetration`\n"
        "- `top 10 zonas con mayor Perfect Orders`\n"
        "- `comparar Perfect Orders entre zonas Wealthy y Non Wealthy`\n"
    )
