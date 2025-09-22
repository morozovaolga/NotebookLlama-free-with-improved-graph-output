import sys
import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from instrumentation import OtelTracesSqlEngine
from sqlalchemy import text

load_dotenv()

sql_engine = OtelTracesSqlEngine(
    engine_url=f"postgresql+psycopg2://{os.getenv('pgql_user')}:{os.getenv('pgql_psw')}@localhost:5432/{os.getenv('pgql_db')}",
    table_name="agent_traces",
    service_name="agent.traces",
)


def display_sql() -> pd.DataFrame:
    query = """CREATE TABLE IF NOT EXISTS agent_traces (
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    parent_span_id TEXT NULL,
    operation_name TEXT NOT NULL,
    start_time BIGINT NOT NULL,
    duration INTEGER NOT NULL,
    status_code TEXT NOT NULL,
    service_name TEXT NOT NULL
    );"""
    sql_engine.execute(text(query))
    return sql_engine.to_pandas()


def filter_traces(sql_query: str):
    df = sql_engine.execute(text(sql_query), return_pandas=True)
    return df


def create_latency_chart(df: pd.DataFrame):
    """Create a line chart showing latency (duration) over time"""
    if df.empty:
        st.warning("No data available for latency chart")
        return

    # Convert start_time from nanoseconds to datetime
    df_chart = df.copy()
    progressive_count = list(range(0, len(df_chart["start_time"])))
    df_chart["progressive_count"] = progressive_count

    fig = px.line(
        df_chart,
        x="progressive_count",
        y="duration",
        title="Latency Overview",
        labels={"duration": "Duration (ns)", "timestamp": "Time"},
        hover_data=["operation_name", "status_code"],
    )

    fig.update_layout(
        xaxis_title="Time", yaxis_title="Duration (nanoseconds)", hovermode="x unified"
    )

    st.plotly_chart(fig)


def create_status_pie_chart(df: pd.DataFrame):
    """Create a pie chart showing status code distribution"""
    if df.empty:
        st.warning("No data available for status code chart")
        return

    # Count status codes
    status_counts = df["status_code"].value_counts()

    # Map common status codes to more readable labels
    status_labels = {
        "OK": "OK",
        "ERROR": "ERROR",
        "UNSET": "UNSET",
        "200": "OK (200)",
        "500": "ERROR (500)",
        "404": "ERROR (404)",
    }

    # Create labels and values for the pie chart
    labels = [status_labels.get(status, status) for status in status_counts.index]
    values = status_counts.values

    # Define colors
    colors = []
    for status in status_counts.index:
        if status in ["OK", "200"]:
            colors.append("#28a745")  # Green for OK
        elif status in ["ERROR", "500", "404"]:
            colors.append("#dc3545")  # Red for ERROR
        else:
            colors.append("#6c757d")  # Gray for others

    fig = go.Figure(
        data=[go.Pie(labels=labels, values=values, hole=0.3, marker_colors=colors)]
    )

    fig.update_layout(
        title="Status Code Distribution",
        annotations=[dict(text="Status", x=0.5, y=0.5, font_size=20, showarrow=False)],
    )

    st.plotly_chart(fig)


# Streamlit UI
st.set_page_config(page_title="NotebookLlaMa - Observability Dashboard", page_icon="🔍")

st.sidebar.header("Observability Dashboard🔍")
st.sidebar.info("To switch to the other pages, select them from above!🔺")
st.markdown("---")
st.markdown("## NotebookLlaMa - Observability Dashboard🔍")

# Get the data
df_data = display_sql()

# Charts section
st.markdown("## 📊 Analytics Overview")

if not df_data.empty:
    col1, col2 = st.columns(2)

    with col1:
        create_latency_chart(df_data)

    with col2:
        create_status_pie_chart(df_data)
else:
    st.info("No trace data available yet. Charts will appear once data is collected.")

st.markdown("---")

# SQL Query section
st.markdown("### SQL Query")
sql_query = st.text_input(label="")

st.markdown("## Traces Table")
dataframe = st.dataframe(data=df_data)

if st.button("Run SQL query", type="primary"):
    if sql_query.strip():
        try:
            filtered_df = filter_traces(sql_query=sql_query)
            st.markdown("### Query Results")
            dataframe = st.dataframe(data=filtered_df)

            # Update charts with filtered data
            if not filtered_df.empty:
                st.markdown("### Updated Charts")
                col1, col2 = st.columns(2)

                with col1:
                    create_latency_chart(filtered_df)

                with col2:
                    create_status_pie_chart(filtered_df)
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")
    else:
        st.warning("Please enter a SQL query")
