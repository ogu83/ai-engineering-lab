"""Plotly chart generation from DataFrame query results."""
from __future__ import annotations

import pandas as pd
import plotly.express as px


def build_chart(df: pd.DataFrame) -> str:
    """Generate a Plotly chart JSON string from a query result DataFrame.

    Strategy:
    - 1+ categorical column + 1+ numeric column → bar chart
    - 2+ numeric columns, no categorical → scatter plot
    - Empty, single-column, or all-categorical → returns ""

    Returns:
        Plotly JSON string, or "" if no suitable chart can be produced.
    """
    if df.empty or len(df.columns) < 2:
        return ""

    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    if not numeric_cols:
        return ""

    sample = df.head(50)

    if categorical_cols:
        fig = px.bar(sample, x=categorical_cols[0], y=numeric_cols[0])
    elif len(numeric_cols) >= 2:
        fig = px.scatter(sample, x=numeric_cols[0], y=numeric_cols[1])
    else:
        return ""

    return fig.to_json()
