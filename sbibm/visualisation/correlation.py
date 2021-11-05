from typing import Any, Dict, List, Optional

import altair as alt
import deneb as den
import pandas as pd


def fig_correlation(
    df: pd.DataFrame,
    metrics: List[str] = ["C2ST", "MMD", "KSD", "MEDDIST"],
    config: Optional[str] = None,
    title: Optional[str] = None,
    title_dx: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    keywords: Dict[str, Any] = {},
    style: Dict[str, Any] = {},
):
    """Plots correlation matrices"""
    keywords["sparse"] = True
    keywords["limits"] = [0.0, 1.0]
    keywords["font_size"] = 14
    keywords["rotate_outwards"] = True

    if config == "manuscript":
        style["font_family"] = "Inter"
        keywords["width"] = 200 if width is None else width
        keywords["height"] = 200 if height is None else height

    if config == "streamlit":
        keywords["width"] = 500 if width is None else width
        keywords["height"] = 500 if height is None else height
        style["font_size"] = 16

    alt.themes.enable("default")

    den.set_style(
        extra={
            "config": {
                "axisX": {
                    "domain": False,
                    "domainWidth": 0,
                    "ticks": False,
                    "tickWidth": 0,
                    "grid": False,
                },
                "axisY": {
                    "domain": False,
                    "domainWidth": 0,
                    "ticks": False,
                    "tickWidth": 0,
                    "grid": False,
                },
            }
        },
        **style,
    )

    chart = den.correlation_matrix(df, metrics=metrics, **keywords)

    if title is not None:
        chart = chart.properties(
            title={
                "text": [title],
            }
        ).configure_title(offset=10, orient="top", anchor="middle", dx=title_dx)

    if config == "manuscript":
        chart = chart.configure_text(font="Inter")

    return chart
