import pandas as pd
from dnafiber.analysis.const import palette
import plotly.express as px


def get_color_association(df):
    """
    Get the color association for each image in the dataframe.
    """
    unique_name = df["image_name"].unique()
    color_association = {i: p for (i, p) in zip(unique_name, palette)}
    return color_association


def plot_ratio(df, color_association=None, only_bilateral=True):
    df = df[["ratio", "image_name", "fiber_type"]].copy()

    df["Image"] = df["image_name"]
    df["Fiber Type"] = df["fiber_type"]
    df["Ratio"] = df["ratio"]
    if only_bilateral:
        df = df[df["Fiber Type"] == "double"]

    df = df.sort_values(
        by=["Image", "Fiber Type"],
        ascending=[True, True],
    )

    # Order the dataframe by the average ratio of each image
    image_order = (
        df.groupby("Image")["Ratio"].median().sort_values(ascending=True).index
    )
    df["Image"] = pd.Categorical(df["Image"], categories=image_order, ordered=True)
    df.sort_values("Image", inplace=True)
    if color_association is None:
        color_association = get_color_association(df)
    unique_name = df["image_name"].unique()
    color_association = {i: p for (i, p) in zip(unique_name, palette)}

    this_palette = [color_association[i] for i in unique_name]
    fig = px.violin(
        df,
        y="Ratio",
        x="Image",
        color="Image",
        color_discrete_sequence=this_palette,
        box=True,  # draw box plot inside the violin
        points="all",  # can be 'outliers', or False
    )

    # Make the fig taller

    fig.update_layout(
        height=500,
        width=1000,
        title="Ratio of green to red",
        yaxis_title="Ratio",
        xaxis_title="Image",
        legend_title="Image",
    )
    return fig
