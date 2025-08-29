from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator


def load_gt(exp: Path):
    sheet_names = pd.ExcelFile(exp).sheet_names
    all_dfs = []
    for sheet in sheet_names:
        print(f"Loading sheet: {sheet}")
        values = pd.read_excel(exp, sheet_name=sheet, header=None)
        fiber_id = values[0].dropna().values

        length = values[1].dropna().values
        # The fiber total length is the sum of two consecutive values
        length = length[::2] + length[1::2]
        
        ratios = values[2][:len(fiber_id)].dropna().values
        df = pd.DataFrame(dict(Ratio=ratios, Length=length))
        df["acquisition"] = sheet
        all_dfs.append(df)
    df = pd.concat(all_dfs, ignore_index=True)
    df["acquisition"] = pd.Categorical(
        df["acquisition"], categories=df['acquisition'].unique())
    
    return df


def load_pred(exp: Path):
    all_files = list(exp.rglob("*.csv"))
    all_dfs = []
    for file in all_files:
        df = pd.read_csv(file)
        df = df[df["Fiber type"] == "double"]

        df["acquisition"] = "-".join(file.stem.split("-")[:-1])
        # Capitalize the acquisition name (if you want to capitalize the first letter)
        df["acquisition"] = df["acquisition"].str.upper()
        df['Length'] = df['First analog (µm)'] + df['Second analog (µm)']
        df["Ratio"] = df["Second analog (µm)"] / df["First analog (µm)"] 
        
        df = df[(df["Ratio"] > 0.25) & (df["Ratio"] < 6)]
        df.dropna(subset=["Ratio", "Length"], inplace=True)
        all_dfs.append(df)
    df = pd.concat(all_dfs, ignore_index=True)
   
    df["acquisition"] = pd.Categorical(
        df["acquisition"],)
    return df

def filter_per_acquisition(df, n, reference_col="Length"):
    """
    Filter the DataFrame to only keep n samples per acquisition.
    If an acquisition has more than n samples, we take the n samples closest to the average of the reference column.
    """
    df_filtered = pd.DataFrame()
    for acquisition in df["acquisition"].unique():
        df_acquisition = df[df["acquisition"] == acquisition].copy()
        if len(df_acquisition) <= n:
            df_filtered = pd.concat([df_filtered, df_acquisition])
        else:
            mean_value = df_acquisition[reference_col].median()
            df_acquisition.loc[:, "distance"] = np.abs(df_acquisition[reference_col] - mean_value)
            df_acquisition = df_acquisition.nsmallest(n, "distance")
            df_filtered = pd.concat([df_filtered, df_acquisition])
    return df_filtered.reset_index(drop=True)

   
def draw(df, ax, cmap, stats=True):
    ax = sns.violinplot(
        data=df,
        y="Ratio",
        x="acquisition",
        hue="acquisition",
        palette=cmap,
        bw_adjust=0.5,
        cut=1,
        linewidth=1,
        ax=ax,
        order=df.groupby("acquisition", observed=False).median("Ratio").sort_values(by="Ratio", ascending=True).index.tolist(),
    )
    if stats:
        acquisitions = df["acquisition"].sort_values(ascending=True).unique()
        pairs = [
            (acquisitions[i], acquisitions[i + 1]) for i in range(len(acquisitions) - 1)
        ]
        pairs += [
            (
                acquisitions[i],
                acquisitions[i + 2] if i + 2 < len(acquisitions) else acquisitions[i],
            )
            for i in range(len(acquisitions) - 2)
        ]
        annotator = Annotator(
            ax,
            pairs,
            data=df,
            x="acquisition",
            y="Ratio",
            order=df["acquisition"].cat.categories.tolist(),
        )
        annotator.configure(
            test="t-test_welch",
            text_format="star",
            loc="outside",
            text_offset=0.05,
            verbose=False,
            comparisons_correction="holm",
        )
        annotator.apply_and_annotate()

    ax.set_xlabel("Acquisition")
    ax.set_ylabel("Ratio")
    ax.set_title("Violin plot of ratios by acquisition")
    # Rotate x-ticks for better readability
    ax.tick_params(axis='x', rotation=25)


def rename_acquisition(df, maps: list[tuple]):
    """
    Rename the acquisition names to be more readable.
    """
    for old_name, new_name in maps:
        df['acquisition'] = df['acquisition'].str.replace(old_name, new_name)

    
    # df["acquisition"] = pd.Categorical(
    #     df["acquisition"],
    #     categories=df.groupby("acquisition")["Ratio"]
    #     .median()
    #     .sort_values(ascending=True)
    #     .index,
    #     ordered=True,
    # )

    df["acquisition"] = pd.Categorical(
        df["acquisition"],
    )

    return df