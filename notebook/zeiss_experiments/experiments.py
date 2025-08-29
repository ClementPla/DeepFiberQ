from pathlib import Path
import pandas as pd
from dnafiber.analysis.ratios import load_experiment, normalize_df, create_violin_plot, compare_pairs, graders_statistical_test, create_boxen_plot



PATH_GT = Path("/home/clement/Documents/data/DNAFiber/GT/")
PATH_PRED = Path("/home/clement/Documents/data/DNAFiber/")

def exp1(pred_folder, gt_file):
    root = PATH_PRED / pred_folder
    path_gt = PATH_GT / gt_file
    combined_df = load_experiment(root, path_gt, filter_invalid=False)
    combined_df["Type"] = combined_df["Type"].replace({"siT+B2": "siTONSL-D+siBRCA2", "siTONS+b1": "siTONSL-D+siBRCA1"})
    order = ['siNT', 'siBRCA1', 'siBRCA2', 'siTONSL-D', 
       'siTONSL-D+siBRCA1', 'siTONSL-D+siBRCA2']
    combined_df["Type"] = pd.Categorical(combined_df["Type"], categories=order, ordered=True)
    combined_df = combined_df.sort_values("Type")

    pairs = [
    ("siNT", "siBRCA1", 2),
    ("siBRCA1", "siTONSL-D", 2),
    ("siTONSL-D", "siTONSL-D+siBRCA1", 2),
    ("siNT", "siTONSL-D", 4),
    ("siBRCA1", "siTONSL-D+siBRCA1", 8),]
    return combined_df, pairs

def exp2(pred_folder, gt_file):
    root = PATH_PRED / pred_folder
    path_gt = PATH_GT / gt_file
    combined_df = load_experiment(root, path_gt, filter_invalid=False)
    combined_df["Type"] = combined_df["Type"].replace({"sitonsl2-2_25": "siTONSL2 2.25", 
                                                    "sitonsl4_1_125": "siTONSL4 1.125",
                                                    "sitonsl4_15": "siTONSL4 15",
                                                    "sitonsl2-1_125": "siTONSL2 1.125",
                                                    "sitonsl4_2_25": "siTONSL4 2.25",
                                                    "sitonsl2-15": "siTONSL2 15",
                                                    "siTONSL 15": "siTONSL2 15",
                                                    "siTONSL 2.5": "siTONSL2 2.25",})
    
    order = ['siNT', 'siTONSL2 1.125', 'siTONSL2 2.25', 'siTONSL2 15',
         'siTONSL4 1.125', 'siTONSL4 2.25', 'siTONSL4 15']
    combined_df["Type"] = pd.Categorical(combined_df["Type"], categories=order, ordered=True)
    combined_df.sort_values("Type", inplace=True)

    pairs = [
    ("siNT", "siTONSL2 1.125", 2),
    ("siNT", "siTONSL2 2.25", 4),
    ("siNT", "siTONSL2 15", 8),
    ("siNT", "siTONSL4 2.25", 16),
    ("siNT", "siTONSL4 15", 24),]
    return combined_df, pairs

def exp3(pred_folder, gt_file):
    root = PATH_PRED / pred_folder
    path_gt = PATH_GT / gt_file
    combined_df = load_experiment(root, path_gt, filter_invalid=False)
    combined_df["Type"] = combined_df["Type"].replace({"si5+si53": "siTONSL+si53BP1"})
    order = ['siNT', 'siTONSL', 'si53BP1', 'siTONSL+si53BP1']
    combined_df["Type"] = pd.Categorical(combined_df["Type"], categories=order, ordered=True)
    combined_df.sort_values("Type", inplace=True)
    pairs = None
    return combined_df, pairs

def exp4(pred_folder, gt_files):
    root = PATH_PRED / pred_folder
    paths_gt = [PATH_GT / gt_file for gt_file in gt_files]
    combined_df = load_experiment(root, paths_gt, filter_invalid=False)
    combined_df["Type"] = combined_df["Type"].replace({"mms22l ko2": "MMS22L K0-2",
                                                   "mms22l ko1": "MMS22L K0-1",
                                                   "u2os-ctl": "U2OS-CTL",
                                                   "siNT +C5": "siNT+C5",
                                                   "siMMS22L +C5": "siMMS22L+C5"
                                                   })
    order = [ 'siNT','MMS22L K0-1', 'MMS22L K0-2', 'U2OS-CTL', 'siMMS22L',
       'siMMS22L+C5', 'siNT+C5']
    combined_df["Type"] = pd.Categorical(combined_df["Type"], categories=order, ordered=True)
    combined_df.sort_values("Type", inplace=True)
    pairs = None
    return combined_df, pairs

def exp5(pred_folder, gt_file):
    root = PATH_PRED / pred_folder
    path_gt = PATH_GT / gt_file
    combined_df = load_experiment(root, path_gt, filter_invalid=False)
    combined_df["Type"] = combined_df["Type"].replace({"siBRCA2_12_5": "siBRCA2_12.5",
                                                       "siBRCA2_7_5": "siBRCA2_7.5",
                                                       "si53bp1_12.5": "si53BP1_12.5",
                                                       "si53BP1_7_5": "si53BP1_7.5",
                                                       "si53BP1_12_5": "si53BP1_12.5",})
    print(combined_df["Type"].unique())
    order = ['siNT', 'siBRCA2_5', 'siBRCA2_7.5', 'siBRCA2_12.5', 'si53BP1_5', 'si53BP1_7.5', 'si53BP1_12.5']
    combined_df["Type"] = pd.Categorical(combined_df["Type"], categories=order, ordered=True)
    combined_df.sort_values("Type", inplace=True)
    pairs = None
    return combined_df, pairs