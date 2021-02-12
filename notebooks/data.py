import pandas as pd

def get_ctp():
    return pd.read_csv('../datasets/ctp/ctp-all.csv')

def get_mobility():
    return pd.read_csv('../datasets/mobility/2020_US_Region_Mobility_Report.csv')

def get_mask_usage():
    return pd.read_csv('../datasets/nyt/mask-use.csv')