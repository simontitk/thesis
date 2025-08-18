import pandas as pd


class AOILoader:
    def load(excel_path: str):
        df = pd.read_excel(excel_path)
        columns = ["img", "AOI", "x", "y"]