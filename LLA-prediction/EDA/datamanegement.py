import pandas as pd

def load_dataset(path, type, separator="\t"):
    """
        This function is used to load datas.
        Parameters:
            path (str): The path to the dataset.
            type (str): The format used to stock data. It can be csv, tsv, json, excel, text, or json
            separator (str): The separator used to stock data.
        Returns:
            Returns a DataFrame, describing datas.
    """
    if(type == "csv"):
        return pd.read_csv(path)
    elif(type == "tsv"):
        return pd.read_csv(path, sep=separator)
    elif(type == "json"):
        return pd.read_json(path)
    elif(type == "excel"):
        return pd.read_excel(path)
    elif(type == "text"):
        return pd.read_table(path, separator)
