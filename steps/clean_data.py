from src.data_cleaning import DataDivideStrategy, DataPreProcessStrategy, DataCleaning
import logging
from typing_extensions import Annotated
from typing import Tuple
import pandas as pd
from zenml import step

@step 
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.DataFrame, "y_train"],
    Annotated[pd.DataFrame, "y_test"],

]:
    try:
        process_strategy  = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed")
        y_train = y_train.to_frame()
        y_test = y_test.to_frame()
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error in data cleaning: {}".format(e))
        raise e

