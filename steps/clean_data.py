import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataSplitStrategy, DataPreProcessing
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data(df: pd.DataFrame) -> Tuple[
Annotated[
    pd.DataFrame, "X_train"],
Annotated[pd.DataFrame, "X_test"],
Annotated[pd.DataFrame, "y_train"],
Annotated[pd.DataFrame, "y_test"],
]:
    
    """ 
    Cleans the data and splits it into training and testing
    
    
    Args: 
        df: Row Data
        
    Returns:
        X_train,
        X_test,
        y_train,
        y_test
    """
    try:
        pre_processing_strategy = DataPreProcessing()
        
        data_cleaning = DataCleaning(df, pre_processing_strategy)
        
        processed_data = data_cleaning.handle_data()
        
        split_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(processed_data, split_strategy)
        
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        
        logging.info("Data Cleaning Finished")
    except Exception as e:
        logging.error('Error in cleaning data: {}'.format(e))
        raise e