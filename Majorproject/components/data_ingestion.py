import os
import sys
from Majorproject.exception import CustomException
from Majorproject.logger import logging

from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass

# Creating a configuration class using @dataclass
# This class stores all file paths related to data ingestion
@dataclass
class DataIngestionconfig:
     # Path where training dataset will be saved
    train_data_path:str=os.path.join('artifact',"train.csv")
     # Path where testing dataset will be saved  
    test_data_path:str=os.path.join('artifact',"test.csv")
     # Path where complete raw dataset will be saved
    raw_data_path:str=os.path.join('artifact',"data.csv") 

# Main Data Ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()
    # Method for starting data ingestion process
    def initiate_data_ingestion(self):
          # Logging message
        logging.info("Entered the Data ingestion method or component")
        try:
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')
            
            # Creating artifact folder if it does not exist
            # os.path.dirname extracts folder name from path
            # exist_ok=True prevents error if folder already exists
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            # Saving complete/raw dataset into artifact/data.csv
            # index=False removes row numbering column
            # header=True keeps column names
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test Split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            # Splitting dataset into train and test datasets
            # test_size=0.2 means 20% data goes to testing
            # random_state=42 ensures same split every time
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()