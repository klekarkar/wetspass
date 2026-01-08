#import necessary libraries
import pandas as pd
import numpy as np
import os
import glob

#---
def read_lab_weights(src, filename):
    """ 
    Read lab weights data from a CSV file, parse date and time columns, and set datetime as index.
    Parameters:
    src (str): Source directory where the CSV file is located.
    filename (str): Name of the CSV file to read.
    Returns:
    pd.DataFrame: DataFrame with datetime index and weight data.
    
    """

    df = pd.read_csv(os.path.join(src, filename))

    # Convert 'date' and 'time' columns to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
    df['time'] = pd.to_datetime(df['time'], format='%H:%M', errors='coerce').dt.time

    # Combine 'date' and 'time' into a single datetime column
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')

    # Drop the original 'date' and 'time' columns
    df = df.drop(columns=['date', 'time'])

    # Set index to datetime
    df.set_index('datetime', inplace=True)
    
    return df
#---------------------------------------------------------------------------------------------------------------

def read_TOMST_data(data_dir, frequency):

    """ 
    Function to read TOMST temperature and soil moisture data from a specified directory,
    resample it to a given frequency, and return a list of logger names and the resampled data.
    Parameters:
    - data_dir (str): Path to the directory containing TOMST data files.
    - frequency (str): Resampling frequency (e.g., 'D' for daily, 'H' for hourly).
    Returns:
    - loggers (list): List of logger serial numbers extracted from file names.
    - micro_climate_data (DataFrame): Resampled microclimate data for all loggers.
    """
    #from the data directory import data from all loggers. Files with data are labelled beginning with data--
    data_files=glob.glob(os.path.join(data_dir,'data*')) #load all files starting with rf in the folder

    #extract the logger names from the file names
    loggers=[]
    for data_file in data_files:
        names=os.path.basename(data_file).split('_')[1]
        loggers.append(names)

    #   Define column names
    col_names=['date','timezone','temp_6cm','temp_0cm','temp_12cm','rawsm']

    resampled_datalist=[]  #stores the resampled data for each logger

    for st in data_files:
        logger_data=pd.read_csv(st, sep=';', header=None,low_memory=False)
        logger_data=logger_data.iloc[:,1:7]
        logger_data.columns=col_names

        #Convert date column to datetime
        #check if date contains . or / and apply the correct format
        mask_dot = logger_data['date'].str.contains(r"\.") #check if date contains .
        mask_slash = logger_data['date'].str.contains(r"/") #check if date contains /

        logger_data.loc[mask_dot, 'datetime'] = pd.to_datetime(logger_data.loc[mask_dot, 'date'], errors='coerce')
        logger_data.loc[mask_slash, 'datetime'] = pd.to_datetime(logger_data.loc[mask_slash, 'date'], errors='coerce')

        #set time column to datetime
        logger_data_timeindex=logger_data.set_index('datetime').drop('date', axis=1)
        logger_data_timeindex.replace(',', '.', regex=True, inplace=True) #replace , with . in Temp values
        logger_data_timeindex=logger_data_timeindex.astype(float)
        logger_data_timeindex=logger_data_timeindex.drop('timezone', axis=1)

        #resample to hourly
        resampled_data=logger_data_timeindex.resample(frequency).mean(numeric_only=True) #choose the frequency of resampling, e.g. 'D' for daily, 'H' for hourly
        resampled_data['logger_id']= os.path.basename(st).split('_')[1]
        resampled_datalist.append(resampled_data)

    micro_climate_data=pd.concat(resampled_datalist)
    micro_climate_data.sort_index(inplace=True)

    return loggers, micro_climate_data