"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    predict_vector = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    df = predict_vector
    #riders = pd.read_csv('assets/Riders.csv')
    #df = df.merge(riders,how='left', on='Rider Id')

    weekday_pickup = [1 if i < 6 else 0 for i in list(df['Pickup - Weekday (Mo = 1)'])]
    df['Weekday pickup'] = weekday_pickup
    
    time_of_day = [i for i in df['Pickup - Time']]
    rush_hour_pickup = []

    for i in range(len(time_of_day)):

        time = time_of_day[i]
        index_of_first_colon = time.find(":") 
        hours = int(time[:index_of_first_colon])
        minutes = int(time[index_of_first_colon + 1:index_of_first_colon+3])

        # if the day is a weekend, add a 0 to rush hour pickup list as there is no rush hour on a weekend. Else, classify as 
        # rush hour (1) or non rush hour (0)
        if weekday_pickup[i] == 0:
            rush_hour = 0
        elif 'AM' in time and hours in range(7,10):
            rush_hour = 1
        elif 'PM' in time and hours in range(4,7):
            rush_hour = 1
        else:
            rush_hour = 0

        rush_hour_pickup.append(rush_hour)
        
    df['Rush Hour Pickup'] = rush_hour_pickup
    
    average_rider_speed = pd.read_csv('assets/average_rider_speed.csv')
    df = df.merge(average_rider_speed,how='left', on='Rider Id')
    
    # fill in the blanks using the average of the average speeds
    average_speed = average_rider_speed['Average speed by trip'].sum()/len(average_rider_speed['Average speed by trip']) 
    df['Average speed by trip'].fillna(average_speed, inplace=True)
    
    predict_vector = df[['Temperature','Precipitation in millimeters','Pickup Lat','Pickup Long','Destination Lat','Destination Long','No_Of_Orders','Age','Average_Rating','No_of_Ratings','Weekday pickup','Rush Hour Pickup','Average speed by trip','Distance (KM)']]
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
