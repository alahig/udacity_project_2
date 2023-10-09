# Disaster Response Pipeline Project


This project contains the code written to solve a Udacity problem.
The problem consists in categorizing tweets from catastrophy regions.
It is a supervised learning problem. 
The data was provided by Udacity. 

## Requirements

To run the notebook, you will need the following:

- Python 3.6 or higher
- Jupyter Notebook
- Numpy
- Pandas
- Matplotlib
- Seaborn

## Installation

To install the required packages, you can use pip or conda. For example, with pip you can run:

bash
./setup.sh


### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
    Training takes quite a while. To demonstrate the functionality of the code there is the possibility
    to pass a flag "fast" so that the training is only done on a subpart of the data (first 1000 rows, 5 features).
    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl fast`
    The accuracy is of course then low on the test set, given the limted availability of data.

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

