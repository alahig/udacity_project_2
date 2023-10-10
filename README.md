# Disaster Response Pipeline Project


This project contains the code written to solve a Udacity problem.
The problem consists in categorizing tweets from catastrophy regions.

The data was provided by Udacity. 
The disaster data comes from Appen and contains real messages that were sent during disaster events. It is a supervised learning problem, i.e. each message belongs to one or several categories. 

We build a machine learning pipeline to categorize these events so that messages can be categorized to an appropiate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. 


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
    The accuracy is of course then low on the test set, given the limited data used.

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### Code basis

The code is organized as follows:
- The webapp is included in the app folder.
- The data folder contains the script The ETL script. The script takes the file paths of the two datasets and database, cleans the datasets, and stores the clean data into a SQLite database in the specified database file path.
The cleaning procedure splits the categories column into separate columns, converts values to binary, and drops duplicates.
- The models folder contains the script to train the model. The script train_classifier.py takes the database file path and model file path, creates and trains a classifier, and stores the classifier into a pickle file to the specified model file path.
The script uses a custom tokenize function using nltk to case normalize, lemmatize, and tokenize text. Moreover, the function replaces named entities, dates, places etc. 
The model is a RandomForest which uses TF-IDF as well as meta data of the sentence (precence of ?, length of sentence, ...). The used features are selected using GridSearchCV.

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
| - disaster_categories.csv  # data to process 
| - disaster_messages.csv  # data to process
| - process_data.py # data process script: Load, merge, clean and store the data in a database.
- models
| - train_classifier.py # training script: train the classifier, export the model to a pickle file
| - models
| | - chosen_parameters.py # The parameters over which the GridSearchCV is ran
| | - custom_tokenizer.py # The custom tokenizer which is parametrised
| | - sentence_meta_data_extractor.py # The tool used to extract meta data of the tweets (length, ...)

- README.md

