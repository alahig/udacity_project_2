import sys
from models.custom_tokenizer import CustomTokenizer
from models.sentence_meta_data_extractor import SentenceMetaData

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
import re
import pickle
from sklearn.externals import joblib

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages', engine)#.iloc[:300, :6]

    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names




def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)
    classification = dict()
    for c in y_pred.columns:
        comp = (Y_test[c], y_pred[c])
        classification[(c, 'precision')] = precision_score(*comp)
        classification[(c, 'recall')] = recall_score(*comp)
        classification[(c, 'f1')] = f1_score(*comp)


    classification = pd.Series(classification).unstack()
    print(classification)





def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = joblib.load(model_filepath)
        
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)


        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()