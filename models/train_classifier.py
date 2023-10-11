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


def load_data(database_filepath, fast=False):
    """Connects to the database and reads the message and category data.
    Function returns the messages, the categories and the names oif the categories.



    Args:
        database_filepath (str): name of the database
        fast (bool, optional): Use only subset of data
                                and categories to speed up the
                                demonstration that the code works.
                                Defaults to False.

    Returns:
        tuple X (pd.DataFrame), Y (pd.DataFrame), category_names (list)
    """
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table("Messages", engine)
    if fast:
        print(
            "Flag fast was passed, only using a small subset of the data and features"
        )
        # Use only subset of data and categories to demonstrate that the code works.
        df = df.iloc[:1000, :10]

    X = df["message"]
    Y = df.drop(["id", "message", "original", "genre"], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def build_model():
    """Builds the classifier¨
      The classifier is a RandomForest using:
      - nlp pipeline (TFID)
      - meta data (lenght of the tweet, precence of ?, ...)
    The classifier is wrapped with a GridSearchCV in order
    to run the parameter search.

    Returns:
        GridSearchCV: classifier to be fitted
    """
    tok = CustomTokenizer()
    cv = CountVectorizer(tokenizer=tok)
    tf = TfidfTransformer()
    cl = MultiOutputClassifier(RandomForestClassifier(), n_jobs=8)

    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("nlp_pipeline", Pipeline([("count", cv), ("tfid", tf)])),
                        ("word_type_counter", SentenceMetaData()),
                    ]
                ),
            ),
            ("classifier", cl),
        ]
    )

    from models.chosen_parameters import parameters

    clf = GridSearchCV(pipeline, parameters)
    return clf


def evaluate_model(model, X_test, Y_test, category_names):
    """Runs the model on X_test.
    Compares the results to Y_test and prints the precision, recall and f1
    for each of the categories.

    Args:
        model (GridSearchCV): The model to test
        X_test (pd.DataFrame): the messages
        Y_test (pd.DataFrame): the correct labels
        category_names (list): list with the names of the message categories (same lenght as the columns of Y_test)
    """
    y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)
    classification = dict()
    for c in y_pred.columns:
        comp = (Y_test[c], y_pred[c])
        classification[(c, "precision")] = precision_score(*comp)
        classification[(c, "recall")] = recall_score(*comp)
        classification[(c, "f1")] = f1_score(*comp)

    classification = pd.Series(classification).unstack()
    print(classification)


def save_model(model, model_filepath):
    """Saves the model to the given file.
       If the file exists its overwritten.

    Args:
        model (GridSearchCV): model to store
        model_filepath (str): name of the file to use as storage
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) in [3, 4]:
        if len(sys.argv) == 4:
            assert sys.argv[3] == "fast", f"Invalid flag: {sys.argv[3]}"
            # Use this to accelerate the training by using only few datapoints
            # Can be used to demonstrate the code works.
            fast = True
        else:
            fast = False

        database_filepath, model_filepath = sys.argv[1:3]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath, fast=fast)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
