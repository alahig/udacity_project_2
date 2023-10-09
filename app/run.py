import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from models import custom_tokenizer
from models import sentence_meta_data_extractor

from models.custom_tokenizer import CustomTokenizer
from models.sentence_meta_data_extractor import SentenceMetaData
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table("Messages", engine)

# load model
model = joblib.load("../models/classifier_tree_150.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    # Prepare the most used words in the dataset
    text = df["message"].sum()
    stopwords = set(stopwords.words("english"))
    text = text.lower()
    text = "".join(c for c in text if c not in "().,:;!?-'\"")
    text = " ".join(w for w in text.split() if w not in stopwords)
    most_used = pd.Series(text.split(" ")).value_counts().sort_values()
    most_used = most_used.iloc[-20:]

    # Prepare correlation matrix
    Y = df.iloc[:, 5:]
    z = Y.corr()

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            "data": [Bar(x=genre_names, y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        },
        {
            "data": [Bar(x=most_used.index, y=most_used.values)],
            "layout": {
                "title": "Most used words in dataset",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Word"},
            },
        },
        {
            "data": [
                {
                    "colorscale": "RdBu_r",
                    "type": "heatmap",
                    "x": z.index,
                    "y": z.columns,
                    "z": z.values,
                }
            ],
            "layout": {
                "coloraxis": {
                    "colorbar": {
                        "tickfont": {"size": 12},
                        "title": {
                            "font": {"size": 14},
                            "text": "Correlation among categories",
                        },
                    }
                },
                "margin": {"t": 60},
                "xaxis": {"side": "bottom", "type": "category"},
                "yaxis": {"autorange": "reversed", "type": "category"},
            },
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3000, debug=True)


if __name__ == "__main__":
    main()
