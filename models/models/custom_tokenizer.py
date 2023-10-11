import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("universal_tagset")
import spacy
import re


class CustomTokenizer(BaseEstimator, TransformerMixin):

    """This class is used to generate the tokens.
    The class takes a sentence and returns a list of words.
    We want to test different ideas:


    named entities: We want to test if the models work better if we replace dates, locations, amounts by tags,
                            i.e. DATE, LOCATION, NUMBER
                    We think that the simple precence of a number is relevant, independent of the exact number
                    I.e. "Eva needs 5 litres of water in Zurich", "Albert needs 10 kilos of rice in Berlin".
                    -> "Person needs Number litres of water in Location"
                    -> "Person needs Number kilos of rice in Location"
                    For this we use spacy

    stemming: We want to test if the simple stemming (PorterStemming) is useful or too simplistic.



    Usage:

    ct = CustomTokenizer(replace_named_entities=False, use_stemming=False)
    text = 'Weather update - a cold front from Cuba that could pass over Haiti'
    ct(text)


    """

    def __init__(self, replace_named_entities=False, use_stemming=False):
        """
        Args:
            replace_named_entities (bool): Whether to replace named entities, numbers, etc. by tags.
            use_stemming (bool): Whether to use the PorterStemming Algorithm

        """

        self.replace_named_entities = replace_named_entities
        self.nlp = spacy.load("en_core_web_sm")

        self.use_stemming = use_stemming

    def __call__(self, text):
        """Tokenizes the text:
        Depending on the parameters set, the following steps are done:
        1) replaces named entities
        2) removes special chars
        3) removes stopwors
        4) Applies Porter stemming
        5) word tokenizations



        Args:
            text (str): text to process

        Returns:
            list[Str]: Tokens
        """

        if self.replace_named_entities:
            doc = self.nlp(text)
            for ele in doc.ents:
                text = text.replace(ele.text, ele.label_)

        text = text.lower()
        text = re.sub("[^a-z0-9]", " ", text)
        words = word_tokenize(text)
        words = list(filter(lambda x: not (x in stopwords.words("english")), words))

        if self.use_stemming:
            # Reduce words to their stems
            words = [PorterStemmer().stem(w) for w in words]

        return words

    def fit(self, X, y=None):
        """Not implemented."""
        return self

    def transform(self, X):
        """transforms the Series X rowwise.

        Args:
            X (pd.Series, np.array): Series to transform

        Returns:
            np.array: Transformed values
        """
        return pd.Series(X).apply(lambda x: self(x)).values
