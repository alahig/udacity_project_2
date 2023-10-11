import nltk
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag.mapping import _UNIVERSAL_TAGS


class SentenceMetaData(BaseEstimator, TransformerMixin):

    """
    This class is used to add features apart from the TfidfTransformer.

    We have the following ideas, which we want to have parametrised, so that we can test them
    automatically using GridSearchCV:

    - pct_capital_letters: What is the percentage of capital letters in a message.
        The idea is that people may use capital letters to add importance to a message:
        i.e. "HELP, I NEED IMMEDIATE ATTENTION"
    - question/exclamation mark: we want to test if the presence of "?" or "!" has implications
    - lenght: we want to test if the lenght of the tweet has implications
    - pct_word_types: We want to see the percentage distribution of words in the tweets.
            For this we use nltk.pos_tag. A sentence with a lot of verbs may be rather
            information/description than emergency.


    Usage:

        stm = SentenceMetaData(use_pct_capital_letters=False, use_question_mark=True)
        text = 'Weather update - a cold front from Cuba that could pass over Haiti'
        stm(text)


    """

    def __init__(
        self,
        use_pct_capital_letters=True,
        use_question_mark=True,
        use_exclamation_mark=True,
        use_pct_word_types=True,
        use_length=True,
    ):
        """
        Args:
            use_pct_capital_letters (bool): Whether to count the percentage of capital letters
            use_question_mark (bool): Whether to check for the presence of ?
            use_exclamation_mark (bool): Whether to check for the presence of !
            use_pct_word_types (bool): Whether to add the distribution of the word types (verbs, nouns, ...)
            use_length (bool): Whether to add the lenght of the sentence.

        """

        self.use_pct_capital_letters = use_pct_capital_letters
        self.use_question_mark = use_question_mark
        self.use_exclamation_mark = use_exclamation_mark
        self.use_pct_word_types = use_pct_word_types
        self.use_lenght = use_length

        self.target_tags = list(_UNIVERSAL_TAGS)

    def __call__(self, text):
        """Extracts the metadata. Metadata is stored in a Series.
            Depending on the parameters used to initialize
            the SentenceMetaData the extracted meta data
            changes (See description of class and __init__)

        Args:
            text (str): text to process

        Returns:
            pd.Series: series containing the results
        """

        def pct_capital_letters(t):
            try:
                r = len(re.findall("[A-Z]", t)) / len(t)
            except:
                r = 0

            return r

        def question_mark(t):
            return "?" in t

        def exclamation_mark(t):
            return "!" in t

        def pct_word_types(t):
            t = t.strip()
            if len(t) == 0:
                return pd.Series()
            text = word_tokenize(t)
            ser = pd.DataFrame(nltk.pos_tag(text, tagset="universal"))[1].value_counts()
            return (ser / ser.sum()).reindex(self.target_tags).fillna(0)

        result = dict()
        if self.use_pct_capital_letters:
            result["pct_capital_letters"] = pct_capital_letters(text)

        if self.use_question_mark:
            result["question_mark"] = question_mark(text)

        if self.use_exclamation_mark:
            result["exclamation_mark"] = exclamation_mark(text)

        if self.use_lenght:
            result["lenght"] = len(text)

        if self.use_pct_word_types:
            result.update(pct_word_types(text))

        return pd.Series(result).astype(float)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """transforms the Series X rowwise.

        Args:
            X (pd.Series, np.array): Series to transform

        Returns:
            np.array: Transformed values
        """
        return pd.Series(X).apply(lambda x: self(x)).values
