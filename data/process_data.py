import sys
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Reads both csv files (messages, categories)
      and returns them as a pandas dataframe

    Parameters:
        messages_filepath (str): filepath to messages csv file
        categories_filepath (str): filepath to categories csv file

    Returns:
        tuple of pd.DataFrame: messages, categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return messages, categories


def clean_data(messages, categories):
    """
    Cleans the data
    - Split the values in the `categories` column on the `;` character so that each value becomes a separate column. You'll find [this method](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html) very helpful! Make sure to set `expand=True`.
    - Use the first row of categories dataframe to create column names for the categories data.
    - Rename columns of `categories` with new column names.
    - merge messages and categories
    - clean for duplicates

    Parameters:
        messages (pd.DataFrame): messages dataframe
        categories (pd.DataFrame): categories dataframe

    Returns:
        merged and cleaned dataframe

    """

    #
    def convert_feature_list(li):
        """Converts a string in the format
        'related-1;request-0;offer-0;
        to a dict where the category names are the keys
        and the values are the integers.
        i.e. {'related': 1, 'request': 0, 'offer': 0}

        """
        elements = li.split(";")
        return {k.split("-")[0]: int(k.split("-")[-1]) for k in elements}

    categories = categories.set_index("id").squeeze()
    categories = pd.DataFrame(
        {i: convert_feature_list(v) for i, v in categories.items()}
    ).T

    # There are a few entries where the related value is 2. This should be considered a dataerror
    # i.e.
    # categories.loc[4145]
    # 'related-2;request-0;offer-0;aid_related-0;medical_help-0;medical_products-0;search_and_rescue-0;security-0;military-0;child_alone-0;water-0;food-0;shelter-0;clothing-0;money-0;missing_people-0;refugees-0;death-0;other_aid-0;infrastructure_related-0;transport-0;buildings-0;electricity-0;tools-0;hospitals-0;shops-0;aid_centers-0;other_infrastructure-0;weather_related-0;floods-0;storm-0;fire-0;earthquake-0;cold-0;other_weather-0;direct_report-0'
    categories = categories.clip(upper=1)

    # Merge the data
    df = messages.merge(categories, how="left", left_on="id", right_index=True)

    ### 6. Remove duplicates.

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # check number of duplicates
    assert not (df.duplicated().any())
    return df


def save_data(df, database_filename):
    """
    Saves the dataframe to a sql lite database.
    The data is saved under the table name "Messages"
    If the table already exists it is replaced.

    Parameters:
        df (pd.DataFrame): dataframe to store
        database_filename (str): the name of the sql lite database file

    Returns:
        None
    """
    ### Save the clean dataset into an sqlite database.
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql("Messages", engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        messages, categories = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(messages, categories)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
