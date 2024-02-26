import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import gensim.downloader as api
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

#############
# Constants #
#############

# Compile regex patterns to reduce overhead
RE_TAGS = re.compile(r'<.*?>')
RE_SPECIAL = re.compile(r'[/\\().,;:!?]')
RE_DESCRIPTION_CHARS = re.compile(r'[^a-zA-Z0-9\s]')
PLACEHOLDER_FOR_MISSING_STRINGS = ""
# Set of English stopwords
STOP_WORDS = set(stopwords.words('english'))
#
TFIDF_MAX_FEATURES = 1000
BRAND_MAX_FEATURES = 100

def clean_text(text):
    # Remove HTML tags
    text = RE_TAGS.sub('', text)
    text = RE_SPECIAL.sub(' ', text)
    # Remove punctuation and numbers
    text = RE_DESCRIPTION_CHARS.sub('', text)
    text = text.lower()
    # Remove stopwords
    tokens = word_tokenize(text)
    filtered_text = ' '.join(
        [word for word in tokens if word not in STOP_WORDS])
    return filtered_text

def load_data(path):
    from lib.modules import DATA_PATH
    if path == DATA_PATH:
        return pd.read_csv(path)
    else:
        return pd.read_csv(path, index_col=0)


"""
The Dataset class is a utility class that provides methods for loading, 
processing, and saving datasets.
"""
class Dataset:
    def __init__(self, file_path, save=True):
        self.file_path = file_path
        self.df = load_data(file_path)
        self.n = len(self.df)
        self.dense_features = []
        self.X = None
        self.y = None
        self.prepare()
        if save: self.save_object()

    #####################
    # Dataset utilities #
    #####################

    def split_data(self, test_size, random_state=42):
        """
        Split the dataset into training and testing sets
        :param test_size:
        :param random_state:
        :return:
        """
        return train_test_split(self.X, self.y, test_size=test_size,
                                random_state=random_state)

    def save_object(self):
        """
        Save the object to a pickle file
        :return:
        """
        file_path = self.file_path.split('.csv')[0] + '.pkl'
        for attribute in ['df','dense_features']:
            delattr(self, attribute)
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_object(cls, file_path):
        """
        Load the object from a pickle file
        :param file_path:
        :return: Dataset object
        """
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    ##################
    # Pre-processing #
    ##################

    def prepare(self):
        """
        Prepare the dataset for training. This includes:
        - Converting types
        - Handling missing values
        - Vectorising textual columns
        - Encoding categorical columns
        - Concatenating dense features
        - Storing the input and output arrays
        """
        self.convert_types()
        self.prep_missing_values()
        #
        word2vec_model = api.load("word2vec-google-news-300")
        self.df[f'reviewText_cleaned'] = self.df['reviewText'].astype(
            str).apply(clean_text)
        self.df[f'summary_cleaned'] = self.df['summary'].astype(
            str).apply(clean_text)
        self.vectorise_textual_col('reviewText', word2vec_model)
        self.vectorise_textual_col_tfidf('reviewText')
        self.vectorise_textual_col('summary', word2vec_model)
        self.vectorise_textual_col_tfidf('summary')
        #
        self.group_and_encode_col('brand')
        self.one_hot_encode('category')
        column_features = self.df[['price', 'vote', 'verified']]
        self.dense_features.append(column_features.values)
        #
        self.X = np.hstack(self.dense_features)
        self.y = self.df['rating'].values

    def convert_types(self):
        """
        Convert the types of the columns to the appropriate types
        :return:
        """
        self.df['verified'] = self.df['verified'].apply(lambda x: 1 if x \
                                                                       == 'TRUE' else 0)
        self.df['feature'] = self.df['feature'].astype(str)
        self.df['price'] = self.df['price'].str.extract(
            r'(\d+\.\d+)').astype(float)
        for col in ['price', 'vote', 'verified', 'rating']:
            # Convert all columns to numeric, coercing errors to NaN
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    def prep_missing_values(self):
        """
        This function handles missing values and clean the dataset
        :return:
        """
        # Drop the unnamed columns
        unnamed_columns = [col for col in self.df.columns if 'Unnamed' in col]
        self.df.drop(columns=unnamed_columns, inplace=True)

        # Handling missing values for 'price'
        median_price = self.df['price'].median()
        self.df['price'] = self.df['price'].fillna(median_price)

        # since 'rating' is the main target, we will drop rows with missing
        # values
        self.df = self.df.dropna(subset=['rating'])

        # for 'brand', 'userName', 'reviewText', 'description', 'summary' we
        # will fill missing values with an empty string
        for col in ['brand', 'userName', 'reviewText', 'description',
                    'summary', 'itemName']:
            self.df[col] = self.df[col].fillna(PLACEHOLDER_FOR_MISSING_STRINGS)

    def vectorise_textual_col(self, col_name: str, word2vec_model):
        """
        This function vectorises the textual columns using word2vec
        :param col_name:
        :param word2vec_model:
        :return: None
        """

        def document_vector(doc):
            # Remove out-of-vocabulary words
            doc = [word for word in doc.split() if
                   word in word2vec_model.key_to_index]
            if not doc:
                return np.zeros(word2vec_model.vector_size)
            return np.mean(word2vec_model[doc], axis=0)

        # Apply text cleaning to the 'description' column
        self.df[f'{col_name}_cleaned'] = self.df[col_name].astype(
            str).apply(clean_text)

        col_vectors = self.df[f'{col_name}_cleaned'].apply(
            lambda x: document_vector(x))
        col_matrix = np.vstack(col_vectors.values)
        self.dense_features.append(col_matrix)

    def vectorise_textual_col_tfidf(self, col_name: str):
        """
        This function vectorises the textual columns using TF-IDF
        :param col_name:
        :return:
        """
        # Apply text cleaning
        self.df[f'{col_name}_cleaned'] = self.df[col_name].astype(
            str).apply(clean_text)
        tfidf_vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES)
        # Fit and transform the cleaned descriptions
        tfidf_matrix = tfidf_vectorizer.fit_transform(
            self.df[f'{col_name}_cleaned'])
        self.dense_features.append(tfidf_matrix.toarray())

    def group_and_encode_col(self, col_name: str):
        """
        This function groups the values of a column and encodes them
        using one-hot encoding
        :param col_name:
        :return:
        """
        # Filter out rows where the brand is blank or null
        valid_values_df = self.df[
            self.df[col_name].notna() & (self.df[col_name] != '')]
        # Identify the top n-1 brands
        values_counts = valid_values_df[col_name].value_counts()
        top_values = values_counts.nlargest(BRAND_MAX_FEATURES - 1).index  #
        # Replace brands not in the top 999 with 'Other' in the original DataFrame
        self.df[f'{col_name}_grouped'] = self.df[col_name].apply(
            lambda x: x if x in top_values else (
                'Other' if x != '' and pd.notna(x) else x))
        encoder = OneHotEncoder(
            sparse=False)
        column_encoded = encoder.fit_transform(
            self.df[[f'{col_name}_grouped']])
        self.dense_features.append(column_encoded)
        return self.df

    def one_hot_encode(self, col_name: str):
        '''
        This function encodes a column using one-hot encoding
        :param col_name:
        :return:
        '''
        encoder = OneHotEncoder(
            sparse=False)
        column_encoded = encoder.fit_transform(
            self.df[[col_name]])
        self.dense_features.append(column_encoded)
        return self.df


if __name__ == '__main__':
    # Download necessary NLTK datasets
    # nltk.download('punkt')
    # nltk.download('stopwords')
    dataset_path = 'datasets/Amazon_Reviews.csv'
    dataset = Dataset(dataset_path)
