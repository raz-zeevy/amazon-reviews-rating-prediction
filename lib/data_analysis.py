import pickle

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

from lib.modules import *
import pandasplus
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

PLACEHOLDER_FOR_MISSING_STRINGS = ""


def sample_data(data: pd.DataFrame, n) -> pd.DataFrame:
    return data.sample(n)


def load_data(path):
    if path == DATA_PATH:
        return pd.read_csv(path)
    else:
        return pd.read_csv(path, index_col=0)


import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK datasets
# nltk.download('punkt')
# nltk.download('stopwords')

# Compile regex patterns outside the function
RE_TAGS = re.compile(r'<.*?>')
RE_SPECIAL = re.compile(r'[/\\().,;:!?]')
RE_DESCRIPTION_CHARS = re.compile(r'[^a-zA-Z0-9\s]')

# Set of English stopwords
stop_words = set(stopwords.words('english'))


def clean_text(text, basic=False):
    # Remove HTML tags
    text = RE_TAGS.sub('', text)
    text = RE_SPECIAL.sub(' ', text)
    # Remove punctuation and numbers
    text = RE_DESCRIPTION_CHARS.sub('', text)
    if basic: return text
    # To lowercase
    text = text.lower()
    # Remove stopwords
    tokens = word_tokenize(text)
    filtered_text = ' '.join(
        [word for word in tokens if word not in stop_words])
    return filtered_text


# def feature(df):
#     df['feature_cleaned'] = df['feature'].astype(str).apply(clean_text)
#     # Step 3: Apply TF-IDF vectorization to the cleaned feature text
#     tfidf_vectorizer_feature = TfidfVectorizer(
#         max_features=500)  # Adjust max_features as needed
#     feature_tfidf_matrix = tfidf_vectorizer_feature.fit_transform(
#         df['feature_cleaned'])


##################
# Data Analysis  #
##################

def plot_brand_data(df):
    import pandas as pd
    import plotly.graph_objects as go

    # Group by 'brand' and aggregate
    grouped = df.groupby('brand').agg(
        unique_item_names=pd.NamedAgg(column='itemName', aggfunc='nunique'),
        total_rows=pd.NamedAgg(column='itemName', aggfunc='size'),
        unique_user_names=pd.NamedAgg(column='userName', aggfunc='nunique')
    )

    # Sort by 'unique_user_names' to divide the dataset
    grouped_sorted = grouped.sort_values(by='total_rows',
                                         ascending=False)

    # Divide the dataset
    top_brands = grouped_sorted.head(999)
    other_brands = grouped_sorted.iloc[999:]

    # Aggregate the two groups to get totals
    top_brands_totals = {
        'Total Unique Item Names': top_brands['unique_item_names'].sum(),
        'Total Rows': top_brands['total_rows'].sum(),
        'Total Unique User Names': top_brands['unique_user_names'].sum()
    }

    other_brands_totals = {
        'Total Unique Item Names': other_brands['unique_item_names'].sum(),
        'Total Rows': other_brands['total_rows'].sum(),
        'Total Unique User Names': other_brands['unique_user_names'].sum()
    }

    # Data for plotting
    categories = ['Total Unique Item Names', 'Total Rows',
                  'Total Unique User Names']
    top_brands_values = [top_brands_totals[category] for category in
                         categories]
    other_brands_values = [other_brands_totals[category] for category in
                           categories]

    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(name='Top 999 Brands', x=categories, y=top_brands_values),
        go.Bar(name='Other Brands', x=categories, y=other_brands_values)
    ])

    # Update layout
    fig.update_layout(barmode='group',
                      title='Comparison of Top 999 Brands vs Other '
                            'Brands (in terms of entries count)')
    fig.update_layout(
        title_font=dict(size=24),  # Increase title font size
        xaxis=dict(
            title='Category',
            title_font=dict(size=20),  # Increase x-axis title font size
            tickfont=dict(size=18)  # Increase x-axis tick font size
        ),
        yaxis=dict(
            title='Total',
            title_font=dict(size=20),  # Increase y-axis title font size
            tickfont=dict(size=18)  # Increase y-axis tick font size
        ),
        legend=dict(
            font=dict(
                size=18  # Increase legend font size
            )
        )
    )
    fig.show()


def correlation(df, col_a, col_b, method='pearson'):
    return df[col_a].corr(df[col_b],
                          method=method
                          )


def calculate_missing_values(df: pd.DataFrame):
    # Identify missing values in the dataset
    missing_values = df.isnull().sum()

    # Calculate the percentage of missing values for each column
    missing_percentage = (missing_values / len(df)) * 100

    missing_summary = pd.DataFrame(
        {'Missing Values': missing_values, 'Percentage': missing_percentage})
    missing_summary.sort_values(by="Percentage", ascending=False)
    return missing_summary


def plot_features_corr(df):
    global corr
    corrs = {}
    for col in ['featuresLength', 'featuresCount', 'price',
                'itemNameLength', 'itemNameCount', 'descriptionCount']:
        corr = correlation(df, col, 'rating', method='spearman')
        corrs[col] = corr
        print(f"Correlation between {col} and rating:"
              f" {corr:.2f}")
    # plot correlation graph for the features using plotly
    import plotly.express as px
    fig = px.bar(x=list(corrs.keys()), y=list(corrs.values()))
    fig.update_layout(title=f'Correlation between Features '
                            f'and Rating',
                      xaxis_title='Features',
                      yaxis_title='Correlation',
                      yaxis=dict(range=[-1, 1]))
    fig.show()


class Data:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = load_data(file_path)
        self.n = len(self.df)
        self.sparse_features = []
        self.dense_features = []
        self.X = None
        self.y = None
        self.prepare()
        self.save_object()

    ##################
    # Data utilities #
    ##################

    def split_data(self, test_size, random_state=42):
        return train_test_split(self.X, self.y, test_size=test_size,
                                random_state=random_state)
        # test_len = round(self.n * test_size)
        # return self.X[test_len - 1:], self.X[:test_len], \
        #     self.y[test_len - 1:], self.y[:test_len]

    def save_object(self):
        file_path = self.file_path.split('.csv')[0] + '.pkl'
        delattr(self, 'df')
        delattr(self, 'sparse_features')
        delattr(self, 'dense_features')
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_object(cls, file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    ##################
    # Pre-processing #
    ##################

    def prepare(self):
        import gensim.downloader as api
        word2vec_model = api.load("word2vec-google-news-300")
        self.convert_types()
        self.prep_missing_values()
        # self.clean_and_vectorise_textual_col('feature')
        # self.df['featuresCount'] = self.df['feature'].apply(lambda x: len(
        #     x.split("',")) if x != '[]' else 0)
        # self.df['featuresLength'] = self.df['feature'].apply(lambda x: len(x))
        # self.df['itemNameLength'] = self.df['itemName'].apply(lambda x:
        #                                                       len(x))
        # self.df['itemNameCount'] = self.df['itemName'].apply(lambda x: len(
        #     x.split()))
        # self.df[f'description_cleaned'] = self.df['description'].astype(
        #     str).apply(clean_text)
        # self.df[f'descriptionCount'] = self.df['description_cleaned'].apply(
        #     lambda x: len(x.split()))
        # self.clean_and_vectorise_textual_col('feature')
        self.df[f'reviewText_cleaned'] = self.df['reviewText'].astype(
            str).apply(clean_text)
        self.df[f'summary_cleaned'] = self.df['summary'].astype(
            str).apply(clean_text)
        self.vectorise_textual_col('reviewText', word2vec_model)
        self.vectorise_textual_col_tfidf('reviewText')
        self.vectorise_textual_col('summary', word2vec_model)
        self.vectorise_textual_col_tfidf('summary')
        self.group_and_encode_col('brand')
        self.one_hot_encode('category')
        # self.count_encode('itemName')
        # self.clean_and_vectorise_textual_col('itemName')
        column_features = self.df[['price', 'vote', 'verified']]
        self.dense_features.append(column_features.values)
        # self.sparse_features.append(sparse.csr_matrix(column_features.values))
        # self.sparse_features = column_features.values[:None]
        # self.X = sparse.hstack(self.sparse_features)
        self.X = np.hstack(self.dense_features)
        self.y = self.df['rating'].values

    def convert_types(self):
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
        :param df:
        :return:
        """
        # Drop the unnamed columns
        unnamed_columns = [col for col in self.df.columns if 'Unnamed' in col]
        self.df.drop(columns=unnamed_columns, inplace=True)

        # Handling missing values for 'price' and 'rating'
        # Now, we can safely calculate the median price and fill missing values
        median_price = self.df['price'].median()
        self.df['price'] = self.df['price'].fillna(median_price)

        # since 'rating' is the main target, we will drop rows with missing values
        self.df = self.df.dropna(subset=['rating'])

        # for 'brand', 'userName', 'reviewText', 'description', 'summary' we
        # will fill missing values with an empty string
        for col in ['brand', 'userName', 'reviewText', 'description',
                    'summary', 'itemName']:
            self.df[col] = self.df[col].fillna(PLACEHOLDER_FOR_MISSING_STRINGS)

    def vectorise_textual_col(self, col_name, word2vec_model):
        def document_vector(doc):
            # Remove out-of-vocabulary words
            doc = [word for word in doc.split() if
                   word in word2vec_model.key_to_index]
            if not doc:
                return np.zeros(word2vec_model.vector_size)
            # Return the mean of the vectors for all words in the document
            return np.mean(word2vec_model[doc], axis=0)

        # Apply text cleaning to the 'description' column
        self.df[f'{col_name}_cleaned'] = self.df[col_name].astype(
            str).apply(clean_text)

        col_vectors = self.df[f'{col_name}_cleaned'].apply(
            lambda x : document_vector(x))
        col_matrix = np.vstack(col_vectors.values)
        self.dense_features.append(col_matrix)

    def vectorise_textual_col_tfidf(self, col_name):
        # Apply text cleaning to the 'description' column
        self.df[f'{col_name}_cleaned'] = self.df[col_name].astype(
            str).apply(clean_text)
        # Initialize the TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer(
            max_features=DESCRIPTION_MAX_FEATURES)
        # Fit and transform the cleaned descriptions
        tfidf_matrix = tfidf_vectorizer.fit_transform(
            self.df[f'{col_name}_cleaned'])
        self.dense_features.append(tfidf_matrix.toarray())

    def group_and_encode_col(self, col_name):
        # Filter out rows where 'brand' is blank or null
        valid_values_df = self.df[
            self.df[col_name].notna() & (self.df[col_name] != '')]
        # Calculate brand frequencies excluding blanks
        values_counts = valid_values_df[col_name].value_counts()
        # Identify the top 999 brands
        top_values = values_counts.nlargest(BRAND_MAX_FEATURES - 1).index  #
        # Replace brands not in the top 999 with 'Other' in the original DataFrame
        self.df[f'{col_name}_grouped'] = self.df[col_name].apply(
            lambda x: x if x in top_values else (
                'Other' if x != '' and pd.notna(x) else x))
        # Initialize the encoder
        encoder = OneHotEncoder(
            sparse=False)
        # Fit and transform the grouped column
        # Ensure only rows with valid 'brand_grouped' values are encoded
        column_encoded = encoder.fit_transform(
            self.df[[f'{col_name}_grouped']])
        self.dense_features.append(column_encoded)
        return self.df

    def binary_encode(self, col_name):
        import category_encoders as ce
        encoder = ce.BinaryEncoder(cols=[col_name])
        df_binary_encoded = encoder.fit_transform(self.df[col_name])
        # self.sparse_features.append(sparse.csr_matrix(df_binary_encoded))
        self.sparse_features.append(df_binary_encoded)

    def one_hot_encode(self, col_name):
        encoder = OneHotEncoder(
            sparse=False)
        # Fit and transform the grouped column
        column_encoded = encoder.fit_transform(
            self.df[[col_name]])
        self.dense_features.append(column_encoded)
        return self.df

    def count_encode(self, col_name):
        self.df[f'{col_name}_cleaned'] = self.df[col_name].astype(str).apply(
            clean_text)
        vectorizer = CountVectorizer(
            max_features=ITEM_NAME_MAX_FEATURES)  # Adjust 'max_features' based on your needs
        X = vectorizer.fit_transform(self.df[f'{col_name}_cleaned'])
        self.sparse_features.append(X)


class AmazonModels():
    def __init__(self, data_path, test_size):
        self.data_path = data_path
        if os.path.isfile(data_path.replace('.csv', '.pkl')):
            data = Data.load_object(data_path.replace('.csv', '.pkl'))
        else:
            data = Data(file_path=data_path)
        self.X_train, self.X_test, self.y_train, self.y_test = data.split_data(
            test_size=test_size, random_state=42)
        self.mse_baseline = self.predict_baseline()

    def save_object(self):
        for attribute in ['X_train', 'X_test', 'y_train', 'y_test']:
            delattr(self, attribute)
        filename = self.data_path.replace('.csv', '_model.pkl')
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_object(self, filename):
        with open(filename, 'rb') as input:
            return pickle.load(input)

    def predict_baseline(self):
        # Calculate the mean of the target variable from the training set
        y_train_mean = np.mean(self.y_train)

        # Predict the mean for all instances in the test set
        predictions_baseline = np.full(shape=self.y_test.shape,
                                       fill_value=y_train_mean)

        # Evaluate the baseline model
        mse_baseline = mean_squared_error(self.y_test, predictions_baseline)
        print(f"Baseline Model (Mean) MSE: {mse_baseline}")
        return mse_baseline

    def train_linear_regression(self):
        self.lr = LinearRegression()
        self.lr.fit(self.X_train, self.y_train)
        predictions_lr = self.lr.predict(self.X_test)
        train_mse = mean_squared_error(self.y_train, self.lr.predict(self.X_train))
        print(f"Linear Regression Train MSE: {train_mse}")
        concat_df = np.hstack([self.X_test, predictions_lr.reshape(-1, 1),
                               self.y_test.reshape(-1, 1)])
        mse_lr = mean_squared_error(self.y_test, predictions_lr)
        print(f"Linear Regression Test MSE: {mse_lr}")
        return mse_lr

    def train_random_forest(self, n_estimators=20, random_state=42):
        # Example: Training a Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=n_estimators,
                                   random_state=random_state)
        rf.fit(self.X_train, self.y_train)
        train_mse = mean_squared_error(self.y_train, rf.predict(self.X_train))
        print(f"Random Forest Train MSE: {train_mse}")
        predictions_rf = rf.predict(self.X_test)
        mse_rf = mean_squared_error(self.y_test, predictions_rf)
        print(f"Random Forest Test MSE: {mse_rf}")

    def train_lasso_regression(self):
        # alphas = [0.01, 0.1, 1, 10, 100]
        alphas = [0.01]
        lasso = LassoCV(alphas=alphas, cv=5, random_state=42)
        lasso.fit(self.X_train, self.y_train)
        train_mse = mean_squared_error(self.y_train,lasso.predict(
            self.X_train))
        print(f"Lasso Regression a={lasso.alpha_} Train MSE: {train_mse}")
        predictions_lasso = lasso.predict(self.X_test)
        mse_lasso = mean_squared_error(self.y_test, predictions_lasso)
        print(f"Lasso Regression a={lasso.alpha_} Test MSE: {mse_lasso}")

    def train_ridge_regression(self):
        # alphas = [0.01, 0.1, 1, 10, 100]
        alphas = [100]
        ridge = RidgeCV(alphas=alphas, cv=5)
        ridge.fit(self.X_train, self.y_train)
        train_mse = mean_squared_error(self.y_train,ridge.predict(
            self.X_train))
        print(f"Ridge Regression a={ridge.alpha_} Train MSE: {train_mse}")
        predictions_ridge = ridge.predict(self.X_test)
        mse_ridge = mean_squared_error(self.y_test, predictions_ridge)
        print(f"Ridge Regression a={ridge.alpha_} Test MSE: {mse_ridge}")

    def train_xgboost(self, n_estimators=200, learning_rate=0.05,
                      max_depth=6, subsample=0.8, random_state=42):
        from xgboost import XGBRegressor
        from sklearn.model_selection import GridSearchCV

        # Example: Training an XGBoost Regressor
        self.xgb_model = XGBRegressor(n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                 random_state=random_state, subsample=subsample,
                                 max_depth=max_depth, objective='reg:squarederror')
        self.xgb_model.fit(self.X_train, self.y_train)
        # xgb_model.fit(self.X_train, self.y_train)
        y_pred_train = self.xgb_model.predict(self.X_train)
        y_pred_test = self.xgb_model.predict(self.X_test)
        # Calculate MSE
        mse_train = mean_squared_error(self.y_train, y_pred_train)
        mse_test = mean_squared_error(self.y_test, y_pred_test)
        print(f"XGB Train MSE: {mse_train}")
        print(f"XGB Test MSE: {mse_test}")

    def train_neural_network(self):
        from sklearn.neural_network import MLPRegressor
        self.mlp = MLPRegressor(hidden_layer_sizes=(1000, 500, 100),
                             max_iter=1000,
                   early_stopping=True, validation_fraction=0.1, alpha=0.001,
                   learning_rate_init=0.001, random_state=42)
        self.mlp.fit(self.X_train, self.y_train)
        train_mse = mean_squared_error(self.y_train, self.mlp.predict(self.X_train))
        print(f"MLP Train MSE: {train_mse}")
        predictions_mlp = self.mlp.predict(self.X_test)
        mse_mlp = mean_squared_error(self.y_test, predictions_mlp)
        print(f"MLP Test MSE: {mse_mlp}")

    def train_catboost(self):
        from catboost import CatBoostRegressor
        # self.catboost = CatBoostRegressor(
        #     iterations=1000,
        #     learning_rate=0.1,
        #     depth=6,
        #     loss_function='RMSE',
        #     eval_metric='RMSE',
        #     random_seed=42,
        #     verbose=100  # It will print the training log every 100 iterations
        # )
        # self.catboost.fit(self.X_train, self.y_train,
        #                     eval_set=(self.X_test, self.y_test),
        #                    use_best_model=True)
        # train_mse = mean_squared_error(self.y_train, self.catboost.predict(self.X_train))
        # print(f"CatBoost Train MSE: {train_mse}")
        # predictions_cat = self.catboost.predict(self.X_test)
        # mse_cat = mean_squared_error(self.y_test, predictions_cat)
        # print(f"CatBoost Test MSE: {mse_cat}")
        param_grid = {
            'iterations': [2000],
            'learning_rate': [0.05, 0.1],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5]  # Regularization term
        }

        # Initialize the CatBoostRegressor
        cb_model = CatBoostRegressor(loss_function='RMSE', eval_metric='RMSE',
                                     random_seed=42, verbose=100)

        # Set up GridSearchCV
        grid_search = GridSearchCV(estimator=cb_model, param_grid=param_grid,
                                   cv=3, scoring='neg_mean_squared_error',
                                   verbose=3)

        # Perform the grid search on the training datasets
        grid_search.fit(self.X_train, self.y_train,
                        # eval_set=(self.X_test, self.y_test),
                        use_best_model=True)

        # Get the best parameters and the corresponding score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f"Best parameters: {best_params}")
        print(f"Best score (negative MSE): {best_score}")

        # Train the final model using the best parameters found
        self.catboost = CatBoostRegressor(**best_params, loss_function='RMSE',
                                          eval_metric='RMSE', random_seed=42,
                                          verbose=100)
        self.catboost.fit(self.X_train, self.y_train,
                          eval_set=(self.X_test, self.y_test),
                          use_best_model=True)

        # Evaluate the model
        train_mse = mean_squared_error(self.y_train,
                                       self.catboost.predict(self.X_train))
        test_mse = mean_squared_error(self.y_test,
                                      self.catboost.predict(self.X_test))
        print(f"CatBoost Train MSE: {train_mse}")
        print(f"CatBoost Test MSE: {test_mse}")


if __name__ == '__main__':
    DESCRIPTION_MAX_FEATURES = 1000
    BRAND_MAX_FEATURES = 100
    ITEM_NAME_MAX_FEATURES = 1000
    #
    from time import time

    start_time = time()
    # #
    dataset = DATA_50K_PATH
    model = AmazonModels(dataset, test_size=0.2)
    print("Test Size: 0.2")
    print(f"Dataset: {os.path.basename(dataset)}")
    # model.train_linear_regression()
    # model.train_lasso_regression()
    # model.train_ridge_regression()
    # model.train_random_forest(n_estimators=100)
    # model.train_xgboost()
    # model.train_neural_network()
    model.train_catboost()
    # model.save_object()
    # model = AmazonModels.load_object('static\\datasets\\amazon_reviews_5k_model.pkl')
    #
    print("--- End Script ---")
    print("--- %s seconds ---" % (time() - start_time))
