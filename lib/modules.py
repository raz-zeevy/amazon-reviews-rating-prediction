import os
from model import AmazonModels
from dataset import Dataset
import pandas as pd

PREDICTED_RATING_COL_NAME = 'ratingPred'

def get_resource(path):
    """Get the path to the resources folder."""
    cur_path = os.path.dirname(__file__)
    path = os.path.join(cur_path, '..', path)
    # validate
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return path

def get_path(path):
    cur_path = os.path.dirname(__file__)
    return os.path.join(cur_path, '..', path)

DATA_PATH = get_resource('lib/datasets/amazon_reviews.csv')
DATA_5K_PATH = get_resource('lib/datasets/amazon_reviews_5k.csv')
DATA_50K_PATH = get_resource('lib/datasets/amazon_reviews_50k.csv')
def get_top_products(data_path, features_path, model_path, n=5):
    """
    Get the top n products from the dataset
    :param data_path:
    :param features_path:
    :param model_path:
    :param n:
    :return:
    """
    products = pd.read_csv(data_path)
    X = Dataset.load_object(features_path).X
    model = AmazonModels.load_object(model_path)
    pred_y = model.mlp.predict(X)
    top_products_indices = pred_y.argsort()[-n:]
    res = products.loc[top_products_indices]
    res[PREDICTED_RATING_COL_NAME] = pred_y[top_products_indices]
    return res

def get_top_products_display():
    res = pd.read_csv(get_resource('lib/top_products.csv'))
    res.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    return res

if __name__ == '__main__':
    MODEL_PATH = "models/amazon_reviews_model.pkl"
    PRODUCTS_PATH = "datasets\\amazon_reviews.csv"
    FEATURES_PATH = "datasets\\amazon_reviews.pkl"
    a = get_top_products(PRODUCTS_PATH, FEATURES_PATH, MODEL_PATH, 5)