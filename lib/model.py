import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from dataset import Dataset


class AmazonModels():
    # todo: add fit method
    # todo: add predict method
    def __init__(self, data_path : str, test_size : float):
        # loading or initializing the dataset
        self.data_path = data_path
        if os.path.isfile(data_path.replace('.csv', '.pkl')):
            data = Dataset.load_object(data_path.replace('.csv', '.pkl'))
        else:
            data = Dataset(file_path=data_path)
        # Split the datasets into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = data.split_data(
            test_size=test_size, random_state=42)
        #
        self.mse_baseline = self.predict_baseline()

    ###################
    # Model Utilities #
    ###################

    def save_object(self):
        """
        Save the object to a pickle file
        :return:
        """
        for attribute in ['X_train', 'X_test', 'y_train', 'y_test']:
            delattr(self, attribute)
        filename = self.data_path.replace('.csv', '_model.pkl')
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_object(self, filename):
        """
        Load the object from a pickle file
        :param filename:
        :return:
        """
        with open(filename, 'rb') as input:
            return pickle.load(input)

    def predict_baseline(self):
        """
        Predict the mean of the target variable for all instances in the
        test set and calculate the mean squared error
        """
        # Calculate the mean of the target variable from the training set
        y_train_mean = np.mean(self.y_train)

        # Predict the mean for all instances in the test set
        predictions_baseline = np.full(shape=self.y_test.shape,
                                       fill_value=y_train_mean)

        # Evaluate the baseline model
        mse_baseline = mean_squared_error(self.y_test, predictions_baseline)
        print(f"Baseline Model (Mean) MSE: {mse_baseline}")
        return mse_baseline

    ##################
    # Model Training #
    ##################

    def train_linear_regression(self):
        self.lr = LinearRegression()
        self.lr.fit(self.X_train, self.y_train)
        predictions_lr = self.lr.predict(self.X_test)
        train_mse = mean_squared_error(self.y_train,
                                       self.lr.predict(self.X_train))
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
        train_mse = mean_squared_error(self.y_train, lasso.predict(
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
        train_mse = mean_squared_error(self.y_train, ridge.predict(
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
                                      random_state=random_state,
                                      subsample=subsample,
                                      max_depth=max_depth,
                                      objective='reg:squarederror')
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
                                early_stopping=True, validation_fraction=0.1,
                                alpha=0.001,
                                learning_rate_init=0.001, random_state=42)
        self.mlp.fit(self.X_train, self.y_train)
        train_mse = mean_squared_error(self.y_train,
                                       self.mlp.predict(self.X_train))
        print(f"MLP Train MSE: {train_mse}")
        predictions_mlp = self.mlp.predict(self.X_test)
        mse_mlp = mean_squared_error(self.y_test, predictions_mlp)
        print(f"MLP Test MSE: {mse_mlp}")

    def train_catboost(self):
        from catboost import CatBoostRegressor
        self.catboost = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            verbose=100
        )
        self.catboost.fit(self.X_train, self.y_train,
                          eval_set=(self.X_test, self.y_test),
                          use_best_model=True)
        train_mse = mean_squared_error(self.y_train,
                                       self.catboost.predict(self.X_train))
        print(f"CatBoost Train MSE: {train_mse}")
        predictions_cat = self.catboost.predict(self.X_test)
        mse_cat = mean_squared_error(self.y_test, predictions_cat)
        print(f"CatBoost Test MSE: {mse_cat}")

if __name__ == '__main__':
    from time import time
    from modules import *
    start_time = time()
    # #
    dataset = DATA_PATH
    model = AmazonModels(dataset, test_size=0.2)
    print("Test Size: 0.2")
    print(f"Dataset: {os.path.basename(dataset)}")
    # model.tra in_linear_regression()
    # model.train_lasso_regression()
    # model.train_ridge_regression()
    # model.train_random_forest(n_estimators=100)
    # model.train_xgboost()
    model.train_neural_network()
    model.train_catboost()
    model.save_object()
    # model = AmazonModels.load_object('static\\datasets\\amazon_reviews_5k_model.pkl')
    #
    print("--- End Script ---")
    print("--- %s seconds ---" % (time() - start_time))
