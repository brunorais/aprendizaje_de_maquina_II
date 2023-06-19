"""
train.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR: Bruno Rais, Joaquín Beitia
FECHA: 5 jun 2023
"""

# Imports
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import logging


class ModelTrainingPipeline(object):

    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:
        """
        Reads the desired DataLake table from a CSV file into a DataFrame.

        Returns:
            pandas_df: The desired DataLake table as a DataFrame.

        Return Type:
            pd.DataFrame
        """

        logging.info("Reading data from CSV: {}".format(self.input_path))
        pandas_df = pd.read_csv(self.input_path)

        return pandas_df

    def model_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trains a linear regression model using the provided DataFrame.

        Args:
            df: Input DataFrame containing the training data.

        Returns:
            Trained linear regression model.

        Return Type:
            sklearn.linear_model.LinearRegression
        """

        logging.info("Model training: Training linear regression model")
        model = LinearRegression()

        # División de dataset de entrenaimento y validación
        X = df.drop(columns='Item_Outlet_Sales')
        y = df['Item_Outlet_Sales']

        # Entrenamiento del modelo
        model.fit(X, y)

        # coef = pd.DataFrame(X.columns, columns=['features'])
        # coef['Coeficiente Estimados'] = model.coef_

        return model

    def model_dump(self, model_trained) -> None:
        """
        Saves the trained model to a file using joblib.

        Args:
            model_trained: Trained model to be saved.

        Returns:
            None
        """

        logging.info("Model dump: Saving trained model to file: {}".format(self.model_path + '/model0.pkl'))  # noqa E501

        joblib.dump(model_trained, self.model_path + '/model0.pkl')

        return None

    def run(self):

        df = self.read_data()
        model_trained = self.model_training(df)
        self.model_dump(model_trained)


if __name__ == "__main__":

    model = ModelTrainingPipeline(
                input_path='./src/features.csv',
                model_path='./src/'
            )
    model.run()
