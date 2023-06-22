"""
predict.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR: Bruno Rais, Joaquín Beitia
FECHA: 5 jun 2023
"""

# Imports
import pandas as pd
import joblib
import logging


class MakePredictionPipeline(object):

    def __init__(self, input_path, output_path, model_path: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path

    def load_data(self) -> pd.DataFrame:
        """
        Load the data from the specified input path and
         return it as a DataFrame.

        Returns:
            pandas_df (pd.DataFrame): The loaded data as a DataFrame.
        """

        try:
            logging.info("Loading data from: {}".format(self.input_path))
            pandas_df = pd.read_csv(self.input_path)
            return pandas_df
        except FileNotFoundError:
            logging.error("File not found: {}".format(self.input_path))
            return pd.DataFrame()
        except Exception as e:
            logging.error("An error occurred while loading data: {}".format(str(e)))  # noqa E501
            return pd.DataFrame()

    def load_model(self) -> None:
        """
        Load the trained model from the specified model path.

        Returns:
            None
        """

        try:
            logging.info("Loading model from: {}".format(self.model_path))
            self.model = joblib.load(self.model_path)  # Use the appropriate function from the corresponding library  # noqa E501
        except FileNotFoundError:
            logging.error("File not found: {}".format(self.model_path))
        except Exception as e:
            logging.error("An error occurred while loading the model: {}".format(str(e)))  # noqa E501

    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using the trained model on the provided data.

        Args:
            data (pd.DataFrame): The data to make predictions on.

        Returns:
            pd.DataFrame: The predicted data.
        """

        try:
            logging.info("Making predictions on provided data")
            new_data = self.model.predict(data)
            return new_data
        except Exception as e:
            logging.error("An error occurred while making predictions: {}".format(str(e)))  # noqa E501
            return pd.DataFrame()

    def write_predictions(self, predicted_data: pd.DataFrame) -> None:
        """
        Write the predicted data to a CSV file.

        Args:
            predicted_data (pd.DataFrame): The predicted data.

        Returns:
            None
        """

        try:
            logging.info("Writing predictions to: {}".format(self.output_path))
            df_predicted_data = pd.DataFrame(predicted_data , columns = ['Prediction'])
            df_predicted_data.to_csv(self.output_path + '/predictions.csv')
        except Exception as e:
            logging.error("An error occurred while writing predictions: {}".format(str(e)))  # noqa E501

    def run(self):

        data = self.load_data()
        self.load_model()
        df_preds = self.make_predictions(data)
        self.write_predictions(df_preds)


if __name__ == "__main__":

    # spark = Spark()

    pipeline = MakePredictionPipeline(
                    input_path='./src/features.csv',
                    output_path='./src',
                    model_path='./src/model0.pkl')
    pipeline.run()
