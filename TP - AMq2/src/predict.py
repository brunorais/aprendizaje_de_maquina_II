"""
predict.py

COMPLETAR DOCSTRING

DESCRIPCIÓN: 
AUTOR: Bruno Rais
FECHA: 5 jun 2023
"""

# Imports
import pandas as pd
from pandas import DataFrame 
import joblib
from feature_engineering import data_transformation

class MakePredictionPipeline(object):
    
    def __init__(self, input_path, output_path, model_path: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
                
                
    def load_data(self) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        """
        pandas_df = pd.read_csv(self.input_path)
        data  = data_transformation(pandas_df)#No estoy seguro
        return data

    def load_model(self) -> None:
        """
        COMPLETAR DOCSTRING
        """    
        self.model = joblib.load(self.model_path) # Esta función es genérica, utilizar la función correcta de la biblioteca correspondiente
        
        return None


    def make_predictions(self, data: DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        """
   
        new_data = self.model.predict(data)

        return new_data


    def write_predictions(self, predicted_data: DataFrame) -> None:
        """
        COMPLETAR DOCSTRING
        """
        predicted_data.to_csv(self.output_path + '/predictions.csv')
        return None


    def run(self):

        data = self.load_data()
        self.load_model()
        df_preds = self.make_predictions(data)
        self.write_predictions(df_preds)


if __name__ == "__main__":
    
    spark = Spark()
    
    pipeline = MakePredictionPipeline(input_path = 'Ruta/De/Donde/Voy/A/Leer/Mis/Datos',
                                      output_path = 'Ruta/Donde/Voy/A/Escribir/Mis/Datos',
                                      model_path = 'Ruta/De/Donde/Voy/A/Leer/Mi/Modelo')
    pipeline.run()  