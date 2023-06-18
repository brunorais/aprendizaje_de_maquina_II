"""
train.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR: Bruno Rais
FECHA: 5 jun 2023
"""

# Imports
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

class ModelTrainingPipeline(object):

    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING 
        
        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """
            
        pandas_df = pd.read_csv(self.input_path)
        
        return pandas_df

    
    def model_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        
        """
        model = LinearRegression()
        
        # División de dataset de entrenaimento y validación
        X = df.drop(columns='Item_Outlet_Sales')
        y = df['Item_Outlet_Sales']
        
        # Entrenamiento del modelo
        model.fit(X,y)
        
        #coef = pd.DataFrame(X.columns, columns=['features'])
        #coef['Coeficiente Estimados'] = model.coef_
        
        
        return model

    def model_dump(self, model_trained) -> None:
        """
        COMPLETAR DOCSTRING
        
        """
        
        joblib.dump(model_trained, self.model_path + '/model0.pkl')

        return None

    def run(self):
    
        df = self.read_data()
        model_trained = self.model_training(df)
        self.model_dump(model_trained)

if __name__ == "__main__":

    ModelTrainingPipeline(input_path = 'Ruta/De/Donde/Voy/A/Leer/Mis/Datos',
                          model_path = 'Ruta/Donde/Voy/A/Escribir/Mi/Modelo').run()