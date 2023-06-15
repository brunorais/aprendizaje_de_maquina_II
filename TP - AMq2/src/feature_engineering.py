"""
feature_engineering.py

DESCRIPCIÓN:
AUTOR: Bruno Rais, Joaquín Beitia
FECHA: 5 jun 2023
"""

# Imports
import pandas as pd


class FeatureEngineeringPipeline(object):

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """

        pandas_df = pd.read_csv(self.input_path)
        return pandas_df

    def data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        """

        df['Outlet_Establishment_Year'] = 2020 - \
            df['Outlet_Establishment_Year']
        categoricals = [  # noqa E501
            'Item_Fat_Content',
            'Item_Type',
            'Outlet_Identifier',
            'Outlet_Size',
            'Outlet_Location_Type',
            'Outlet_Type'
        ]
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace(
            {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}
        )
        productos = list(df[df['Item_Weight'].isnull()]['Item_Identifier'].unique())  # noqa E501

        for producto in productos:
            moda = (df[df['Item_Identifier'] == producto][['Item_Weight']]).mode().iloc[0, 0]   # noqa E501
            df.loc[df['Item_Identifier'] == producto, 'Item_Weight'] = moda
        outlets = list(df[df['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())  # noqa E501
        data_aux = df[df['Outlet_Size'].isnull()]  # noqa E501

        for outlet in outlets:
            df.loc[df['Outlet_Identifier'] == outlet, 'Outlet_Size'] = 'Small'

        # FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'  # noqa E501
        df.loc[df['Item_Type'] == 'Household', 'Item_Fat_Content'] = 'NA'
        df.loc[df['Item_Type'] == 'Health and Hygiene', 'Item_Fat_Content'] = 'NA'   # noqa E501
        df.loc[df['Item_Type'] == 'Hard Drinks', 'Item_Fat_Content'] = 'NA'
        df.loc[df['Item_Type'] == 'Soft Drinks', 'Item_Fat_Content'] = 'NA'
        df.loc[df['Item_Type'] == 'Fruits and Vegetables', 'Item_Fat_Content'] = 'NA'  # noqa E501

        # FEATURES ENGINEERING: creando categorías para 'Item_Type'
        df['Item_Type'] = df['Item_Type'].replace({
            'Others': 'Non perishable',
            'Health and Hygiene': 'Non perishable',
            'Household': 'Non perishable',
            'Seafood': 'Meats',
            'Meat': 'Meats',
            'Baking Goods': 'Processed Foods',
            'Frozen Foods': 'Processed Foods',
            'Canned': 'Processed Foods',
            'Snack Foods': 'Processed Foods',
            'Breads': 'Starchy Foods',
            'Breakfast': 'Starchy Foods',
            'Soft Drinks': 'Drinks',
            'Hard Drinks': 'Drinks',
            'Dairy': 'Drinks'
        })

        # FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'  # noqa E501
        df.loc[df['Item_Type'] == 'Non perishable', 'Item_Fat_Content'] = 'NA'
        df['Item_MRP'] = pd.qcut(
            df['Item_MRP'],
            4,
            labels=[1, 2, 3, 4]
        )
        df_transformed = df.drop(columns=[
            'Item_Type', 'Item_Fat_Content']).copy()

        # Codificación de variables ordinales
        df_transformed['Outlet_Size'] = df_transformed['Outlet_Size'].replace(
            {'High': 2, 'Medium': 1, 'Small': 0})
        df_transformed['Outlet_Location_Type'] = df_transformed['Outlet_Location_Type'].replace(  # noqa E501
            {'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0})  # Estas categorias se ordenaron asumiendo la categoria 2 como más lejos  # noqa E501

        df_transformed = pd.get_dummies(
            df_transformed, columns=['Outlet_Type'])

        # Eliminación de variables que no contribuyen a la predicción por ser muy específicas  # noqa E501
        df_transformed = df_transformed.drop(
            columns=['Item_Identifier', 'Outlet_Identifier', 'Set'])

        return df_transformed

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        COMPLETAR DOCSTRING
        """

        transformed_dataframe.to_csv(self.output_path + '/features.csv')

        return None

    def run(self):

        df = self.read_data()
        df_transformed = self.data_transformation(df)
        self.write_prepared_data(df_transformed)


if __name__ == "__main__":

    FeatureEngineeringPipeline(
        input_path='Ruta/De/Donde/Voy/A/Leer/Mis/Datos',
        output_path='Ruta/Donde/Voy/A/Escribir/Mi/Archivo'
    ).run()
