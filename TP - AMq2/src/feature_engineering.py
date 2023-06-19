"""
feature_engineering.py

DESCRIPCIÓN:
AUTOR: Bruno Rais, Joaquín Beitia
FECHA: 5 jun 2023
"""

# Imports
import pandas as pd
import logging


class FeatureEngineeringPipeline(object):

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """
        Reads the data from the specified input path and returns it
        as a DataFrame.

        :return: The desired DataLake table as a DataFrame.
        :rtype: pd.DataFrame
        """

        logging.info("Reading data from input path: %s", self.input_path)
        pandas_df = pd.read_csv(self.input_path)
        logging.info("Data successfully read into a DataFrame.")

        return pandas_df

    def data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies data transformations to the input DataFrame.

        Args:
            df: Input DataFrame containing the data to be transformed.

        Returns:
            Transformed DataFrame after applying the data transformations.

        """

        df['Outlet_Establishment_Year'] = 2020 - \
            df['Outlet_Establishment_Year']
        # categoricals = [
        #     'Item_Fat_Content',
        #     'Item_Type',
        #     'Outlet_Identifier',
        #     'Outlet_Size',
        #     'Outlet_Location_Type',
        #     'Outlet_Type'
        # ]

        logging.info("CLEANING: Unifying labels for 'Item_Fat_Content'")
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace(
            {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}
        )

        logging.info("CLEANING: Handling missing values in product weights")
        productos = list(df[df['Item_Weight'].isnull()]['Item_Identifier'].unique())  # noqa E501
        for producto in productos:
            moda = (df[df['Item_Identifier'] == producto][['Item_Weight']]).mode().iloc[0, 0]   # noqa E501
            df.loc[df['Item_Identifier'] == producto, 'Item_Weight'] = moda

        logging.info("PROCESSING: Checking recorded values for store sizes")
        outlets = list(df[df['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())  # noqa E501
        # data_aux = df[df['Outlet_Size'].isnull()]

        logging.info("CLEANING: Handling missing values in store sizes")
        for outlet in outlets:
            df.loc[df['Outlet_Identifier'] == outlet, 'Outlet_Size'] = 'Small'

        logging.info("FEATURES ENGINEERING: Assigning new categories for 'Item_Fat_Content'")  # noqa E501
        df.loc[df['Item_Type'] == 'Household', 'Item_Fat_Content'] = 'NA'
        df.loc[df['Item_Type'] == 'Health and Hygiene', 'Item_Fat_Content'] = 'NA'  # noqa E501
        df.loc[df['Item_Type'] == 'Hard Drinks', 'Item_Fat_Content'] = 'NA'
        df.loc[df['Item_Type'] == 'Soft Drinks', 'Item_Fat_Content'] = 'NA'
        df.loc[df['Item_Type'] == 'Fruits and Vegetables', 'Item_Fat_Content'] = 'NA'  # noqa E501

        logging.info("FEATURES ENGINEERING: Creating categories for 'Item_Type'")  # noqa E501
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

        logging.info("FEATURES ENGINEERING: Assigning new categories for 'Item_Fat_Content'")  # noqa E501
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
        Writes the prepared data to a CSV file.

        Args:
            transformed_dataframe: DataFrame containing the prepared data.

        Returns:
            None.

        """

        transformed_dataframe.to_csv(self.output_path + '/features.csv')

        return None

    def run(self):

        df = self.read_data()
        df_transformed = self.data_transformation(df)
        self.write_prepared_data(df_transformed)


if __name__ == "__main__":

    FeatureEngineeringPipeline(
        input_path='./data/Train_BigMart.csv',
        output_path='./src'
    ).run()
