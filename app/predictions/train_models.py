# Les imports
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


data_wine = pd.read_csv('app/data/Wines.csv')

def analyse_model(data: pd.DataFrame)->pd.DataFrame:
    """Cette fonction permet d'analyser mon jeu de donnée. 

    En effet, pour cela nous regardons dans un premier temps les information sdes différentes colonnes, si elle contient des valeurs nulles, ou des valeurs aberrantes.
    On regardera aussi si les variables sont corrélés ou pas. 
    Elle retournera le nouveau jeu de donnée.

    Args:
        data (pd.DataFrame): La dataFrame Wine.csv à étudier.

    Returns:
        pd.DataFrame: Retourne le nouveau dataFrame sans valeurs aberrantes.

    """
    data = data.drop(columns =['Id'])

    data = data.drop(columns =['fixed acidity'])

    indexNames = data[ (data['chlorides'] >= 0.6)].index
    data.drop(indexNames , inplace=True)

    indexNames = data[ (data['total sulfur dioxide'] >= 250)].index
    data.drop(indexNames , inplace=True)

    indexNames = data[ (data['sulphates'] >= 1.75)].index
    data.drop(indexNames , inplace=True)

    # Il faut vérifier la distribution 
    print("\n --- La distribution des classes ")
    distribution = data['quality'].value_counts()
    print(distribution)

    # Significativité des variables
    data = data.drop(columns =['residual sugar'])
    data = data.drop(columns =['citric acid'])
    x,y = data.loc[:,data.columns != 'quality'], data.loc[:,'quality']    

    return data


def train_Random_Forest(data : pd.DataFrame)-> pd.DataFrame:
    """
        Training our AI model.
        Args :
            data: input dataset
        Returns:
            Y_test_predict: prediction result
            Y_Test : prediction
            model : AI model
            acc : mean of accuracy
    """
    acc = 0
    for i in range(30):
        train, test = setting_model()
        x ,y = data.loc[:,data.columns != 'quality'], data.loc[:,'quality']
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size = train, test_size = test)
        model = RandomForestClassifier()
        model.fit(X_train, Y_train)
        Y_test_predict = model.predict(X_test)
        save_model(model)
        acc = acc + (round(accuracy_score(Y_test, Y_test_predict),2))
    acc = acc / 30
    print(round(acc,2))
    return Y_test_predict, Y_test, model, acc


def save_model(model : RandomForestClassifier):
    """
        Save the AI model into file
        Args:
            AI model
        Returns:
            N/A
    """
    joblib.dump(model, "app/data/random_forest.joblib")

 
def load_model()-> RandomForestClassifier:
    """
        Load the AI model from file
        Args:
            N/A
        Returns:
            AI model
    """
    loaded_rf = joblib.load("app/data/random_forest.joblib")
    return loaded_rf


def metrique_model(Y_test : pd.DataFrame, Y_test_predict : pd.DataFrame, acc:int)->float:
    """
        Calcul accuracy rate  of AI
        Args:
             Y_test: ndarray contains score of all wine in training dataset
             Y_test_predict: ndarray contains score of all wine
             acc : mean of accuracy
        Returns:
            Accuracy rate
    """
    print(classification_report(Y_test, Y_test_predict))
    accuracy = acc
    return accuracy


def setting_model()->list:
    """
        Get parameter of AI model
        Args:
            N/A
        Returns:
            List of all model parameters
    """
    train = 0.85
    test = 0.15
    return train, test

def info_wine_to_predict(volatile:float, chlorides:float, free:int, total:int, density:float, ph:float, sulphate:float, alcohol:float)->np.ndarray:
    """
        Predict wine quality score
        Args:
            volatile: volatile acidity of wine
            chlorides: chloride of wine
            free: free sulfur dioxide in wine
            total: total sulfur dioxide in wine
            density: density of wine
            ph: acidity/base score of wine
            sulphate: percent of sulphate in wine
            alcohol: alcohol degree in wine
        Returns:
            Array with only one element contains quality score of wine
    """
    vin = pd.DataFrame([{'volatile acidity': volatile, 'chlorides': chlorides, 'free sulfur dioxide': free, 'total sulfur dioxide': total, 'density' : density, 'pH' : ph, 'sulphates' : sulphate, "alcohol" : alcohol}])
    model = load_model()
    Y_predict = model.predict(vin)
    return Y_predict


def add_wine_dataFrame(fixed:float, volatile:float, citric:float, residual:float, chlorides:float, free:float, total:int, density:int, ph:float, sulphate:float, alcohol:float, quality:int, data:pd.DataFrame)->pd.DataFrame:
    """
        Add wine  to dataframe
        Args:
            fixed acidity
            volatile: volatile acidity of wine
            citric: citric acidity of wine
            residual : residual sugar of wine
            chlorides: chloride of wine
            free: free sulfur dioxide in wine
            total: total sulfur dioxide in wine
            density: density of wine
            ph: acidity/base score of wine
            sulphate: percent of sulphate in wine
            alcohol: alcohol degree in wine
            quality: quality of wine
            data : dataframe
        Returns:
            Return dataframe with another line
    """
    vin = pd.DataFrame([{'fixed acidity' : fixed,'volatile acidity': volatile, 'citric acid' : citric, 'residual sugar' : residual , 'chlorides': chlorides, 'free sulfur dioxide': free, 'total sulfur dioxide': total, 'density' : density, 'pH' : ph, 'sulphates' : sulphate, "alcohol" : alcohol, "quality" : quality}])
    data_complet = pd.concat([vin, data], ignore_index = True)
    return data_complet

def wine_perfect() -> pd.DataFrame:
    """
        perfect wine (quality = 10)
        Args:
            N/A
        Returns:
            Return dataframe with the perfect wine
    """
    vin = pd.DataFrame([{'volatile acidity': 0.4, 'citric acid' : 0.5, 'residual sugar' : 2.66 , 'chlorides': 0.005, 'free sulfur dioxide': 8, 'total sulfur dioxide': 25, 'density' : 0.9945, 'pH' : 0.32, 'sulphates' : 0.85, "alcohol" : 13, "quality" : 10}])
    return vin
