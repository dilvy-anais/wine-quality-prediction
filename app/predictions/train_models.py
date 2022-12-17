# Les imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


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
    print("\n --- L'information des colonnes :")
    data.info()

    data = data.drop(columns =['Id'])

    print(" \n --- Les valeurs nulles")
    data.isnull().sum()
    # Conclusion : pas de valeur null avec 1143 données

    print("\n --- La matrice de corrélation")
    corr_df = data.corr(method='pearson')
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True)
    #plt.show()
    # Conclusion : Il n'y a pas trop de corrélation à part entre (ph et fixed acidity) -0.69 et (density et fixed acidity) 0.68
    # Si le modèle n'est pas bien on enlèvement la variable fixed acidity

    data = data.drop(columns =['fixed acidity'])

    data.info()

    #print("\n --- Les attributs de chaque colonne : \n",attribue) 
    #attribue = data.describe().T    

    #print("Les valeurs aberrantes :")
    #plt.hist(data['fixed acidity'])
    #plt.show()
    #plt.title("fixed acidity")
        # Conculsion : 
    # en volatile acidity => sup apres 1.3 mais pas obligé
    # citric acid => sup apres 0.9 mais pas obligé
    # residual sugar => sup après 12 mais aps oblige
    # chlorides => sup apres 0.3 mais pas obligé sinon sur enlever apres 0.6  -- OBLIGE
    # free sulfur dioxide => sup apres 60 mais pas oblige
    # total sulfur dioxide=> sup au dessus en 250 -- OBLIGE
    # sulphates => sup après 1.75 -- OBLIGE

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
    print("\n --- Le AIC ")
    data = data.drop(columns =['residual sugar'])
    data = data.drop(columns =['citric acid'])
    x,y = data.loc[:,data.columns != 'quality'], data.loc[:,'quality']
    model = sm.OLS(y, x).fit()
    #view AIC of model
    print(model.aic)    
    # 2207 => sans le citric acid et la residual sugar

    return data

def entrainement_models_regression_lineaire(data):
    x,y = data.loc[:,data.columns != 'quality'], data.loc[:,'quality']
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state=5)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    lmodellineaire = LinearRegression()
    lmodellineaire.fit(X_train, Y_train)

    # model evaluation for testing set
    y_test_predict = lmodellineaire.predict(X_test)
    for i in range(len(y_test_predict)):
        y_test_predict[i] = round(y_test_predict[i])
    r2 = round(r2_score(Y_test, y_test_predict),2)

    print('le score R2 est {}'.format(r2))
    print('Accuracy :', round(accuracy_score(Y_test, y_test_predict),2))
    print('Regression MSE ', round(mean_squared_error(Y_test,y_test_predict),2))

# Conclusion : 0.58

def entrainement_models_neighbors_complexi(data):
    knn = KNeighborsClassifier(n_neighbors = 27)
    x,y = data.loc[:,data.columns != 'quality'], data.loc[:,'quality']
    knn.fit(x,y)
    prediction = knn.predict(x)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
    knn = KNeighborsClassifier(n_neighbors = 3)
    x,y = data.loc[:,data.columns != 'quality'], data.loc[:,'quality']
    knn.fit(x_train,y_train)
    y_test_predict = knn.predict(x_test)
    # Model complexity
    neig = np.arange(1, 25)
    train_accuracy = []
    test_accuracy = []
    # Loop over different values of k
    for i, k in enumerate(neig):
        # k from 1 to 25(exclude)
        knn = KNeighborsClassifier(n_neighbors=k)
        # Fit with knn
        knn.fit(x_train,y_train)
        #train accuracy
        train_accuracy.append(knn.score(x_train, y_train))
        # test accuracy
        test_accuracy.append(knn.score(x_test, y_test))

    # Plot
    plt.figure(figsize=[13,8])
    plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
    plt.plot(neig, train_accuracy, label = 'Training Accuracy')
    plt.legend()
    plt.title('-value VS Accuracy')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.xticks(neig)
    plt.savefig('graph.png')
    #plt.show()
    print('Accuracy :', round(accuracy_score(y_test, y_test_predict),2))
    #print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))

# Conclusion : 0.5

def entrainement_model_Ridge(data):
    x,y = data.loc[:,data.columns != 'quality'], data.loc[:,'quality']
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state=5)
    # ridge = RidgeCV(alphas = 0.3, store_cv_values=True)
    ridge = Ridge(alpha = 0.3)
    # print(ridge.alpha_)
    # Affiche 0.3
    ridge.fit(X_train,Y_train)
    Y_test_predict = ridge.predict(X_test)
    for i in range(len(Y_test_predict)):
        Y_test_predict[i] = round(Y_test_predict[i])
    print('Ridge Score: ', round(r2_score(Y_test,Y_test_predict),2))
    print('Ridge MSE ', round(mean_squared_error(Y_test,Y_test_predict),2))
    print('Accuracy :', round(accuracy_score(Y_test, Y_test_predict),2))

# Conclusion : 0.59

# Le meilleur model qu'on a pour le moment
def entrainement_Random_Forest(data):
    """
        Training our AI model.
        Args :
            data: input dataset
        Returns:
            Y_test_predict: prediction result
            Y_Test : prediction
            clf : AI model
    """
    n_estimators, random_state, test_size, random_state_train = parametre_model()
    x,y = data.loc[:,data.columns != 'quality'], data.loc[:,'quality']
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = test_size, random_state=random_state_train)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, Y_train)
    Y_test_predict = clf.predict(X_test)
    return Y_test_predict, Y_test, clf

# Conclusion : 0.65

# Sauvegarde le model
def enregistrer_model(model):
    """
        Save the AI model into file
        Args:
            AI model
        Returns:
            N/A
    """
    joblib.dump(model, "app/data/random_forest.joblib")

# Charge le model 
def charger_model():
    """
        Load the AI model from file
        Args:
            N/A
        Returns:
            AI model
    """
    loaded_rf = joblib.load("app/data/random_forest.joblib")
    return loaded_rf

#Affiche les métriques de performance du model
def metrique_model(Y_test, Y_test_predict)->float:
    """
        Calcul accuracy rate  of AI
        Args:
             Y_test: ndarray contains score of all wine in training dataset
             Y_test_predict: ndarray contains score of all wine
        Returns:
            Accuracy rate
    """
    print(classification_report(Y_test, Y_test_predict))
    accuracy = round(accuracy_score(Y_test, Y_test_predict),2)
    return accuracy

# return les paramètre utiliser dans le model
def parametre_model()->list:
    """
        Get parameter of AI model
        Args:
            N/A
        Returns:
            List of all model parameters
    """
    n_estimators=900
    random_state=42
    test_size = 0.5
    random_state_train=5
    return n_estimators, random_state, test_size, random_state_train

def info_vin_a_predire(volatile:float, chlorides:float, free:int, total:int, density:float, ph:float, sulphate:float, alcohol:float)->np.ndarray:
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
    model = charger_model()
    Y_predict = model.predict(vin)
    return Y_predict

# le data du vin celui de base pas celui apres l'analyse
def ajout_vin_dataFrame(volatile:float, citric:float, residual:float, chlorides:float, free:float, total:int, density:int, ph:float, sulphate:float, alcohol:float, data:pd.DataFrame)->pd.DataFrame:
    """
        Add wine  to dataframe
        Args:
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
            data : dataframe
        Returns:
            Return dataframe with another line
    """
    vin = pd.DataFrame([{'volatile acidity': volatile, 'citric acid' : citric, 'residual sugar' : residual , 'chlorides': chlorides, 'free sulfur dioxide': free, 'total sulfur dioxide': total, 'density' : density, 'pH' : ph, 'sulphates' : sulphate, "alcohol" : alcohol}])
    data_complet = pd.concat([vin, data], ignore_index = True)
    return data_complet

def vin_parfait():
    # analyser kes graphiques
    return 0








