# Les imports
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
class PredictionModel:
    def __init__(self):
        self.data = pd.read_csv('app/data/Wines.csv')
        self.data = self.data.drop(columns=["Id"])
        self.model_backup_path="app/data/random_forest.joblib"
        self.accuracy = 0.0
        self.test_sample=0.15
        self.train_sample=0.85
        self.model = None

    def analyse_model(self) -> pd.DataFrame:
        """Clean the dataframe.

        Args:
            data (pd.DataFrame): Wine.csv dataFrame.

        Returns:
            pd.DataFrame: same dataframe without outlier value.

        """
        clean_data = self.data.drop(columns=['fixed acidity'])

        indexNames = clean_data[(clean_data['chlorides'] >= 0.6)].index
        clean_data.drop(indexNames, inplace=True)

        indexNames = clean_data[(clean_data['total sulfur dioxide'] >= 250)].index
        clean_data.drop(indexNames, inplace=True)

        indexNames = clean_data[(clean_data['sulphates'] >= 1.75)].index
        clean_data.drop(indexNames, inplace=True)

        # Il faut vérifier la distribution
        print("\n --- La distribution des classes ")
        distribution = clean_data['quality'].value_counts()
        print(distribution)

        # Significativité des variables
        clean_data = clean_data.drop(columns=['residual sugar'])
        clean_data = clean_data.drop(columns=['citric acid'])
        x, y = clean_data.loc[:, clean_data.columns != 'quality'], clean_data.loc[:, 'quality']

        return clean_data

    def train_Random_Forest(self,data: pd.DataFrame) ->():
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
            train, test = self.train_sample, self.test_sample
            x, y = data.loc[:, data.columns != 'quality'], data.loc[:, 'quality']
            X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=train, test_size=test)
            self.model = RandomForestClassifier()
            self.model.fit(X_train, Y_train)
            Y_test_predict = self.model.predict(X_test)
            self.save_model()
            acc = acc + (round(accuracy_score(Y_test, Y_test_predict), 2))
        acc = acc / 30
        self.accuracy= round(acc, 2)

    def save_model(self)->():
        """
            Save the AI model into file
            Args:
                AI model
            Returns:
                N/A
        """
        joblib.dump(self.model, "app/data/random_forest.joblib")

    def load_model(self) -> ():
        """
            Load the AI model from file
            Args:
                N/A
            Returns:
                AI model
        """
        self.model = joblib.load(self.model_backup_path)


    def info_wine_to_predict(self, volatile: float, chlorides: float, free: int, total: int, density: float, ph: float,
                             sulphate: float, alcohol: float) -> int:
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
        vin = pd.DataFrame([{'volatile acidity': volatile, 'chlorides': chlorides, 'free sulfur dioxide': free,
                             'total sulfur dioxide': total, 'density': density, 'pH': ph, 'sulphates': sulphate,
                             "alcohol": alcohol}])
        self.load_model()
        Y_predict = self.model.predict(vin)
        return int(Y_predict[0])


    def add_wine_dataFrame(self, fixed: float, volatile: float, citric: float, residual: float, chlorides: float, free: float,
                           total: int, density: int, ph: float, sulphate: float, alcohol: float, quality: int) ->():
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
        vin = pd.DataFrame([{'fixed acidity': fixed, 'volatile acidity': volatile, 'citric acid': citric,
                             'residual sugar': residual, 'chlorides': chlorides, 'free sulfur dioxide': free,
                             'total sulfur dioxide': total, 'density': density, 'pH': ph, 'sulphates': sulphate,
                             "alcohol": alcohol, "quality": quality}])
        self.data = pd.concat([vin, self.data], ignore_index=True)
    @staticmethod
    def wine_perfect() -> dict:
        """
            perfect wine (quality = 10)
            Args:
                N/A
            Returns:
                Return dataframe with the perfect wine
        """
        vin = {'fixed acidity': 9.4, 'volatile acidity': 0.4, 'citric acid': 0.5, 'residual sugar': 2.66,
               'chlorides': 0.005, 'free sulfur dioxide': 8, 'total sulfur dioxide': 25, 'density': 0.9945, 'pH': 3.32,
               'sulphates': 0.85, "alcohol": 13, "quality": 10}
        return vin

model = PredictionModel()
