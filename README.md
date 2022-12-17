# TP Prédiction qualité du vin 
Par Emmy Jaymes & Anaïs Dilvy 

## Description du projet 

    Nous allons prédire la qualité du vin en fonction des différentes paramètres d'entrée que nous pouvons voir ci-dessous.

 --  Description du jeu de donnée : Wines.csv : 

Input variables (based on physicochemical tests):
- fixed acidity | Acidité fixe : il s'agit de l'acidité naturelle du raisin comme l'acide malique ou l'acide tartrique.
- volatile acidity | Acidité volatile : l'acidité volatile d'un vin est constituée par la partie des acides gras comme l'acide acétique appartenant à la série des acides qui se trouvent dans le vin soit à l'état libre, soit à l'état salifié. L'acidité volatile donne au vin du bouquet.
- citric acid | Acide citrique : utilisé pour la prévention de la casse ferrique et participe au rééquilibrage de l'acidité des vins. 
- residual sugar | Sucre résiduel : sucres (glucose + fructose) encore présents dans le vin après fermentation.
- chlorides | Chlorures : matière minérale contenue naturellement dans le vin (sel, magnésium...)
- free sulfur dioxide | Sulfites libres : exhacerbent les propriétés antioxydantes du vin
- total sulfur dioxide | Sulfites libres + Sulfites liées à la réaction avec d'autres molécules du vin
- density | Densité du vin (g/l)
- pH | PH du vin
 - sulphates | Sulfates : sels composés d'anions SO4(2-) != sulfites
 - alcohol | degré d'alcool

Output variable (based on sensory data):
- quality | Qualité générale : note comprise en 0 et 10

## Choix de l'analyse du modèle

Concernant l'analyse du jeu de donnée, nous avons fait plusieurs étapes tels qu': 

- Afficher les informations des différentes colonnes. 

![plot](images/info_colonne.png)

- Supprimer la colonne "Id", qui n'est pas utile pour prédire des données. Elle ne donne pas une informations importantes.

- Vérifier les valeurs nulles dans le jeu de donnée, mais après vérification nous en avons aucune. 

- Regarder la corrélation entre les différentes variables. Nous analyserons toutes les colonnes sauf celle de la qualité du vin. Nous pouvons voir sur la photo ci-dessous, qu'il y a peu de corrélation entre les différentes variables. Mais nous pouvons quand même voir que entre pH et Fixed actidity nous avons -0.69 de corrélation et 0.68 pour Density et Fixed acidity. Nous règlerons ce problème en enlevons le variable Fixed acidity.

![plot](images/matrice_de_correlation.png)

- Regarder les attributs des colonnes. 

![plot](images/attribut_colonne.png)

- Vérifier les valeurs abérrantes de toutes les variables. Voici-ci dessous un exemple pour les chlorides. 

![plot](images/chlorides_valeur_aberrantes.png)

Nous remarquons de valeurs qui ne sont pas énormement aberrantes. Mais si nous regardons de plus près, nous en trouvons quelque uns quand même pour : 
    * les chlorides nous supprimons les valeurs au-dessus 0.6,
    * le total sulfur dioxide nous supprimons les valeurs au-dessus en 250,
    * les sulphates nous supprimons les valeurs au-dessus 1.75.

- Vérifier la distribution des classes. En effet, nous avons trop beaucoup de données pour obtenir une qualité de 6 et 5 sur 10. Mais aucune donnée pour 1, 2, 9 et 10/10. Mais cela peut être dû au fait qu'il est tres compliqué d'avoir des vins parfaits ou médiocres. 

- Regarder la significativité des varaibles. Nous allons utilisé la technique du AIC. Cette étape est longue puisque nous calculons toutes les combinaisons possible pour au final trouver le AIC le plus petit. Nous enlèverons au final citric acid et residual sugar.

### Choix du modèle 

Nous choisissons d'utiliser la technique du machine learning puisqu'il est plus efficace, avec notre jeu de donée. En effet, l'autre technique qui est le deep learning, permet de traiter des données non-structurées : des images, du son, du texte... Ici ce n'est pas notre cas. 

De plus comme tous les varaibles sont bien étiquetté, nous utiliserons les différentes méthodes du "supervised learning". 

Nous avons essayer différentes techniques comme le random Forest (accuracy de 0.65 à 0.73%), ridge (accuracy de 0.59%), régression linéraire (accuracy de 0.58%), neughbors (accuracy de 0.5%)... 

Après étude de l'accuracy des différents modèles nous décidons de choisir l'entrainement de modèle fait avec Radom Forest. 

Voici ci-dessous un image des différents accuracy que nous avons obtenu avec Ramdom Forest : 
![plot](images/accuracy_model.png)

Dans cette étapes, nous avons régarder l'importance de toutes les variables. Nous remarquons que nous n'avons pas de variable qui se démarque et qui n'est pas utile. 

![plot](images/feature_importants.png)

## Choix des paramètres du model 

Nous avons essayer de changer les valeurs tels que n_estimator, ramdom_state, max_depth... Mais cela baissé ou ne changé pas l'accuracy, nous avons donc laissé, après plisuers tests les paramètres par défault. 

Nous avons juste décider de séparer le jeu d'entrainement et de test à 85% pour l'entrainement et 15% pour le test.

## Installation

Il faudra ouvrir un terminal et aller dans le dossier "wine-quality-prediction"


