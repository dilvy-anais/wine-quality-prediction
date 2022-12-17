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

## Choix Techniques 

### Choix du modèle 

Nous choisissons d'utiliser la technique du machine learning puisqu'il est plus efficace, avec notre jeu de donée. En effet, l'autre technique qui est le deep learning, permet de traiter des données non-structurées : des images, du son, du texte... Ici ce n'est pas notre cas. 

De plus comme tous les varaibles sont bien étiquetté, nous utiliserons les différentes méthodes du "supervised learning".

## Installation

Il faudra ouvrir un terminal et aller dans le dossier "wine-quality-prediction"


