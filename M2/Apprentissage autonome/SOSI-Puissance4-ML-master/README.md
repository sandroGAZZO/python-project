# SOSI-Puissance4-ML

**Ce projet ayant été réalisé en 5 jours en utilisant des technologies nouvelles pour nous (Python et machine learning), le code n'est pas propre et pas parfaitement fonctionnel**

## Introduction
Puissance 4 réalisé en une semaine dans le cadre d'un projet SOSI à l'INSA de Rouen en ASI 4. L'IA de ce jeu sera basé sur **un réseau de neurones artificiels**.

## Présentation
Nous avons utilisé Keras pour créer le réseau de neurones et nous nous sommes servis d'un [dataset de l'UCI](https://archive.ics.uci.edu/ml/datasets/Connect-4) pour l'apprentissage.

A partir de ce dataset, un autre dataset a été créée indiquant le meilleur coup à jouer dans chaque situation du premier dataset. Le meilleur coup a été évalué par l'algorithme Minimax (adapté de celui de [ce projet](https://github.com/erikackermann/Connect-Four) car ce n'était pas le sujet de notre projet).

Le dataset ne contenant que des parties à l'état de 8 pions sur le terrain. L'IA "réseau de neurones" implantée dans le jeu est assez mauvaise en vraie partie.


## Auteurs
* RIDEL Morgan
* LEROGERON Hugo
* DE FILIPPIS Michael
* EL YAAGOUBI Anass
