# Comparaison de méthodes de machine learning pour la reconstitution de profils verticaux de chlorophylle à partir de données océaniques de surface
Maria CANE – Lydia CHIBANE

Projet Long fait au sein du laboratoire LOCEAN encadré par Mr AA. Charantonis. 

Objectif: Retrouver les profils verticaux de chlorophylle-a à partir des mesures satellitaires sur la surface de l'océan.

# Python Packages:
+ Numpy
+ Matplotlib
+ Pandas
+ Sklearn
+ Keras

# Méthodes:
+ Perceptron Multi-Couches (PMC)
+ PMC + Cartes Auto-organisatrices (SOM)
+ PMC + Decodeur
+ Réseau Récurrent Bidirectionnel (BiRNN)
+ BiRNN + Cartes Auto-organisatrices (SOM)
+ BiRNN + Decodeur

# Analyse des Performances:
| Méthodes | RMS | Min | Max | RMS | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| Référence Thèse | --- | --- | --- |0.0303|0.0076|0.0310|
