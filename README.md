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
+ TRIED sompy
+ TRIED ctk
+ TRIED tools

# Méthodes:
+ Perceptron Multi-Couches (PMC)
+ PMC + Cartes Auto-organisatrices (SOM)
+ PMC + Decodeur
+ Réseau Récurrent Bidirectionnel (BiRNN)
+ BiRNN + Cartes Auto-organisatrices (SOM)
+ BiRNN + Decodeur

# Analyse des Performances:
| Méthodes | RMS (2006-2008) | Min (2006-2008) | Max (2006-2008) | RMS (2008) | Min (2008) | Max (2008) |
| --- | --- | --- | --- | --- | --- | --- |
| PROFHMM | | | | 0.0303 | 0.0076 | 0.0310 |
| PMC | 0.0644 | 0.0078 | 0.1459 | 0.0511 | 0.0087 | 0.1049 |
| PMC+SOM | 0.0659 | 0.0056 | 0.1403 | 0.0587 | 0.0065 | 0.1243 | 
| PMC+SOM | 0.0659 | 0.0056 | 0.1403 | 0.0587 | 0.0065 | 0.1243 | 
| RNN | 0.0618 | 0.0046 | 0.1480 | 0.0547 | 0.0052 | 0.1226 | 
| RNN + SOM | 0.0677 | 0.0060 | 0.1498 | 0.0630 | 0.0065 | 0.1359 |
| RNN + Décodeur | 0.0618 | 0.0036 | 0.1406 | 0.0552 | 0.0034 | 0.1263 | 





