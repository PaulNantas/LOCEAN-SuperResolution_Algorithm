## Codes pour réaliser une optimisation Bayesinne sur l'architecture Resac 

L'optimisation bayesienne est une méthode qui permet d'optimiser une fonction dépendant de multiples paramètres et d'approcher son minimum global en limitant les possibilités de rester coincer dans un minimum local. 

Cette méthode s'appliquent ici aux réseaux de neurones et permet d'approcher le minimum global de la fonction de coût en très peu d'étapes de calculs. Elle repose sur l'approximation de la fonction de coût par une fonction de substitution basée sur différents points échantillonnés. 
A chaque itération, une combinaison d'hyperparamètres est choisie et un nouveau point est associé à la fonction de coût. A partir de cela, des fonctions de substitution émergent à l'aide de processus gaussiens estimant différentes allures de la fonction. 

Dans le cadre de mon travail au sien du laboratoire LOCEAN, j'ai optimisé les différents "étages" de l'algorithme de super-Résolution Resac grâce à la bibliothèque Python BayesianOptimization.
L'algorithme de descente d'échelles se base sur différentes résolutions :

R81 : 16*17 pixels (120 * 122 km²)
R27 : 48*51 pixels (40 * 41 km²)
R09 : 144*153 pixels (13 * 14 km²)
R03 : 432*459 pixels (4.5 * 4.5 km²)
R03 : 1296*1377 pixels (1.5 * 1.5 km²)

J'ai réalisé l'optimisation bayesienne sur chaque étages au fur et à mesure.

## Codes

Les fichiers OBXxYy.py correspondent à l'optimisation de l'étage passant de la résolution RXx à la résolution RYy. 

Le fichier du premier étage R8127_com.py est particulièrement commenté pour comprendre le fonctionnement de la bibliothèque BayesianOptimization.

Le fichier ViewOBResults.py permet d'avoir un résumé des meilleurs résultats qui ont été sauvegardés lors du lancement de codes OBXxYy.py. Il suffit de reférencer la localisation du fichier .json dans la variable "folder_BO".

Les fichiers resacartparm.py et resacartdef.py servent à la mise en forme des données. Il faut adapter la variable "SCENARCHI" dans le fichier resacartparm.py en fonction de l'architecture utilisée et les sorties désirées (notamment les courants U et V).
