##Dossier Sigma.

Ce dossier contient une nouvelle approche de l’ajout des prédictions d’incertitude à notre modèle.
Nous avons pensé à réaliser dans un premier temps un pré-entrainement du réseau qui comparera les prédictions aux données d’entrainements et l’incertitude des prédictions à un bruit fictif défini de la manière suivante 

\sigma_{fictif} = frac{\mu + \epsilon}{C} avec \epsilon -> N(0,0.2)

Dans ce pré-entrainement, la fonction de coût serait : 

J(\hat{y}_{ij}, \hat{\sigma_{ij}} ) = \frac{1}{2N}\sum_{i, j}(y_{ij}-\hat{y}_{ij})^2 + (\sigma_{ij}-\hat{\sigma}_{fictif,\ ij})²

À la suite de ce pré-entrainement, nous pourrions réaliser l’entrainement qui ne s’est pas avéré concluant pour voir si le ”réseau guidé” en pré-entrainement permet d’avoir des résultats sur cette incertitude. 
