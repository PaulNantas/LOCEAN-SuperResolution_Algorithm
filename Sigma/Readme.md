## Dossier Sigma.

Ce dossier contient une nouvelle approche de l’ajout des prédictions d’incertitude à notre modèle.
Nous avons pensé à réaliser dans un premier temps un pré-entrainement du réseau qui comparera les prédictions aux données d’entrainements et l’incertitude des prédictions à un bruit fictif défini de la manière suivante 
![equation](https://latex.codecogs.com/svg.image?\sigma_{fictif}&space;=&space;\frac{\mu&space;&plus;&space;\epsilon}{C}&space;\&space;\&space;&space;\epsilon&space;&space;\hookrightarrow&space;N(0,0.2))  

Dans ce pré-entrainement, la fonction de coût serait : 

![equation](https://latex.codecogs.com/svg.image?J(\hat{y}_{ij},&space;\hat{\sigma_{ij}}&space;)&space;=&space;\frac{1}{2N}\sum_{i,&space;j}(y_{ij}-\hat{y}_{ij})^2&space;&plus;&space;(\sigma_{ij}-\hat{\sigma}_{fictif,\&space;ij})^2)

À la suite de ce pré-entrainement, nous pourrions réaliser l’entrainement qui ne s’est pas avéré concluant pour voir si le ”réseau guidé” en pré-entrainement permet d’avoir des résultats sur cette incertitude. 
