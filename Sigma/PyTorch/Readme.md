##Ajout d’une incertitude dans les prédictions de notre model

Un autre objectif que nous nous étions fixé durant ces travaux est d’estimer, en plus de la prédiction de la valeur du pixel d’une image, l’incertitude avec laquelle est faite cette prédiction.

L’idée derrière cette méthode est d’ajouter une image en plus en sortie (on se retrouve avec une image de prédiction de la SSH et en même temps ”une image des incertitudes”) et aussi de modifier la fonction de coût en conséquence pour prendre en compte cette incertitude. 

On aurait la fonction de coût suivante : 

J(\hat{y}_{ij}, \hat{\sigma_{ij}} ) = \frac{1}{2N}\sum_{i, j}(\frac{y_{ij}-\hat{y}_{ij}}{\hat{\sigma_{ij}}^2})^2 + 2log(\hat{\sigma_{ij}}) + log(2\pi)

##Codes

J’ai réalisé une conversion de l’architecture Resac de Tensorflow vers PyTorch pour pouvoir avoir plus facilement le contrôle du réseau et de ses différents paramètres.

J’ai d’abord réalisé une architecture en PyTorch de Resac classique pour m’acclimater à PyRotch qui est une nouvelle bibliothèque pour moi. Ce sont les fichiers PTRxX.py.

Ensuite, j’ai réalisé l’architecture avec incertitude précédemment évoquée. À cause de certains soucis de version de torch, il y a un code compatible avec torch1.7 et un autre avec torch1.9. Le code archi_pytorch19.py est le plus abouti des deux.

La méthode d’ajout d’incertitude n’a pas donné de résultats concluant pour le moment.
Nous avons donc pensé à une nouvelle architecture qui est implémenté dans le dossier Sigma/.
