from resacartparm import *
from resacartdef import *

import os
from functools import partial
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from operator import itemgetter

def fit_with(Nconv1, Nfilt1,function, lr):

  model = my_model(Nconv1, Nfilt1)

    # Train the model for a specified number of epochs.
  optimizer = optimizers.Adam(learning_rate=2*(10**lr))
  model.compile(loss='logcosh',
                optimizer=optimizer)
# Train the model with the train dataset.
  H = model.fit(x_train, y_train, epochs=Niter,batch_size=Bsize, 
            shuffle=True, verbose=2, validation_data=(x_valid, y_valid))
  print(H.history['loss'])
  return  -min(H.history['loss'])

fit_with_partial = partial(fit_with)
 
folder_BO = f"Net_kmodel_ACRATOPOTES_E20-BS32_20210729-144306_SIG01/bayesianOpt_InitP-{init_points}_NIter-{n_iter}" 
#Nom du dossier où le .json est sauvegardé

####################Changer le nom du dossier où le fichier .json est sauvegardé : folder_BO ########################
name_log = os.path.join("Save_Model/", folder_BO) #Nom du dossier de Save-Model
#########################################################################################################

#Faire attention à mettre les mêmes paramètres que ceux du model entrainés

#Redéfinition des paramètres de l'optimisation pour pouvoir charger le fichier .json soit pour continuer l'optimisation soit pour lire les résultats
#Même en lecture des résultats il faut redefinir fit_with et new_optimizer
new_optimizer = BayesianOptimization(
    f=fit_with_partial,
    pbounds = {'Nconv1': (1, 5),'Nfilt1': (2,32), 'function':(0,3), 'lr': (-4.5,-1) }#,'function' :(0,3)} #function 0-1: linear, 1-2: sig01, 2-3: sigmoid
    ,verbose=2,
    random_state=7,
)

new = load_logs(new_optimizer, logs=[os.path.join(name_log, "logs.json")]);


#Affichage correct des résultats
print("Affichage des résultats de toutes les itérations")
for i, res in enumerate(new_optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res)+"\n")


print(f"Affichage du meilleur résultat : {new_optimizer.max}"+"\n")


loss= []
hist = []
for i, res in enumerate(new_optimizer.res):
    resultat = res['target']
    loss.append((i,resultat))
    hist.append((i,res))
tri = sorted(loss, key=itemgetter(1), reverse=True)[0:11]

print(f"Affichage des 10 meilleurs résultats triés : {tri}"+"\n")

for k in tri:
  print(f"Meilleures itérations n° {tri.index(k)+1} : {hist[k[0]]}"+"\n")



