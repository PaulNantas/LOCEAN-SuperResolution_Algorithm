import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from   time  import time
import datetime
import pandas as pd 
import matplotlib.colors as mcolors

from resacartparm import *
from resacartdef import *

#Chargement du model
model = load_model("Save_Model/FICHIER_R27/Archi")
model.load_weights('Save_Model/FICHIER_R27/Weights/modelkvalid.ckpt') 
parametre = np.load("Save_Model/FICHIER_R27/coparm.npy", allow_pickle=True)

#Chargement des données
datasets_dir = os.getenv('RESAC_DATASETS_DIR', False)
data_dir="NATL60byVar"
SSH_R81=np.load(os.path.join(datasets_dir,data_dir,"NATL60_SSH_R81.npy"))
SST_R27=np.load(os.path.join(datasets_dir,data_dir,"NATL60_SST_R27.npy"))
SSH_R27= np.load(os.path.join(datasets_dir,data_dir,"NATL60_SSH_R27.npy"))

print(parametre)

def normalisation_donnees(jour=0,SSH_R81=SSH_R81,SST_R27=SST_R27, parametre=parametre): 
  #Normalisation des donnees d'entrées adaptée au modele, parametre de normalisation calculé lors de l'apprentissage
  #Jour [0;365] -> 0 : 1er Octobre 2012 au 1er Octobre 2013
  SSH_R81_norm = recodage(SSH_R81[jour,:,:], parametre[0][0])                                                                                 
  SST_R27_norm = recodage(SST_R27[jour,:,:], parametre[0][1])
  return SSH_R81_norm,  SST_R27_norm

def decodage_sortie(predictions,parametre):
  #Dé-normalisation' des prédictions pour les comparer aux données modèles
  SSH_R27_dec = decodage(predictions[0], parametre[1][0])
  return  SSH_R27_dec

def predictions_decodees(model, jour=0): #Jour correspond à la journée que l'on veut prédire/comparer
  SSH_R81_norm, SST_R27_norm = normalisation_donnees(jour=jour) #Entrées normalisées
  R81, R27 = SSH_R81_norm.shape,SST_R27_norm.shape #Récupère les différentes résolutions

#Prédictions de SSH_R27, SSH_R09, U_R09, V_R09
  sample_to_predict = [SSH_R81_norm.reshape((1,R81[0],R81[1],1)), SST_R27_norm.reshape((1,R27[0],R27[1],1))] #Input en "4" dimensions donc reshape
  predictions = model.predict(sample_to_predict)  
  SSH_R27_dec = decodage_sortie(predictions, parametre)

#Création des différences prédictions/modèle
  #difference_R09 = SSH_R09[jour,:,:] - SSH_R09_dec.reshape(R09)
  return SSH_R27_dec.reshape(R27)


#print("SSH R09s", SSH_R09s.shape) (72,90)
#print("SSH R03s", SSH_R03.shape)  (216,270)


from keras.layers import Input, experimental
from keras.models import Model
from keras.optimizers import Adam


#Prédictions par resac et interpolation bicubic

jour = 260#330 est le pire 
prediction = predictions_decodees(model,jour)

resac_flatten   = prediction.reshape(-1)
#Création des fichiers et des plots
nowstr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

Mdl2save = 'Save_Model/TESTR27'
directory = f"Prediction_jour n°{jour}"
dir_name = os.path.join(Mdl2save, directory)


os.makedirs('Save_Model',exist_ok = True)
os.makedirs(Mdl2save,exist_ok = True)
os.makedirs(dir_name,exist_ok = True)


plt.imshow(prediction, cmap='RdBu', origin='lower')
plt.title("Prediction Sat R03")
plt.colorbar()
plt.savefig(os.path.join(dir_name, "Prediction Sat R03"))
plt.clf()

plt.imshow(SSH_R27[jour], cmap='RdBu', origin='lower')
plt.title("Model SSH R03")
plt.colorbar()
plt.savefig(os.path.join(dir_name, "Model SSH R03"))
plt.clf()

plt.imshow(SSH_R81[jour], cmap='RdBu', origin='lower')
plt.title("Sat SSH R09")
plt.colorbar()
plt.savefig(os.path.join(dir_name, "Sat SSH R09"))
plt.clf()

