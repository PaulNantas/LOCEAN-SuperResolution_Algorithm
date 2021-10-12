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
model = load_model("Save_Model/Net_kmodel_ACRATOPOTES_E10000-BS32_20210717-165901_SIG01/Archi")
model.load_weights('Save_Model/Net_kmodel_ACRATOPOTES_E10000-BS32_20210717-165901_SIG01/Weights/modelkvalid.ckpt') 
parametre = np.load("Save_Model/Net_kmodel_ACRATOPOTES_E10000-BS32_20210717-165901_SIG01/coparm.npy", allow_pickle=True)

#Chargement des données
datasets_dir = os.getenv('RESAC_DATASETS_DIR', False)
data_dir="Satellite/SatbyVar"
SSH_R09s=np.load(os.path.join(datasets_dir,data_dir,"SAT_SSH_R09s.npy"))
SST_R03s=np.load(os.path.join(datasets_dir,data_dir,"SAT_SST_R03s.npy"))
SSH_R03= np.load(os.path.join(datasets_dir,'NATL60byVarRXXs',"NATL60_SSH_R03s.npy"))

print(parametre)

def normalisation_donnees(jour=0,SSH_R09s=SSH_R09s,SST_R03s=SST_R03s, parametre=parametre): 
  #Normalisation des donnees d'entrées adaptée au modele, parametre de normalisation calculé lors de l'apprentissage
  #Jour [0;365] -> 0 : 1er Octobre 2012 au 1er Octobre 2013
  SSH_R09_norm = recodage(SSH_R09s[jour,:,:], parametre[0][0])                                                                                 
  SST_R03_norm = recodage(SST_R03s[jour,:,:], parametre[0][1])
  return SSH_R09_norm,  SST_R03_norm

def decodage_sortie(predictions,parametre):
  #Dé-normalisation' des prédictions pour les comparer aux données modèles
  SSH_R03_dec = decodage(predictions[0], parametre[1][0])
  return  SSH_R03_dec

def predictions_decodees(model, jour=0): #Jour correspond à la journée que l'on veut prédire/comparer
  SSH_R09_norm, SST_R03_norm = normalisation_donnees(jour=jour) #Entrées normalisées
  R09, R03 = SSH_R09_norm.shape,SST_R03_norm.shape #Récupère les différentes résolutions

#Prédictions de SSH_R27, SSH_R09, U_R09, V_R09
  sample_to_predict = [SSH_R09_norm.reshape((1,R09[0],R09[1],1)), SST_R03_norm.reshape((1,R03[0],R03[1],1))] #Input en "4" dimensions donc reshape
  predictions = model.predict(sample_to_predict)  
  SSH_R03_dec = decodage_sortie(predictions, parametre)

#Création des différences prédictions/modèle
  #difference_R09 = SSH_R09[jour,:,:] - SSH_R09_dec.reshape(R09)
  return SSH_R03_dec.reshape(R03)


#print("SSH R09s", SSH_R09s.shape) (72,90)
#print("SSH R03s", SSH_R03.shape)  (216,270)


from keras.layers import Input, experimental
from keras.models import Model
from keras.optimizers import Adam

def bicubicR03(height, width,input):
  ArchiOut = []
  test = Input(shape=(height,width,1))
  ArchiA = keras.layers.experimental.preprocessing.Resizing(
  3*height, 3*width, interpolation="bicubic")(test)
  ArchiOut.append(ArchiA)

  model   = Model(test, ArchiOut)
  model.summary(); 
  model.compile(loss='logcosh', optimizer='adam')
  #return(model.predict(input.reshape(1,height,width,1)))
  return(model.predict(input).reshape(input.shape[0],3*height,3*width))

def bicubicR01(height, width,input):
  ArchiOut = []
  test = Input(shape=(height,width,1))
  ArchiA = keras.layers.experimental.preprocessing.Resizing(
  3*height, 3*width, interpolation="bicubic")(test)

  ArchiA = keras.layers.experimental.preprocessing.Resizing(
  9*height, 9*width, interpolation="bicubic")(ArchiA)
  ArchiOut.append(ArchiA)

  model   = Model(test, ArchiOut)
  model.summary(); 
  model.compile(loss='logcosh', optimizer='adam')
  #return(model.predict(input.reshape(1,height,width,1)))
  return(model.predict(input).reshape(input.shape[0],9*height,9*width))

#Prédictions par resac et interpolation bicubic

jour = 260#330 est le pire 
prediction = predictions_decodees(model,jour)
bicubic03s = bicubicR03(72,90, SSH_R09s)
bicubic01s = bicubicR01(72,90, SSH_R09s)

bicubic_flatten = bicubic03s[jour].reshape(-1)
resac_flatten   = prediction.reshape(-1)
#Création des fichiers et des plots
nowstr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

Mdl2save = 'Save_Model/TestSat'
directory = f"Prediction_jour n°{jour}"
dir_name = os.path.join(Mdl2save, directory)


os.makedirs('Save_Model',exist_ok = True)
os.makedirs(Mdl2save,exist_ok = True)
os.makedirs(dir_name,exist_ok = True)

plt.scatter(bicubic_flatten,resac_flatten, color='red',s=1, label='Points')

x = np.arange(np.min([np.min(bicubic_flatten),np.min(resac_flatten)]),np.max([np.max(bicubic_flatten),np.max(resac_flatten)]),0.05)
plt.plot(x,x,color='blue',label='Identité')
plt.title("ScatterPlot Sat R03 bicubic/Resac")
plt.xlabel("Bicubic")
plt.ylabel("Resac")
plt.savefig(os.path.join(dir_name, "ScatterPlot Sat R03 bicubic-Resac"))
plt.show()
plt.clf()

plt.imshow(prediction, cmap='RdBu', origin='lower')
plt.title("Prediction Sat R03")
plt.colorbar()
plt.savefig(os.path.join(dir_name, "Prediction Sat R03"))
plt.clf()

plt.imshow(SSH_R03[jour], cmap='RdBu', origin='lower')
plt.title("Model SSH R03")
plt.colorbar()
plt.savefig(os.path.join(dir_name, "Model SSH R03"))
plt.clf()

plt.imshow(SSH_R09s[jour], cmap='RdBu', origin='lower')
plt.title("Sat SSH R09")
plt.colorbar()
plt.savefig(os.path.join(dir_name, "Sat SSH R09"))
plt.clf()

plt.imshow(bicubic03s[jour], cmap='RdBu', origin='lower')
plt.title("Sat SSH R03 bicubic")
plt.colorbar()
plt.savefig(os.path.join(dir_name, "Sat SSH R03 bicubic"))
plt.clf()

plt.imshow(bicubic01s[jour], cmap='RdBu', origin='lower')
plt.title("Sat SSH R01 bicubic")
plt.colorbar()
plt.savefig(os.path.join(dir_name, "Sat SSH R01 bicubic"))
plt.clf()

