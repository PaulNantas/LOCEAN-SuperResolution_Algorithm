#-*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import os
from   matplotlib  import cm
import sys
from   time  import time
import datetime
import matplotlib.pyplot as plt

from resacartparm import *
from resacartdef import *
import time
t0 = time.time()

Niter, Bsize = 10000, 32
lr = 0.0005#2*(10**(-2.89637961))

def load_resac_by_var_and_resol(varIn,varOut,ResoIn,ResoOut, subdir='NATL60byVar', noise=RESAC_WITH_NOISE):
    """
    Exemple d'usage:
        V_data_list, couple_var_reso_list = load_resac_by_var_and_resol(varIn,varOut,ResoIn,ResoOut)

    Lecture des données RESAC par variable/résolution.  La function s'attend à trouver
    le répertoire des données dans la variable d'environnement RESAC_DATASETS_DIR.

    Les données par Variable/résolution se trouvent normalement dans un sous-dossier
    appelé 'NATL60byVar'. Il est posisble de spécifier un autre nom avec l'option 
    subdir='DOSSIER'.
    
    Retourne deux éléments:
        
        - une liste d'array 3D ([np.time steps, y size, x size]) des données contenant
          les arrays individuels par variable et résolution nécessaires au cas (selon
          les variables d'entrée varIn, varOut, ResoIn, ResoOut)
          
        - une liste de couples (Variable, Résolution) intervenant dans le cas.
    """
    # ---- datasets location
    #
    datasets_dir = os.getenv('RESAC_DATASETS_DIR', False)
    if datasets_dir is False:
        print('\n# ATTENTION !!\n#')
        print('#  Le dossier contenant les datasets est introuvable !\n#')
        print('#  Pour que les notebooks et programmes puissent le localiser, vous')
        print("#  devez préciser la localisation de ce dossier datasets via la")
        print("#  variable d'environnement : RESAC_DATASETS_DIR.\n#")
        print('#  Exemple :')
        print("#    Dans votre fichier .bashrc :")
        print('#    export RESAC_DATASETS_DIR=~/mon/datasets/folder\n')
        print('#')
        print('# Valeurs typiques :')
        print('# - Au Locean, toute machine (acces lent):')
        print('#     export RESAC_DATASETS_DIR="/net/pallas/usr/neuro/com/carlos/Clouds/SUnextCloud/Labo/Stages-et-Projets-long/Resac/donnees"')
        print('#')
        print('# - Au Locean, uniquement Acratopotes (accès rapide, disque SSD):')
        print('#     export RESAC_DATASETS_DIR="/datatmp/home/carlos/Projets/Resac/donnees"')
        print('#')
        print('# - Dans le cluster GPU Hal (accès rapide, disque SSD):')
        print('#     export RESAC_DATASETS_DIR="/net/nfs/ssd3/cmejia/Travaux/Resac/donnees"')
        assert False, 'datasets folder not found, please set RESAC_DATASETS_DIR env var.'
    # Resolve tilde...
    datasets_dir=os.path.expanduser(datasets_dir)
    #
    couple_var_reso_list = []
    for v,r in zip(varIn+varOut,ResoIn+ResoOut):
        if not (v,r) in couple_var_reso_list :
            couple_var_reso_list.append((v,r))

    dir_labels='NATL60byVarRXXs'
    # Lecture Des Donnees    
    V_data_list = []; D_dico_list = []
    print('Les données en entrées sont bruitées :', RESAC_WITH_NOISE,"\n")
    if noise: 
        index = 0
        for v,r in couple_var_reso_list:
            if index >= len(varIn):
                print("Chargement des données NATL60 dimensions satellites pour la sortie")
                print(f"loading data: '{v}' at R{r}","\n")
                data_tmp = np.load(os.path.join(datasets_dir,dir_labels,f"NATL60_{v.upper()}_R{r:02d}s.npy")) #s à la fin pour les résolutions satellites
                dimension_tmp = np.load(os.path.join(datasets_dir,dir_labels,f"NATL60_coords_R{r:02d}s.npz"))
                
            else:
                print("Chargement données satellites pour l'entrée bruitée")
                print(f"loading data: '{v}' at R{r}","\n")
                data_tmp = np.load(os.path.join(datasets_dir,subdir,f"SAT_{v.upper()}_R{r:02d}s.npy"))
                dimension_tmp = np.load(os.path.join(datasets_dir,subdir,f"SAT_coords_R{r:02d}s.npz"))
        
            dico_dim = { 'time': dimension_tmp['time'],
                            'lat' : dimension_tmp['latitude'],
                            'lon' : dimension_tmp['longitude'],
                            'lat_border' : dimension_tmp['latitude_border'],
                            'lon_border' : dimension_tmp['longitude_border'] }
            index +=1
            V_data_list.append(data_tmp)
            D_dico_list.append(dico_dim)
    else: 
        for v,r in couple_var_reso_list:
            print(f"loading data: '{v}' at R{r}")
            data_tmp = np.load(os.path.join(datasets_dir,subdir,f"NATL60_{v.upper()}_R{r:02d}.npy")) #s à la fin pour les résolutions satellites
            dimension_tmp = np.load(os.path.join(datasets_dir,subdir,f"NATL60_coords_R{r:02d}.npz"))
            dico_dim = { 'time': dimension_tmp['time'],
                            'lat' : dimension_tmp['latitude'],
                            'lon' : dimension_tmp['longitude'],
                            'lat_border' : dimension_tmp['latitude_border'],
                            'lon_border' : dimension_tmp['longitude_border'] }
            V_data_list.append(data_tmp)
            D_dico_list.append(dico_dim)
            
    return V_data_list, couple_var_reso_list, D_dico_list



if LOAD_DATA_BY_VAR_AND_RESOL :
    if RESAC_WITH_NOISE:
        V_data_list, couple_var_reso_list, D_dico_list = load_resac_by_var_and_resol(varIn,varOut,ResoIn,ResoOut, subdir='Satellite/SatbyVar')
    else:
        V_data_list, couple_var_reso_list, D_dico_list = load_resac_by_var_and_resol(varIn,varOut,ResoIn,ResoOut)

    time_axis = D_dico_list[0]['time']
    Nimg_ = V_data_list[0].shape[0]       # nombre de patterns ou images (ou jours)
    #
    if FLAG_STAT_BASE_BRUTE : # Statistiques de base sur les variables brutes lues (->PPT)
        print('Code for FLAG_STAT_BASE_BRUTE not implemented yet in LOAD_DATA_BY_VAR_AND_RESOL mode!')
    # Splitset Ens App - Val - Test
    print(f"Splitset Ens App - Val - Test {pcentSet}% or", end='')
    indA, indV, indT = isetalea(Nimg_, pcentSet);
    print(f" {(len(indA), len(indV), len(indT))} images par ensemble.")
    if FLAG_STAT_BASE_BRUTE_SET :
        print('Code for FLAG_STAT_BASE_BRUTE_SET not implemented yet in LOAD_DATA_BY_VAR_AND_RESOL mode!')
    # Liste de données brutes separéee par set (A,T,V) pour OUT et pour IN
    VAout_brute, VVout_brute, VTout_brute = data_repartition(V_data_list, couple_var_reso_list,
                                                             varOut, ResoOut, indA, indV, indT)
    VAin_brute, VVin_brute, VTin_brute = data_repartition(V_data_list, couple_var_reso_list,
                                                          varIn, ResoIn, indA, indV, indT)
    # Liste de dictionnaires de dimensions ('time','lat','lat_border', ...) separés pour OUT et pour IN
    Din_dico_list = dic_dimension_repartition(D_dico_list, couple_var_reso_list, varIn, ResoIn) 
    Dout_dico_list = dic_dimension_repartition(D_dico_list, couple_var_reso_list, varOut, ResoOut) 
else:
    FdataAllVar,varlue,diccoord = load_resac_data("natl60_htuv_01102012_01102013.npz")
    #
    time_axis = diccoord['time']    
    Nvar_, Nimg_, _, _ = np.shape(FdataAllVar) #(4L, 366L, 1296L, 1377L)
    #
    if FLAG_STAT_BASE_BRUTE : # Statistiques de base sur les variables brutes lues (->PPT)
        print("Statistique de base sur les variables brutes lues")
        stat2base(FdataAllVar, varlue)
        if LIMLAT_NORDSUD > 0 : # stat de base Nord-Sud
            for i in np.arange(len(FdataAllVar)) :
                XiNord_, XiSud_ = splitns(FdataAllVar[i], LIMLAT_NORDSUD)
                print("%s Nord: "%varlue[i], end='')
                statibase(XiNord_)
                print("%s Sud : "%varlue[i], end='')
                statibase(XiSud_)
            del XiNord_, XiSud_
    #
    # Splitset Ens App - Val - Test
    print("Splitset Ens App - Val - Test ...", end='')
    indA, indV, indT = isetalea(Nimg_, pcentSet)
    VA_brute = []
    VV_brute = []
    VT_brute = []
    for i in np.arange(Nvar_) : # Pour chaque variable (i.e. liste) (dans l'ordre de tvwmm ...)
        VA_brute.append(FdataAllVar[i][indA])
        VV_brute.append(FdataAllVar[i][indV])
        VT_brute.append(FdataAllVar[i][indT])
    #
    del FdataAllVar #<<<<<<<  



    if FLAG_STAT_BASE_BRUTE_SET : #pass
        # Stats de base en donn�es brute par ensemble
        print("APP :") 
        stat2base(VA_brute, varlue)
        print("VAL :")
        stat2base(VV_brute, varlue)
        print("TEST:")
        stat2base(VT_brute, varlue)
    #
    if FLAG_HISTO_VAR_BRUTE_SET : #pass
        # Hist de comparaison des distributions par variable et par ensemble (->draft, ppt)
        for i in np.arange(len(varlue)) :
            plt.figure() 
            plt.suptitle("Histo %s"%varlue[i])
            
            H_ = VA_brute[i]
            plt.subplot(3,1,1) 
            plt.hist(H_.ravel(), bins=50)
            plt.title("APP")
            
            H_ = VV_brute[i]
            plt.subplot(3,1,2)
            plt.hist(H_.ravel(), bins=50)
            plt.title("VAL")
            
            H_ = VT_brute[i]
            plt.subplot(3,1,3) 
            plt.hist(H_.ravel(), bins=50)
            plt.title("TEST")
        plt.show()
    #
    #

    VAout_brute, VVout_brute, VTout_brute, VAin_brute, VVin_brute, VTin_brute \
    = setresolution(VA_brute,VV_brute,VT_brute,varlue,ResoIn,ResoOut)

    del VA_brute, VV_brute, VT_brute

print("# Mise en forme") 
for i in np.arange(NvarIn) :  # App In
    NdonA, pixlinA, pixcinA  = np.shape(VAin_brute[i])   # Nombre de donn�es et tailles
    VAin_brute[i] = VAin_brute[i].reshape(NdonA,1,pixlinA,pixcinA)
for i in np.arange(NvarOut) : # App Out
    NoutA, pixloutA, pixcoutA = np.shape(VAout_brute[i])
    VAout_brute[i] = VAout_brute[i].reshape(NdonA,1,pixloutA,pixcoutA)   
if NdonA != NoutA :
    raise ValueError("Probl�me A") # ce n'est pas suffisant

if TEST_ON :
    for i in np.arange(NvarIn) :  # Tst In
        NdonT, pixlinT, pixcinT  = np.shape(VTin_brute[i])     # Nombre de donn�es et tailles
        VTin_brute[i] = VTin_brute[i].reshape(NdonT,1,pixlinT,pixcinT)
    for i in np.arange(NvarOut) : # Tst Out       
        NoutT, pixloutT, pixcoutT = np.shape(VTout_brute[i])
        VTout_brute[i] = VTout_brute[i].reshape(NdonT,1,pixloutT,pixcoutT)
    if NdonT != NoutT :
        raise ValueError("Probl�me T") # ce n'est pas suffisant
if VALID_ON : 
    for i in np.arange(NvarIn) :  # Val In
        NdonV, pixlinV, pixcinV  = np.shape(VVin_brute[i]) # Nombre de donn�es et tailles
        VVin_brute[i] = VVin_brute[i].reshape(NdonV,1,pixlinV,pixcinV)
    for i in np.arange(NvarOut) : # Val Out       
        NoutV, pixloutV, pixcoutV = np.shape(VVout_brute[i])
        VVout_brute[i] = VVout_brute[i].reshape(NdonV,1,pixloutV,pixcoutV)
    if NdonV != NoutV :
        raise ValueError("Probl�me V") # ce n'est pas suffisant

#======================================================================
#                   CODIFICATION / NORMALISATION
#======================================================================
print("# Codification / Normalisation")
# PLM, CE DOIT ETRE OBLIGATOIRE car la sauvegarde des param�tres n'est 
# pas faite, Il faut repasser ici pour les recalculer � chaque fois
VAin = []
coparmAin = []
for i in np.arange(NvarIn) :
    VAin_,  coparmAin_  = codage(VAin_brute[i],  "fit01")
    print(coparmAin_)
    VAin.append(VAin_)
    coparmAin.append(coparmAin_)
del VAin_, coparmAin_
x_train = VAin
NcanIn  = len(x_train)
#
VAout = []
coparmAout = []
for i in np.arange(NvarOut) :
    VAout_,  coparmAout_  = codage(VAout_brute[i],  "fit01")
    print(coparmAout_)
    VAout.append(VAout_)
    coparmAout.append(coparmAout_)
del VAout_, coparmAout_
parametre = np.array([coparmAin,coparmAout],dtype=object)
print("COPARM", parametre)
y_train = VAout
NensA   = len(y_train[0])

if TEST_ON : # Il faut appliquer le mï¿½me codage et dans les mï¿½mes conditions
    # (i.e. avec les mï¿½mes paramï¿½tres) que ceux de l'apprentissage.
    VTin = []
    for i in np.arange(NvarIn) :
        VTin_ = recodage(VTin_brute[i], coparmAin[i])
        VTin.append(VTin_)
    del VTin_
    x_test = VTin
    #
    VTout = []
    for i in np.arange(NvarOut) :
        VTout_ =  recodage(VTout_brute[i], coparmAout[i])
        VTout.append(VTout_)
    del VTout_
    #
    y_test = VTout
    NensT = len(y_test[0])

if VALID_ON : # Il faut appliquer le mï¿½me codage et dans les mï¿½mes conditions
    # (i.e. avec les mï¿½mes paramï¿½tre) que ceux de l'apprntissage.
    VVin = []
    for i in np.arange(NvarIn) :
        VVin_ = recodage(VVin_brute[i], coparmAin[i])
        VVin.append(VVin_)
    del VVin_
    x_valid = VVin
    #
    VVout = []
    for i in np.arange(NvarOut) :
        VVout_ =  recodage(VVout_brute[i], coparmAout[i])
        VVout.append(VVout_)
    del VVout_
    #
    y_valid = VVout
    NensV   = len(y_valid[0])

# POUR AVOIR CHANEL LAST, en Linux dans ~/.keras/keras.json
# Windowd c:/Users/charles/.keras/keras.json
#y_test_brute=[]
#y_train_brute=[]
for i in np.arange(NvarIn):
    x_train[i] = x_train[i].transpose(0,2,3,1)
    x_valid[i] = x_valid[i].transpose(0,2,3,1)
    x_test[i] = x_test[i].transpose(0,2,3,1)
for i in np.arange(NvarOut):
    y_train[i] = y_train[i].transpose(0,2,3,1)
    y_valid[i] = y_valid[i].transpose(0,2,3,1)
    y_test[i] = y_test[i].transpose(0,2,3,1)
#    y_test_brute.append( VTout_brute[i].transpose(0,2,3,1))
#    y_train_brute.append(VAout_brute[i].transpose(0,2,3,1))

#======================================================================
#######################################################################
#                       THE ARCHITECTURE
#######################################################################
#======================================================================

#Sur le Notebook je travaille sur 1/3 des données pour faciliter les calculs soit 4 mois 
for k in range(len(x_train)):
  print(f"Dimension de x_train[{k}] : {x_train[k].shape}")
for k in range(len(y_train)):
  print(f"Dimension de y_train[{k}] : {y_train[k].shape}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from torchsummary import summary

from tqdm import tqdm


dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = torch.device(dev)


class Resac(nn.Module):
  def __init__(self):
    super(Resac,self).__init__()
    #UpSampling d'un facteur (3x3)
    self.Up = nn.Upsample(scale_factor=3, mode='bicubic', align_corners=True) #Agrandissemnt de la taille de l'image

    #Couches d'input 
    self.conv_inputR8127 = nn.Conv2d(in_channels=2,out_channels=32,kernel_size=(3,3), padding=1,padding_mode='replicate')
    self.conv_inputR2709 = nn.Conv2d(in_channels=2,out_channels=23,kernel_size=(3,3), padding=1,padding_mode='replicate')

    #Couches différentes à chaque résolution
    self.conv_32x32 = nn.Conv2d(32,32,kernel_size=(3,3), padding=1,padding_mode='replicate')
    self.conv_23x23 = nn.Conv2d(23,23,kernel_size=(3,3), padding=1,padding_mode='replicate')

    #BatchNormalisation toutes les 2 convolutions
    self.BN1 = nn.BatchNorm2d(32)
    self.BN2 = nn.BatchNorm2d(23)

    #Couches de sorties
    self.conv_32x8 = nn.Conv2d(32,8, kernel_size=(3,3), padding=1, padding_mode='replicate')
    self.conv_23x8 = nn.Conv2d(23,8, kernel_size=(3,3), padding=1, padding_mode='replicate')
    self.conv_8x1 = nn.Conv2d(8,1, kernel_size=(3,3), padding=1, padding_mode='replicate')
    self.conv_8x2 = nn.Conv2d(8,2, kernel_size=(3,3), padding=1, padding_mode='replicate')


  def forward(self, x):
    #y = self.Up1(x[0]); 
    factiv = nn.LeakyReLU()
    out = torch.cat((self.Up(x[0]),x[1]),1);
    out = self.conv_inputR8127(out); out = factiv(out);

    for k in range(5):
      out = self.conv_32x32(out); out = factiv(out);
      out = self.conv_32x32(out); out = factiv(out);
      out = self.BN1(out)

    out = self.conv_32x8(out); out = factiv(out);
    out = self.conv_8x1(out); torch.sigmoid(out);

    
    #out = self.Up(); 
    out = torch.cat((self.Up(out),x[2]),1);
    out = self.conv_inputR2709(out); out = factiv(out);

    for k in range(3):
      out = self.conv_23x23(out); out = factiv(out);
      out = self.conv_23x23(out); out = factiv(out);
      out = self.BN2(out)

    out = self.conv_23x8(out); out = factiv(out);
    out = self.conv_8x2(out); torch.relu(out);

    return out

def toTensor(x):
  X = []
  for idx in range(len(x)):
    shape = (x[idx].shape[0],1,x[idx].shape[1],x[idx].shape[2]) #Channel first
    X_int = torch.Tensor([i for i in x[idx]]).view(shape).to(dev)
    #X_int = X_int.to(device)
    X.append(X_int)
    #X.append(torch.Tensor([i for i in x[idx]]).view(shape))
  return X
  
x_trainPT = toTensor(x_train)
x_validPT = toTensor(x_valid)
y_trainPT = toTensor(y_train)
y_validPT = toTensor(y_valid)

dir_model ='Save_Model/PyTorch/Net_ACRATOPOTES_E10000-BS32_20210719-124030/Model/Trained_model-E10000-BS32_1.pt'
loaded_checkpoint = torch.load(dir_model)
trained_model = Resac()
trained_model.load_state_dict(loaded_checkpoint['model_state'])
trained_model.eval()
trained_model.to(device)

def predict(model,X):
  with torch.no_grad():
     prediction = model(X)
  return prediction

pred = predict(trained_model, x_trainPT).cpu().numpy()
print("Shape", pred.shape)
#SSH_dec = decodage_sortie(pred,parametre)#.cpu().numpy()
#SSH_dec_flatten = pred.reshape(-1)


