# -*- coding: utf-8 -*-
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

################################# ARCHITECTURE CLASSIQUE DE RESAC EN PYTORCH DE R81 A R27 ######################"

Niter, Bsize = 5000, 32
lr = 0.001#2*(10**(-2.89637961))#learning rate utiisé dans l'architecture

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

SSH_true = VAout_brute[-1]

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
from copy import deepcopy
from tqdm import tqdm

#######Link to GPU#######@
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(dev)

########Folder creation#######@
nowstr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

Mdl2save = 'Save_Model/PyTorch'
directory = f"Net_{current_hostname.upper()}_E{Niter}-BS{Bsize}_{nowstr}"
dir_name = os.path.join(Mdl2save, directory)
dir_image = os.path.join(dir_name, 'Images')
dir_model = os.path.join(dir_name, 'Model')

os.makedirs('Save_Model',exist_ok = True)
os.makedirs(Mdl2save,exist_ok = True)
os.makedirs(dir_name,exist_ok = True)
os.makedirs(dir_image,exist_ok = True)
os.makedirs(dir_model,exist_ok = True)


#torch.save(trained_model.state_dict(),os.path.join(dir_model,f'Trained_model-E{Niter}-BS{Bsize}_1.pt'))


#======================================================================
#######################################################################
#                       THE ARCHITECTURE
#######################################################################
#======================================================================
'''
class ResacR09(nn.Module):
  def __init__(self):
    super(ResacR09,self).__init__()
    #UpSampling d'un facteur (3x3)
    self.Up = nn.Upsample(scale_factor=3, mode='bicubic', align_corners=True) #Agrandissemnt de la taille de l'image

    #Couches d'input 
    self.conv_inputR8127 = nn.Conv2d(in_channels=2,out_channels=32,kernel_size=(3,3), padding='same')#,padding_mode='replicate')
    self.conv_inputR2709 = nn.Conv2d(in_channels=2,out_channels=16,kernel_size=(3,3), padding='same')#,padding_mode='replicate')

    #Couches différentes à chaque résolution
    self.conv_32x32 = nn.Conv2d(32,32,kernel_size=(3,3), padding='same')#,padding_mode='replicate')
    self.conv_23x23 = nn.Conv2d(16,16,kernel_size=(3,3), padding='same')#,padding_mode='replicate')

    #BatchNormalisation toutes les 2 convolutions
    self.BN1 = nn.BatchNorm2d(32)
    self.BN2 = nn.BatchNorm2d(16)

    #Couches de sorties
    self.conv_32x8 = nn.Conv2d(32,8, kernel_size=(3,3), padding='same')#, padding_mode='replicate')
    self.conv_23x8 = nn.Conv2d(16,8, kernel_size=(3,3), padding='same')#, padding_mode='replicate')
    self.conv_8x1 = nn.Conv2d(8,1, kernel_size=(1,1), padding='same')#, padding_mode='replicate')
    self.conv_8x2 = nn.Conv2d(8,2, kernel_size=(1,1), padding='same')#, padding_mode='replicate')


  def forward(self, x):
    #y = self.Up1(x[0]); 
    factiv = nn.LeakyReLU() #F.silu()
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
    out = self.conv_8x1(out); torch.sigmoid(out);

    return out
'''
class ResacR27(nn.Module):
  def __init__(self):
    super(ResacR27,self).__init__()
    #UpSampling d'un facteur (3x3)
    self.Up = nn.Upsample(scale_factor=3, mode='bicubic', align_corners=True) #Agrandissemnt de la taille de l'image

    #Couches d'input 
    self.conv_inputR8127 = nn.Conv2d(in_channels=2,out_channels=32,kernel_size=(3,3), padding='same')#,padding_mode='replicate')

    #Couches différentes à chaque résolution
    self.conv_32x32 = nn.Conv2d(32,32,kernel_size=(3,3), padding='same')#,padding_mode='replicate')

    #BatchNormalisation toutes les 2 convolutions
    self.BN1 = nn.BatchNorm2d(32)

    #Couches de sorties
    self.conv_32x8 = nn.Conv2d(32,8, kernel_size=(3,3), padding='same')#, padding_mode='replicate')
    self.conv_8x1 = nn.Conv2d(8,1, kernel_size=(1,1), padding='same')#, padding_mode='replicate')


  def forward(self, x):
    #y = self.Up1(x[0]); 
    factiv = nn.LeakyReLU() #F.silu()
    out = torch.cat((self.Up(x[0]),x[1]),1);
    out = self.conv_inputR8127(out); out = factiv(out);

    for k in range(5):
      out = self.conv_32x32(out); out = factiv(out);
      out = self.conv_32x32(out); out = factiv(out);
      out = self.BN1(out)

    out = self.conv_32x8(out); out = factiv(out);
    out = self.conv_8x1(out); torch.sigmoid(out);
    
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

class custom_loss(torch.nn.Module):
  def __init__(self):
    super(custom_loss,self).__init__()

  def forward(self,x,y):
    pred = x.reshape(-1)
    output = y[-1].reshape(-1)
    loss = nn.MSELoss()
    return torch.square(loss(pred,output))

def train_model(model, x_train,y_train,x_valid,y_valid,  EPOCHS=Niter, BATCH_SIZE=Bsize):
  loss_list_train = []
  loss_list_valid = []

  criterion = custom_loss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  best_loss= 1000000
  for epoch in range(EPOCHS):
    print(f"Epoch n° : {epoch}/{Niter} commencée")
    loss_train = 0
    loss_valid = 0
    

    for i in tqdm(range(0, len(x_train), BATCH_SIZE)):
      batch_X = []
      batch_y = []
      for k in range (len(x_train)):
        batch_X.append(Variable(x_train[k][i:i+BATCH_SIZE]))
      for k in range(len(y_train)):
        batch_y.append(Variable(y_train[k][i:i+BATCH_SIZE]))

      optimizer.zero_grad()

      outputs = model(batch_X)
      loss = criterion(outputs,batch_y)
      loss.backward()
      optimizer.step()

      loss_train += loss
      print("\n","loss par Batch train=",loss,"\n")
    
    model.eval()     # Optional when not using Model Specific layer
    for i in tqdm(range(0, len(x_valid), 6)):
      #x_valid = Variable(x_valid)
      batchv_X = []
      batchv_y = []

      for k in range (len(x_valid)):
        batchv_X.append(Variable(x_valid[k][i:i+BATCH_SIZE]))
      for k in range(len(y_valid)):
        batchv_y.append(Variable(y_valid[k][i:i+BATCH_SIZE]))

      target = model(batchv_X)
      loss = criterion(target,batchv_y)
      loss_valid += loss
      print("\n","loss par Batch valid=",loss,"\n")
      
    print("\n","loss par epoch train =",loss_train)
    print("\n","loss par epoch valid =",loss_valid)

    
    loss_list_train.append(loss_train.item())
    loss_list_valid.append(loss_valid.item())
    
    if best_loss > loss_valid.item(): 
      checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }
      torch.save(checkpoint ,os.path.join(dir_model,f'Trained_model-E{Niter}-BS{Bsize}_1.pth'))
      
      best_loss = loss_valid.item()
      

  print('Finish training')
  return model, loss_list_train,loss_list_valid

model = ResacR27()
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        #xavier(m.bias.data)

model.apply(weights_init)
model.cuda()


trained_model, loss_list_train, loss_list_valid = train_model(model, x_trainPT, y_trainPT, x_validPT, y_validPT,  EPOCHS=Niter,BATCH_SIZE=Bsize)

loss_list_train_1 = np.array(loss_list_train)
loss_list_valid_1 = np.array(loss_list_valid)
np.save(os.path.join(dir_name,'train_loss'), loss_list_train_1)
np.save(os.path.join(dir_name,'valid_loss'),loss_list_valid_1)
print("loss train", loss_list_train,"\n")
print("loss valid", loss_list_valid,"\n")


############Loss Plot##############
epochs = np.arange(Niter)
plt.plot(epochs,loss_list_train_1, color='r', label='Training loss 1')
plt.plot(epochs,loss_list_valid_1, color='b', label='Valid loss 1')

plt.yscale("log")

plt.title(f"Loss function avec {Niter} EPOCHS et {Bsize} Batch Size")

plt.legend(fontsize=20)
plt.grid()
plt.savefig(os.path.join(dir_image, "loss function"))
plt.clf()

###############Trained model loading#########@

loaded_checkpoint = torch.load(os.path.join(dir_model,f'Trained_model-E{Niter}-BS{Bsize}_1.pth'))
trained_model = ResacR27()
trained_model.load_state_dict(loaded_checkpoint['model_state'])
trained_model.to(device)

##############Prédictions##########
#trained_model = ResacR09()
#trained_model.load_state_dict(torch.load(os.path.join(dir_model,f'Trained_model-E{Niter}-BS{Bsize}_1.pt')))
def decodage_sortie(predictions,parametre):
  ''' 'Dé-normalisation' des prédictions pour les comparer aux données modèles'''
  #SSH_R27_dec = decodage(predictions[0], parametre[1][0])
  SSH_R09_dec = decodage(predictions, parametre[1][0])
  #U_dec = decodage(predictions[2], parametre[1][2])
  #V_dec = decodage(predictions[3], parametre[1][3])
  return  SSH_R09_dec

def predict(model,X):
  with torch.no_grad():
     prediction = model(X)
  return prediction

parametre = np.array([coparmAin,coparmAout],dtype=object)

pred = predict(trained_model, x_trainPT)
SSH_dec = decodage_sortie(pred,parametre).cpu().numpy()
SSH_dec_flatten = SSH_dec.reshape(-1)

#SSH_true = decodage_sortie(y_trainPT[-1],parametre).cpu().detach().numpy()
print(SSH_true.shape)
SSH_true_flatten = SSH_true.reshape(-1)
from scipy.stats import linregress
#a, b , R2 = linregress(SSH_dec_flatten, SSH_true)



def reg(a,b,x):
    return a*x + b 
b, a , R2= linr2(SSH_dec_flatten, SSH_true_flatten)
x=np.linspace(-1,1.15,25)

plt.plot(x, reg(a,b,x), color='red', label=f"Régression linéaire, reg(x) = {round(a,4)}x + {round(b,4)}", linewidth=2)
plt.plot(x,x, color='g', label='Identité')
plt.xlabel("NATL60")
plt.ylabel("RESAC")
plt.title(f"Scatter plot : {SSH_dec_flatten.shape} pixels. R2 : {round(R2,5)}")
plt.scatter(SSH_true, SSH_dec_flatten,  marker='.', color='b', s=3, label="Nuage de points")
plt.savefig(os.path.join(dir_image, "scatter_plot"))

plt.clf()
jour = 5
print(SSH_dec.shape)
shap = SSH_dec.shape[2]
plt.imshow(SSH_dec[jour].reshape(shap,-1), origin='lower')
plt.savefig(os.path.join(dir_image, "Prediction"))
plt.clf()
plt.imshow(SSH_true[jour].reshape(shap,-1), origin="lower")
plt.savefig(os.path.join(dir_image, "Réel SSH"))
