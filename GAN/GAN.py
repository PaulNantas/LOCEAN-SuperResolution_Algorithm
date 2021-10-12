# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
from   time  import time
import datetime
import numpy as np
import pandas as pd
#import pickle
#import random
import matplotlib.pyplot as plt
#from   matplotlib  import cm
from   resacartdef import *
import time
#
def load_resac_data(npz_data_file) :
    """
    Exemple d'usage:
        FdataAllVar,varlue = load_resac_data("natl60_htuv_01102012_01102013.npz")

    Lecture des données RESAC.  La function s'attend à trouver le repertoire
    des données dans la variable d'environnement RESAC_DATASETS_DIR.

    Retourne l'array 4D des donnees ([nb.variable, np.time steps, y size, x size])
    et la liste des noms de variables dans l'array.
    """
    # ---- datasets location

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
    # join dataset dir with filename
    data_set_filename = os.path.join(datasets_dir,npz_data_file)

    # Lecture Des Donnees
    print(f"Lecture Des Donnees du fichier {npz_data_file} ... ", end='', flush=True)
    #Data_       = np.load(data_set_filename)
    Data_       = np.load("../../donnees/natl60_htuv_01102012_01102013.npz")
    FdataAllVar = Data_['FdataAllVar']
    varlue      = list(Data_['varlue'])
    varlue      = ['U' if i==b'SSU' else 'V' if i==b'SSV' else i.decode() for i in varlue] # Pour enlever les b devant les chaines de caracteres lors de la lecture et pour le conversion de 'SSU','SSV' en 'U','V'
    print(f'\nArray avec {len(varlue)} variables: {varlue}')
    print(f'contenant des images de taille {FdataAllVar.shape[2:]} pixels')
    print(f'et {FdataAllVar.shape[1]} pas de temps (une image par jour).')
    print(f'Dimensions de l\'array: {np.shape(FdataAllVar)}')
    #
    _, Nimg_, Nlig_, Ncol_ = np.shape(FdataAllVar) #(4L, 366L, 1296L, 1377L)
    #
    dimensions = {}
    # Coordonnees : TIME
    dimensions['time'] = pd.date_range("2012-10-01", periods=Nimg_)
    # Limites absoluts NATL60 de la zone geographique (au centre des pixels)
    nav_lat = [ 26.57738495,  44.30360031]
    nav_lon = [-64.41895294, -40.8841095 ]
    # Coordonnees : Lat / Lon (au centre du pixel)
    all_lat = np.linspace(nav_lat[0],nav_lat[1],num=Nlig_) # latitude of center of the pixel
    all_lon = np.linspace(nav_lon[0],nav_lon[1],num=Ncol_)
    delta_lat = (all_lat[1]-all_lat[0])
    delta_lon = (all_lon[1]-all_lon[0])
    # Bords inferieur et supperieur des pixels
    dimensions['lat'] = all_lat
    dimensions['lon'] = all_lon
    dimensions['lat_border'] = [nav_lat[0] - delta_lat/2, nav_lat[1] + delta_lat/2] # latitude border inf and sup for the zone
    dimensions['lon_border'] = [nav_lon[0] - delta_lon/2, nav_lon[1] + delta_lon/2] # longitude border inf and sup for the zone
    #
    return FdataAllVar,varlue,dimensions
#
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

#
#%%
#======================================================================
#######################################################################
#           GET AND SET THE REQUIRED BRUTE DATA
#######################################################################
#======================================================================

# ----------------------------------------------------------------------------
# Declarez votre repertoire de donnees RESAC dans une variable d'environnement
# RESAC_DATASETS_DIR. Par exemple, en Linux:
#    export RESAC_DATASETS_DIR="~/chemin/a/mes/donnees"
# ----------------------------------------------------------------------------
# Lecture des donnees Resac
print("Lecture Des Données en cours ...");
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
    #
    if FLAG_STAT_BASE_BRUTE_SET :
        # Stats de base en donnï¿½es brute par ensemble
        print("APP :")
        stat2base(VA_brute, varlue)
        print("VAL :")
        stat2base(VV_brute, varlue)
        print("TEST:")
        stat2base(VT_brute, varlue)
    #
    if FLAG_HISTO_VAR_BRUTE_SET :
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
        #plt.show()
    #
    # Make resolution for IN and OUT
    VAout_brute, VVout_brute, VTout_brute, VAin_brute, VVin_brute, VTin_brute \
    = setresolution(VA_brute,VV_brute,VT_brute,varlue,ResoIn,ResoOut)
#%%
if CALENDAR_FROM_DATA :
    calA_ = np.array([pd.to_datetime(str(t)).strftime('%d-%b-%Y') for t in time_axis[indA]])
    calV_ = np.array([pd.to_datetime(str(t)).strftime('%d-%b-%Y') for t in time_axis[indV]])
    calT_ = np.array([pd.to_datetime(str(t)).strftime('%d-%b-%Y') for t in time_axis[indT]])
else:
    calA_ = calendrier[indA]
    calV_ = calendrier[indV]
    calT_ = calendrier[indT]
calA = [calA_, indA]
calV = [calV_, indV]
calT = [calT_, indT]
print("done (and for calendar too)");
#
print("# Mise en forme")
for i in np.arange(NvarIn) :  # App In
    NdonA, pixlinA, pixcinA  = np.shape(VAin_brute[i])   # Nombre de donnï¿½es et tailles
    VAin_brute[i] = VAin_brute[i].reshape(NdonA,1,pixlinA,pixcinA)
for i in np.arange(NvarOut) : # App Out
    NoutA, pixloutA, pixcoutA = np.shape(VAout_brute[i])
    VAout_brute[i] = VAout_brute[i].reshape(NdonA,1,pixloutA,pixcoutA)
if NdonA != NoutA :
    raise ValueError("Problï¿½me A") # ce n'est pas suffisant

if TEST_ON :
    for i in np.arange(NvarIn) :  # Tst In
        NdonT, pixlinT, pixcinT  = np.shape(VTin_brute[i])     # Nombre de donnï¿½es et tailles
        VTin_brute[i] = VTin_brute[i].reshape(NdonT,1,pixlinT,pixcinT)
    for i in np.arange(NvarOut) : # Tst Out
        NoutT, pixloutT, pixcoutT = np.shape(VTout_brute[i])
        VTout_brute[i] = VTout_brute[i].reshape(NdonT,1,pixloutT,pixcoutT)
    if NdonT != NoutT :
        raise ValueError("Problï¿½me T") # ce n'est pas suffisant
if VALID_ON :
    for i in np.arange(NvarIn) :  # Val In
        NdonV, pixlinV, pixcinV  = np.shape(VVin_brute[i]) # Nombre de donnï¿½es et tailles
        VVin_brute[i] = VVin_brute[i].reshape(NdonV,1,pixlinV,pixcinV)
    for i in np.arange(NvarOut) : # Val Out
        NoutV, pixloutV, pixcoutV = np.shape(VVout_brute[i])
        VVout_brute[i] = VVout_brute[i].reshape(NdonV,1,pixloutV,pixcoutV)
    if NdonV != NoutV :
        raise ValueError("Problï¿½me V") # ce n'est pas suffisant


if RUN_MODE=="LEARN":
    print("\nLEARN MODE ...");
    try:
        Mdl2save = os.path.join(Mdl2dirname,f"Net_{Mdl2name}_{current_hostname.upper()}_E{Niter}-BS{Bsize}_{nowstr}")
        if factoutlbl4train is not None:
            Mdl2save += f'_{factoutlbl4train.upper()}'
        print(f'Mdl2save: "{Mdl2save}"')
        historique_dir  = os.path.join(Mdl2save,"Historique_Loss")
        archi_train_dir = os.path.join(Mdl2save,"Archi")
        weights_dir     = os.path.join(Mdl2save,"Weights")
        logs_fit_dir    = os.path.join(Mdl2save,'logs','fit')
        images_dir      = os.path.join(Mdl2save,"Images")
        # Creation du repertoire de sauvegarde du produit de l'apprentissage (modele, poids, historique, images)
        os.makedirs(Mdl2save,        exist_ok = True)
        os.makedirs(historique_dir,  exist_ok = True)
        os.makedirs(archi_train_dir, exist_ok = True)
        os.makedirs(weights_dir,     exist_ok = True)
        os.makedirs(logs_fit_dir,    exist_ok = True)
        os.makedirs(images_dir,      exist_ok = True)
    except:
        print(f"Unexpected error when creating '{RUN_MODE}' directories in {Mdl2dirname}/", sys.exc_info()[0])
        raise


#%%
#======================================================================
#                   CODIFICATION / NORMALISATION
#======================================================================
print("# Codification / Normalisation")
# PLM, CE DOIT ETRE OBLIGATOIRE car la sauvegarde des paramï¿½tres n'est
# pas faite, Il faut repasser ici pour les recalculer ï¿½ chaque fois
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
y_train = VAout
NensA   = len(y_train[0])
#
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

#----------------------------------------------------------------------
# POUR AVOIR CHANNEL LAST, en Linux dans ~/.keras/keras.json
# Windows c:/Users/charles/.keras/keras.json
for i in np.arange(NvarIn):
    x_train[i] = x_train[i].transpose(0,2,3,1)
    x_valid[i] = x_valid[i].transpose(0,2,3,1)
    x_test[i] = x_test[i].transpose(0,2,3,1)
for i in np.arange(NvarOut):
    y_train[i] = y_train[i].transpose(0,2,3,1)
    y_valid[i] = y_valid[i].transpose(0,2,3,1)
    y_test[i] = y_test[i].transpose(0,2,3,1)
#----------------------------------------------------------------------

if 1 : # Affichage des shapes ...
    for i in np.arange(NvarIn) :
        print("%s shape x_train : "%varIn[i], np.shape(x_train[i]))
    for i in np.arange(NvarOut) :
        print("%s shape y_train : "%varOut[i], np.shape(y_train[i]))
    if VALID_ON :
        for i in np.arange(NvarIn) :
            print("%s shape x_valid : "%varIn[i], np.shape(x_valid[i]))
        for i in np.arange(NvarOut) :
            print("%s shape y_valid : "%varOut[i], np.shape(y_valid[i]))
    if TEST_ON :
        for i in np.arange(NvarIn) :
            print("%s shape x_test : "%varIn[i], np.shape(x_test[i]))
        for i in np.arange(NvarOut) :
            print("%s shape y_test : "%varOut[i], np.shape(y_test[i]))


#======================================================================
#######################################################################
#                       THE ARCHITECTURE
#######################################################################
#======================================================================
print("# Build and compile Architecture")

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, experimental
from keras.layers import Concatenate, concatenate, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.nn import swish
from tqdm import tqdm
import sys
import numpy as np

SSH_R27  = np.load("../../donnees/NATL60byVar/NATL60_SSH_R27.npy")[indA]
#SSH_R27 = SSH_R27[indA]

# D�termination de la prise en compte (ou pas) de la SST en entr�e
# de l'archi selon les r�solutions indiqu�es (dans resacparm.py).
all_Kinput_img = []
IS_SSTR81 = IS_SSTR27 = IS_SSTR09 = IS_SSTR03 =IS_SSTR01= False
for ii in np.arange(NvarIn) :
    #Ncan_, NL_, NC_ = np.shape(x_train[ii][0]) # CHANEL FIRST
    #all_Kinput_img.append(Input(shape=(Ncan_,NL_,NC_)))
    NL_, NC_, Ncan_  = np.shape(x_train[ii][0]) # CHANEL LAST
    all_Kinput_img.append(Input(shape=(NL_,NC_,Ncan_)))
    
    if varIn[ii]=="SST" :
        if ResoIn[ii]==81 :
            if IS_SSTR81 == False :
                ISSTR81   = all_Kinput_img[ii]
                IS_SSTR81 = True
            else :
                ISSTR81 = concatenate([ISSTR81, all_Kinput_img[ii]], axis=1)                   
        elif ResoIn[ii]==27 :
            if IS_SSTR27 == False :
                ISSTR27   = all_Kinput_img[ii]
                IS_SSTR27 = True
            else :
                ISSTR27 = concatenate([ISSTR27, all_Kinput_img[ii]], axis=1)
        elif ResoIn[ii]==9 :
            if IS_SSTR09 == False :
                ISSTR09   = all_Kinput_img[ii]
                IS_SSTR09 = True
            else :
                ISSTR09 = concatenate([ISSTR09, all_Kinput_img[ii]], axis=1)
        elif ResoIn[ii]==3 :
            if IS_SSTR03 == False :
                ISSTR03   = all_Kinput_img[ii]
                IS_SSTR03 = True
            else :
                ISSTR03 = concatenate([ISSTR03, all_Kinput_img[ii]], axis=1)
        elif ResoIn[ii]==1:
            if IS_SSTR01 == False :
                ISSTR01 = all_Kinput_img[ii]
                IS_SSTR01 = True
            else:
                ISSTR01= concatenate([ISSTR01, all_Kinput_img[ii]], axis=1)
        else :
            raise ValueError("Other resolution of SST not prevue")



class DCGAN():
    def __init__(self):
        #Input shape
        self.height   = 16
        self.width    = 17
        self.channels = 3
        self.days = 256
        self.init = 'he_normal'
        self.factiv = swish
        self.factout = 'sigmoid'
        self.upfactor = 3
        self.gen_shape = (self.height*3, self.width*3, 2)
        self.img_shape = (self.height*3, self.width*3, 1)

        optimizer = Adam(0.0002)

      # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer)
        

        # Build the generator
        self.generator = self.build_generator()

        #Input
        z = Input(shape=(self.gen_shape))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        
        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def resize(self, x_train):
      
      #input = Input(shape=(self.height,self.width,1))
      SSH = x_train[0]
      SST = x_train[1]
      Archi = UpSampling2D((self.upfactor, self.upfactor), interpolation='bilinear')(SSH)
      Archi = concatenate([Archi, SST], axis=3)

      return Archi
      
    def build_generator(self):
      first = Input(shape=((self.height*3, self.width*3, 2)))
      '''
      model= Sequential()
      for i in np.arange(3) : 
          model.add(Conv2D(32,(3,3), activation=self.factiv, padding='same',kernel_initializer=self.init))
          model.add(Conv2D(32,(3,3), activation=self.factiv, padding='same',kernel_initializer=self.init))
          model.add(BatchNormalization(axis=self.channels))
      model.add(Conv2D(8,(3,3), activation=self.factiv, padding='same',kernel_initializer=self.init))
      model.add(Conv2D(1,(1,1), activation=self.factout, padding='same',kernel_initializer=self.init,name='archiA'))
      '''
      ArchiOut = []
      for i in np.arange(3) : 
        ArchiA = Conv2D(32,(3,3), activation=self.factiv, padding='same',kernel_initializer=self.init)(first)
        ArchiA = Conv2D(32,(3,3), activation=self.factiv, padding='same',kernel_initializer=self.init)(ArchiA)
        ArchiA = BatchNormalization(axis=3)(ArchiA)
      ArchiA = Conv2D(8,(3,3), activation=self.factiv, padding='same',kernel_initializer=self.init)(ArchiA)
      ArchiA = Conv2D(1,(1,1), activation=self.factout, padding='same',kernel_initializer=self.init,name='archiA')(ArchiA)
      ArchiOut.append(ArchiA)
      
      #model.build(self.gen_shape)
      #model.summary()
      #input = modelinput([self.input,self.SST27])
      model = Model(first,ArchiOut)
      img = model(first) 
      return Model(first,img)


    def build_discriminator(self):

      model = Sequential()

      model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
      model.add(LeakyReLU(alpha=0.2))
      model.add(Dropout(0.25))
      model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
      model.add(ZeroPadding2D(padding=((0,1),(0,1))))
      model.add(BatchNormalization(momentum=0.8))
      model.add(LeakyReLU(alpha=0.2))
      model.add(Dropout(0.25))
      model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
      model.add(BatchNormalization(momentum=0.8))
      model.add(LeakyReLU(alpha=0.2))
      model.add(Dropout(0.25))
      model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
      model.add(BatchNormalization(momentum=0.8))
      model.add(LeakyReLU(alpha=0.2))
      model.add(Dropout(0.25))
      model.add(Flatten())
      model.add(Dense(1, activation='sigmoid'))

      model.summary()

      img = Input(shape=self.img_shape)
      validity = model(img)

      return Model(img, validity)

    def train(self, epochs=200, batch_size=61, save_interval=50):

      # Adversarial ground truths
      valid = np.ones((self.days, 1))
      fake = np.zeros((self.days, 1))


      d_loss_real = []
      d_loss_fake = []
      d_loss= []
      g_loss= []
      t0 = time.time()

      for epoch in range(epochs):
        if epoch%100==0:
           print("EPOCH n°",epoch)
        t1= time.time()

        loss_real_epoch = 0
        loss_fake_epoch = 0
        loss_d_epoch = 0
        loss_g_epoch = 0

        for i in tqdm(range(0, self.days, batch_size)):

          # ---------------------
          #  Train Discriminator
          # ---------------------

          # Select a random half of images
          imgs = SSH_R27[i:i+batch_size]
          print(SSH_R27.shape)
          print(x_train[0].shape)
          print(valid.shape)

          # Sample noise and generate a batch of new images
          first = [x_train[i][i:i+batch_size] for i in range(2)]
          
          first = self.resize(x_train=first)

          gen_imgs = self.generator.predict(first)

          # Train the discriminator (real classified as ones and generated as zeros)
          d_loss_real_ = self.discriminator.train_on_batch(imgs, valid[i:i+batch_size])
          d_loss_fake_ = self.discriminator.train_on_batch(gen_imgs, fake[i:i+batch_size])
          d_loss_ = 0.5 * np.add(d_loss_real_, d_loss_fake_)

          loss_real_epoch += d_loss_real_
          loss_fake_epoch += d_loss_fake_
          loss_d_epoch += d_loss_
          # ---------------------
          #  Train Generator
          # ---------------------

          # Train the generator (wants discriminator to mistake images as real)
          g_loss_ = self.combined.train_on_batch(first, valid[i:i+batch_size])

          loss_g_epoch+=g_loss_
          # Plot the progress
        
          print ("Loss par batch : " ,(epoch, d_loss_, 100*d_loss_, g_loss_))
        
        d_loss_real.append(loss_real_epoch)
        d_loss_fake.append(loss_fake_epoch)
        d_loss.append(loss_d_epoch)
        g_loss.append(loss_g_epoch)

        print("Temps d'une epoch", time.time()-t1)

      print("Loss discriminator:",d_loss )
      print("Loss combined:", g_loss )
      print("Temps total de l'entrainement", time.time()-t0)
      return d_loss, g_loss , self.generator, self.discriminator, self.combined

EPOCHS, BATCH_SIZE = 10000, 32
if __name__ == '__main__':
    dcgan = DCGAN()
    d_loss, g_loss, gene, disc, comb = dcgan.train(epochs=EPOCHS, batch_size=BATCH_SIZE)

nowstr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
folder=f"Version{EPOCHS}EPOCHS-{BATCH_SIZE}BSIZE-{nowstr}"
dir_label=os.path.join('Save_Model/GAN/',folder)
os.makedirs(dir_label, exist_ok=True)
os.makedirs(os.path.join(dir_label, 'Images'), exist_ok=True)
os.makedirs(os.path.join(dir_label, 'Archis'), exist_ok=True)
os.makedirs(os.path.join(dir_label, 'Weights'), exist_ok=True)

gene.save(os.path.join(dir_label,'Archis/generator'))
comb.save(os.path.join(dir_label,'Archis/combined'))
disc.save(os.path.join(dir_label,'Archis/discriminator'))

os.makedirs(os.path.join(dir_label,'Weights/generator'))
os.makedirs(os.path.join(dir_label,'Weights/combined'))
os.makedirs(os.path.join(dir_label,'Weights/discriminator'))

gene.save_weights(os.path.join(dir_label,'Weights/generator/generator'))
comb.save_weights(os.path.join(dir_label,'Weights/combined/combined'))
disc.save_weights(os.path.join(dir_label,'Weights/discriminator/discriminator'))
print("Poids enregistrés")
gene.load_weights(os.path.join(dir_label,'Weights/generator/generator'))
print("Poids load")
plt.plot(np.arange(EPOCHS), d_loss, label='disciminator')
plt.plot(np.arange(EPOCHS), g_loss, label='combined')
plt.yscale("log")
plt.legend()
plt.savefig(os.path.join(dir_label,'Images/Loss.png'))
#plt.show()
