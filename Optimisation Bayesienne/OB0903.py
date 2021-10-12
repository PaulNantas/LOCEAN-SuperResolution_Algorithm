from resacartparm import *
from resacartdef import *

############################!!!!!!!!!!!!!!!!!!!!!!##################################
#Modifier SCENARCHI = 4 dans resacartparm.py
#Scénario ayant comme entrée ["SSH"09,"SST"03] et en sortie ["SSH"03]
####################################################################################

from   matplotlib  import cm
import sys
from   time  import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os



def load_resac_by_var_and_resol(varIn,varOut,ResoIn,ResoOut, subdir='NATL60byVar'):
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
    #
    # Lecture Des Donnees    
    V_data_list = []; D_dico_list = []
    for v,r in couple_var_reso_list:
        print(f"loading data: '{v}' at R{r}")
        data_tmp = np.load(os.path.join(datasets_dir,subdir,f"NATL60_{v.upper()}_R{r:02d}.npy"))
        dimension_tmp = np.load(os.path.join(datasets_dir,subdir,f"NATL60_coords_R{r:02d}.npz"))
        dico_dim = { 'time': dimension_tmp['time'],
                     'lat' : dimension_tmp['latitude'],
                     'lon' : dimension_tmp['longitude'],
                     'lat_border' : dimension_tmp['latitude_border'],
                     'lon_border' : dimension_tmp['longitude_border'] }
        V_data_list.append(data_tmp)
        D_dico_list.append(dico_dim)
    #
    return V_data_list, couple_var_reso_list, D_dico_list


print("Lecture Des Données en cours ...");

if LOAD_DATA_BY_VAR_AND_RESOL :
    V_data_list, couple_var_reso_list, D_dico_list = load_resac_by_var_and_resol(varIn,varOut,ResoIn,ResoOut)
    #
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



#======================================================================
#######################################################################
#                       THE ARCHITECTURE
#######################################################################
#======================================================================
print("# Build and compile Architecture")
from keras.layers import Input, Dense, Flatten, Reshape, AveragePooling2D, Dropout 
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D #, Deconvolution2D
from keras.layers import Concatenate, concatenate, BatchNormalization
from keras.models    import Model
from keras.models    import load_model
import keras.callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from keras import optimizers

if 1 : # Avoir sa propre fonction d'activation
    from keras.layers import Activation
    from keras import backend as K
    from keras.utils.generic_utils import get_custom_objects
    def sig010(x) : # sigmo�d dans l'intervalle [1.10 , 0.10]
        return  (K.sigmoid(x) * 1.20) - 0.10
    get_custom_objects().update({'sig01': Activation(sig010)})


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
#


from functools import partial
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from tensorflow.nn import swish

#THE ARCHITECTURE TO OPTIMIZE
def my_model(Nconv1, Nfilt1):
  ArchiOut = []
  upfactor = 3
  factiv= 'relu'
  init       = 'he_normal'   #'glorot_normal'+ #'orthogonal' #'normal'
  factout  = 'sigmoid'
  
  ISSHreso = all_Kinput_img[0] # Input SSH reso 81
  ArchiC   = UpSampling2D((upfactor, upfactor), interpolation='bilinear')(ISSHreso)
  #ArchiB   = concatenate([ArchiB, ISSTR27], axis=1) # CHANNEL FIRST
  ISSTR03 = all_Kinput_img[1]
  ArchiC   = concatenate([ArchiC, ISSTR03], axis=3) # CHANNEL LAST
  for i in np.arange(Nconv1) : #7
    #ArchiB = Conv2D(36,(6,6), activation=factiv, padding='same',kernel_initializer=init)(ArchiB)
    ArchiC = Conv2D(Nfilt1,(3,3), activation=factiv, padding='same',kernel_initializer=init)(ArchiC)
    ArchiC = Conv2D(Nfilt1,(3,3), activation=factiv, padding='same',kernel_initializer=init)(ArchiC)
    ArchiC = BatchNormalization(axis=3)(ArchiC)
  ArchiC = Conv2D(8,(3,3), activation=factiv, padding='same',kernel_initializer=init)(ArchiC)
  ArchiC = Conv2D(1,(1,1), activation=factout, padding='same',kernel_initializer=init,name='ArchiC')(ArchiC)
  ArchiOut.append(ArchiC);

  print(f"Les paramètres choisies sont : NConv : {Nconv1}, Nfilt : {Nfilt1}")
  return Model(all_Kinput_img, ArchiOut)



def fit_with(Nconv1,  Nfilt1, lr):

  model = my_model(Nconv1,  Nfilt1)

    # Train the model for a specified number of epochs.
  optimizer = optimizers.Adam(learning_rate=2*(10**lr))
  model.compile(loss='logcosh',
                optimizer=optimizer)
# Train the model with the train dataset.
  H = model.fit(x_train, y_train, epochs=Niter,batch_size=Bsize, 
            shuffle=True, verbose=2, validation_data=(x_valid, y_valid))
  return  -min(H.history['loss'])

fit_with_partial = partial(fit_with)
folder_BO = f"bayesianOpt_InitP-{init_points}_NIter-{n_iter}"
bayes_result = os.path.join(Mdl2save,folder_BO )
os.makedirs(bayes_result,   exist_ok = True)

# Bounded region of parameter space
pbounds = {'Nconv1': (1, 5), 'Nfilt1': (2,24),  'lr': (-4.5,-1) } #function 0-1: linear, 1-2: sig01, 2-3: sigmoid
#If nc or nl are not integers, they are rounded up to the higher integer for Conv2D and number of loops

#Apply the bayesian optimization
optimizer = BayesianOptimization(
    f=fit_with_partial,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    
)

print("Affichage des résultats de toutes les itérations")
for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print(f"Le meilleur résultat: {optimizer.max}")

file_BO = os.path.join(bayes_result,"logs.json")
try:
  logger = JSONLogger(path=file_BO)
  optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
except:
  print("Probleme pour relog les résultats")
optimizer.maximize(init_points=init_points, n_iter=n_iter,)
