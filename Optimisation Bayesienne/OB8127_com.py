from resacartparm import *
from resacartdef import *

############################!!!!!!!!!!!!!!!!!!!!!!##################################
#Modifier SCENARCHI = 1 dans resacartparm.py
#Scénario ayant comme entrée ["SSH"81,"SST"27] et en sortie ["SSH"27,"U"27,"V"27]
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
##############################################################################OPTIMISATION BAYESIENNE#############################################################################################"""""

from functools import partial
from bayes_opt import BayesianOptimization  #Biblio pour l'optimisation bayesienne
from bayes_opt.logger import JSONLogger     
from bayes_opt.event import Events
from bayes_opt.util import load_logs        #Pour charger les données à l'issu de l'optimisation 
from tensorflow.nn import swish

#Définition du model avec en arguments les hyper-paramètres DU MODEL à optimiser avec l'Optim. Bay.
def my_model(Nconv1, Nfilt1, function):
  ArchiOut = []
  upfactor = 3
  factiv= swish     #relu, sig01
  init       = 'he_normal'   #'glorot_normal'+ #'orthogonal' #'normal'
  factout  = 'sigmoid'

  if IS_SSTR27: # # SSH_R81 + SST_R09 to SSH_R27
    ISSHreso = all_Kinput_img[0] # Input SSH reso 81
    ArchiA   = UpSampling2D((upfactor, upfactor), interpolation='bilinear')(ISSHreso)
    #ArchiA   = concatenate([ArchiA, ISSTR27], axis=1) # CHANNEL FIRST
    ArchiA   = concatenate([ArchiA, ISSTR27], axis=3) # CHANNEL LAST
    for i in np.arange(Nconv1) : #7
      #ArchiA = Conv2D(36,(6,6), activation=factiv, padding='same',kernel_initializer=init)(ArchiA)
      ArchiA = Conv2D(Nfilt1,(3,3), activation=factiv, padding='same',kernel_initializer=init)(ArchiA)
      ArchiA = Conv2D(Nfilt1,(3,3), activation=factiv, padding='same',kernel_initializer=init)(ArchiA)
      ArchiA = BatchNormalization(axis=3)(ArchiA)
    ArchiA = Conv2D(8,(3,3), activation=factiv, padding='same',kernel_initializer=init)(ArchiA)
  if function < 1:
    ArchiA = BatchNormalization(axis=3)(ArchiA)
    factout = 'linear' 
  elif 1 < function < 2 :
    factout = 'sig01'
  else : 
    factout = 'sigmoid'
  ArchiA = Conv2D(1,(1,1), activation=factout, padding='same',kernel_initializer=init,name='archiA')(ArchiA)
  ArchiOut.append(ArchiA);

  nc, ml, mc, nl = (16, 5, 5, 1);
  if IS_SSTR09 : # SST R09 utilisï¿½ pour trouver U R09 et V R09
      #ArchiUV  = concatenate([ArchiB, ISSTR09], axis=1) # CHANNEL FIRST
    ArchiUV  = concatenate([ArchiA, ISSTR27], axis=3) # CHANNEL LAST
    ArchiUV  = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiUV) ##
  else :
    ArchiUV = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiA) ##
  for i in np.arange(nl) :
    ArchiUV = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiUV)
    ArchiUV = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiUV)
    ArchiUV = BatchNormalization(axis=3)(ArchiUV)
  ArchiUV = Conv2D(nc/2,(ml, mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiUV)
  ArchiUV = Conv2D(1,(1, 1), activation=factout, padding='same',kernel_initializer=init)(ArchiUV)
  #
  ArchiU = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiUV)
  for i in np.arange(nl) :
    ArchiU = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiU)
    ArchiU = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiU)
    ArchiU = BatchNormalization(axis=3)(ArchiU)
  ArchiU = Conv2D(nc/2,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiU)
  ArchiU = Conv2D(1,(1, 1), activation=factout, padding='same',kernel_initializer=init,name='archiU')(ArchiU)
  ArchiOut.append(ArchiU)
  #
  ArchiV = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiUV)
  for i in np.arange(nl) :
    ArchiV = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiV)
    ArchiV = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiV)
    ArchiV = BatchNormalization(axis=3)(ArchiV)
  ArchiV = Conv2D(nc/2,(ml, mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiV)
  ArchiV = Conv2D(1,(1, 1), activation=factout, padding='same',kernel_initializer=init,name='archiV')(ArchiV)
  ArchiOut.append(ArchiV)
  print(factout)
  return Model(all_Kinput_img[0:2], ArchiOut)

#Fonction à fit avec l'optimisation bayesienne avec en arguments TOUS les hyper-paramètres à optimiser
def fit_with(Nconv1,  Nfilt1, function, lr):

  model = my_model(Nconv1,  Nfilt1,function)

    # Train the model for a specified number of epochs.
  optimizer = optimizers.Adam(learning_rate=2*(10**lr))
  model.compile(loss='logcosh',
                optimizer=optimizer)

# Train the model with the train dataset.
  H = model.fit(x_train[0:2], y_train[0], epochs=Niter,batch_size=Bsize, 
            shuffle=True, verbose=2, validation_data=(x_valid[0:2], y_valid)[0])

  return  -min(H.history['loss'])   #Mesure référence des performances pour l'optimisation bayesienne
fit_with_partial = partial(fit_with)


#L'opimisation bayesienne est un mélange de méthodes d'exploration (changement radical de la zone d'étude en modifiant drastiquement 
#la combinaison d'hyperparamètres) et d'exploitation (ajustement autour des valeurs des hyperparamètres actuels)

#Définition des dossiers pour sauver les résultats de l'optimisation 
folder_BO = f"bayesianOpt_InitP-{init_points}_NIter-{n_iter}" #init_points : nombre d'explorations ; n_iter : nombre d'exploitation dans une zone
#Le nombre d'itérations total = n_iter + init_points (Une itération = réalisation de N_epochs sur un model avec certains hyperparamètres)
bayes_result = os.path.join(Mdl2save,folder_BO )
os.makedirs(bayes_result,   exist_ok = True)




# Définition des intervalles des différents hyperparamètres (Pour les variables entières tel que le nb de filtre dans une Conv2D, le réel est arrondi à l'entier supérieur)
pbounds = {'Nconv1': (1, 5), 'Nfilt1': (2,32),  'lr': (-4.5,-1) ,'function' :(0,3)} #function 0-1: linear, 1-2: sig01, 2-3: sigmoid
optimizer = BayesianOptimization(
    f=fit_with_partial,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 relève uniquement quand un maximum est observé,  verbose = 0 RIEN
    
)

#Relog le fichier à la fin de l'optimisation et affiche les meilleurs résultats triés par performance
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

