# -*- coding: cp1252 -*-
#######################################################################
#                       Flags about behaviour of program
#----------------------------------------------------------------------
# LOAD_DATA_BY_VAR_AND_RESOL ... if True, reads data for specific variable/resolution
#         instead of loading the huge R01 resolution 4 vars numpy array.
#
# CALENDAR_FROM_DATA ... if True, reads coords files to extract 'time' variable
#         for calendar, or in case of LOAD_DATA_BY_VAR_AND_RESOL False, builds-it
#         using pd.date_range(). Calendar is thus dtype='datetime64[ns]'.
#
#######################################################################
#RESAC_WITH_NOISE = True #x_train,valid sont composés de données satellites (=bruitées) et y_train/valid sont les données NATL60
RESAC_WITH_NOISE = False
#LOAD_DATA_BY_VAR_AND_RESOL = False
#----------------------------------------------------------------------
LOAD_DATA_BY_VAR_AND_RESOL = True
#----------------------------------------------------------------------
#CALENDAR_FROM_DATA = False
CALENDAR_FROM_DATA = True
#----------------------------------------------------------------------
# Activer (True) si KERAS n'est pas installe en tant que module independant (dans HAL, le cluster GPU):
KERASBYTENSORFLOW=True
#KERASBYTENSORFLOW=False
#----------------------------------------------------------------------
# POUR utiliser le backend PlaidML pour cartes non NVidia (et obligatoirement pas tensorflow.keras, mais keras):
PLAIDMLKERASBACKEND=False
#PLAIDMLKERASBACKEND=True
# si vrai forcer le backend avec:
#    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#----------------------------------------------------------------------

import os, datetime

#######################################################################
#                       Extraction of meta-data
#######################################################################
# the date and hour now ! (execution time at begining)
nowstr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# name of current computer
current_hostname = os.uname()[1].split('.')[0]  
#
#######################################################################
#                       About THE DATAS
#######################################################################
#----------------------------------------------------------------------
# R�partition al�atoire des donn�es entre APP, VAL et TEST (en %)
pcentSet = pcentA, pcentV, pcentT = (0.7, 0.15, 0.15);  # (256,55,55)
#
# Nombre de lignes et colonnes EN DURE de la r�solution R01, (si on changeait
# de fichier de donn�es, il faudra peut �tre adapter ces valeurs). Utilis�es
#NLigR01, NColR01 = (1296, 1377); # pour les coordonn�es et autres fonctions
# en particulier pour d�duire la r�solution : reso_=NligR01/Nlig.
#
# Pour le calcul de l'Enstrophie :
# Delta x et y de R01 (donn�es initiales � la r�solution la plus grande) :
dxR01, dyR01 = (1500, 1500); # "Carlos arrondis"
#
#======================================================================
# Param�tre de Visualisation (1)
# . Some stuff
import matplotlib as mpl #see: ../matplotlib/rcsetup.py
mpl.rcParams['figure.facecolor'] = 'w';
mpl.rcParams['figure.figsize']   = [12, 9];
x_figtitlesize  = 14;
x_figsuptitsize = 14;
from  matplotlib import cm;
#
# . D�finition des norms et �chelles d'affichage des variables
tvwmm  = ["SSH",  "SST",  "U",     "V",    "NRJ",    "ENS",     "ADD", "MUL"];
wnorm  = [ None,  None,   None,   None,    None,     None,      None,  None];
wbmin  = [-0.67,  -2.30,  -2.25,  -2.50, 4.0849e-10, 3.868e-25,  0.0,   0.0 ];
wbmax  = [ 1.36,  30.00,   2.60,   2.35,   3.52,     1.158e-09, 3.52,  1.34e-09];
#wdmin = [-0.27,  None,   None,   None,   -9.00,    -24.00]; # Bornes vmin, vmax pour les ...
#wdmax = [ 0.36,  None,   None,   None,    9.20,     20.00]; # ... diff�rences EXP1, EXP2
wdmin  = [-0.16,  None,   None,   None,   -7.50,    -20.00]; # Bornes satur�es vmin, vmax pour les ...
wdmax  = [ 0.10,  None,   None,   None,    7.50,     17.00]; # ... diff�rences EXP1, EXP2
         # "SSH",    "SST",  "U",       "V",
whdmin = [-0.065684, None, -0.258711, -0.256959]; # Pour harmonisation des Histogrammes des DIFF ...
whdmax = [ 0.065785, None,  0.255844,  0.254520]; # ... entre EXP1 et EXP2 ...
whnmax = [   120000, None,    120000,    120000]; # ... � +|- 3 ecarts types d'EXP1
whNmax = [       33, None,         9,         9]; # Pour Histogrammes Normalis�s
whSmax = [       43, None,        12,        12]; # Pour Histogrammes Normalis�s cas Nord et Sud
         #   "SSH",     "SST",    "U",       "V",     "NRJ",     "ENS" : map de couleur par variable
wcmap = [cm.jet, cm.jet, cm.jet, cm.jet, cm.jet, cm.jet];
#
# . S�lection de quelques images � afficher par leur indices dans leur ensemble respectif
# (256APP, 55VAL, 55TEST), sachant qu'ils ont �t� randomis�s
im2showA  = [0];  # pour APP
im2showT  = [24]; # Pour Test (Art - Pres)
                  # 0: 18 aout 2013 ; 28: 23 juin 2013 ; 24: 1er septembre 2013
im2showV  = [16]; # Avec le random, 16 de Valid est la 1�re image des donn�es [0] correspondant au 1.10.2012
#
# . Some Flag de stat (et histo and other) des donn�es brutes
FLAG_STAT_BASE_BRUTE     = 0;  # Statistiques de base sur les variables brutes lues (->draft, ppt)
LIMLAT_NORDSUD           = 0;  # 0; 35 # Latitude limite de s�paration Nord-Sud
FLAG_STAT_BASE_BRUTE_SET = 0;  # Statistiques de base des variables brutes par ensemble (->draft, ppt)
FLAG_HISTO_VAR_BRUTE_SET = 0;  # Histogramme par variable et par ensemble (->draft, ppt)
NBINS                    = 45; # Default value du nombre de nbins des Histogrammes
FLAG_DAILY_MOY_EE        = 0;  # Figures des Moyennes Journali�res par pixel pour l'Energie et l'Enstrophie
                               # (need to have U and V as output)
SIGT_NOISE               = 0.0;# Sigma pour le bruitage de l'ensemble de TEST (seulement)
#
# . Flag de visulisation (figure) d'images par ensemble (en tenant compte de wnorm, wbmin, wbmax)
# 1 : toutes les images de l'ensemble pour chacune des variables en entr�e/sortie (!time et mc consuming)
# 2 : selon im2show*,
# 3 : = 1 + 2;
#  App,  Test,   Valid,   stop (1 : arret apr�s l'�dition des figures, sinon y'en a plein partout)
VisuBA, VisuBT, VisuBV, VisuBstop = \
(  0,      0,      0,      0); # 0: pas de figure;
                               # 1: images pour toutes les journ�es (def showimgdata) (non recommand�)
                               # 2: figures que pour les indices de im2show*
                               # 3: = 1 + 2 (non recommand�)
SUBP_ORIENT = "horiz";   # "horiz"; # else Vertical
CMAP_DEF    = cm.jet;    # Map de couleur par defaut (jet, Accent, ...)
ORIGINE     = 'lower';   # 'upper'; !!!quiver
#
# . Param�tres pour les plots des vecteurs UV avec quiver selon la r�solution
quvreso  = [1,   3,  9, 27, 81]; # Les r�solutions
quvmode  = [1,   1,  1,  1,  1]; # 1: 'step' ; sinon 'moyenne'
quvmask  = [18,  6,  2,  1,  1]; # Mask de moyenne ou de step des vecteurs
quvscale = [30, 30, 30, 30, 20]; # Taille (inverse) des fl�ches
#
#----------------------------------------------------------------------
#######################################################################
#                       About THE ARCHITECTURE
#######################################################################
#----------------------------------------------------------------------
# Indicateurs d'ajout d'un co�t d'Enstrophie ou op�rateur arithm�tique(+,*)
# � positionner selon les sorties de l'architecture ci apr�s.
#�lossENS=-1; lossADD=-1; lossMULT=-1; # initialement on ajoute rien
acide = 0;  # seed plac� pour l'init de l'architecture
# See the architecture coding in resac.py for the others parameters
# of the architecture (init, factiv, factout, upfactor).
# Ps: loss='logcosh' and optimizer='adam' are in DURE in the code
#
#----------------------------------------------------------------------


######################################################################################################################
#######################################################################################################################
SCENARCHI = 10; ############## VALEUR A MODIFIER EN FONCTION DES USAGES ###################################
######################################################################################################################"""
######################################################################################



# Choisir fichiers et variables (among ['SSH', 'SST', 'U', 'V'])
# les OUT cibles est la derni�re sont d�finies par targetout.
# targetout = # list des indexes (� partir de 0) des variables cibles
              # en sortie, (celles pour lesquelles ont veut des figures ;

 # Pour produire des figures � diff�rentes r�solutions
    # figures des variables
    #ResoIn = [ 81,    27,   9,    3,    1];    ResoOut = [ 81 ];
    #varIn  = ["SSH","SSH","SSH","SSH","SSH"];  varOut  = ["SSH"];
    #varIn  = ["SST","SST","SST","SST","SST"];  varOut  = ["SST"];
    #varIn  = ["U",  "U",  "U",  "U",  "U"  ];  varOut  = ["U"  ];
    #varIn  = ["V",  "V",  "V",  "V",  "V"  ];  varOut  = ["V"  ];

    #figure Daily mean Energy et Enstrophy FLAG_DAILY_MOY_EE = 1
    #varIn  = ["SSH"];  varOut  = ["SSH","SSH","U","V"]; # (need to have u
    #ResoIn = [ 81];    ResoOut = [  27,   9,    9,    9];   # and v as output)

    # figures des vecteurs
    #varIn  = ["SSH"];                          varOut  = ["SSH",  "U",   "V" ];
    #ResoIn = [ 81];                            ResoOut = [ 1,      1,     1  ];
    #ResoIn = [ 81];                            ResoOut = [ 9,      9,     9  ];

    # other (test du code, ...)
if SCENARCHI == 1 : #RESAC_inf (sans SST) 'kmodel/kmodel_ktd3_366';
    varIn     = ["SSH","SST"];                    varOut  = ["SSH", "U",  "V"];
    ResoIn    = [  81,27];                     ResoOut = [  27,   27,      27];
    targetout = [1, 2, 3];
elif SCENARCHI == 2 : #RESAC (avec SST) #'kmodel/kmodel_ktd10_366'; 402113w
    varIn  = ["SSH","SST","SST"];           varOut  = ["SSH","SSH", "U",  "V"];
    ResoIn = [  81,  27,    9 ];            ResoOut = [  27,   9,    9,    9];
    targetout = [1, 2, 3];

elif SCENARCHI == 3 : #from R27 to R09 
    varIn  = ["SSH","SST"];           varOut  = ["SSH", "U",  "V"];
    ResoIn = [ 27,    9 ];            ResoOut = [  9,    9,    9];
    targetout = [0, 1, 2];

elif SCENARCHI == 4 : #from R27 to R09 
    varIn  = ["SSH","SST"];           varOut  = ["SSH"];
    ResoIn = [ 9,    3 ];            ResoOut = [  3];
    targetout = [0];

elif SCENARCHI == 5 : #from R27 to R09 
    varIn  = ["SSH","SST","SST","SST"];           varOut  = ["SSH","SSH","SSH", "U", "V"];
    ResoIn = [  81,  27,  9,   3 ];            ResoOut = [ 27, 9, 3, 3, 3 ];
    targetout = [2,3,4];

elif SCENARCHI == 6 : #from R27 to R09 
    varIn  = ["SSH","SST","SST"];           varOut  = ["SSH"];
    ResoIn = [  81,  27,  9 ];            ResoOut = [  9 ];
    targetout = [0];
elif SCENARCHI == 7 : #from R27 to R09 
    varIn  = ["SSH","SST","SST","SST"];           varOut  = ["SSH","SSH","SSH"];
    ResoIn = [  81,  27,  9,   3 ];            ResoOut = [ 27, 9, 3 ];
    targetout = [2];
elif SCENARCHI == 8 : #from R27 to R09 
    varIn  = ["SSH","SST"];           varOut  = ["SSH"];
    ResoIn = [  9,   3 ];            ResoOut = [  3 ];
    targetout = [0];
elif SCENARCHI == 9 : #from R27 to R09 
    varIn  = ["SSH","SST","SST"];           varOut  = ["SSH","SSH"];
    ResoIn = [  9,   3, 1 ];            ResoOut = [  3, 1 ];
    targetout = [0,1];
elif SCENARCHI == 10 : #from R27 to R09 
    varIn  = ["SSH","SST"];           varOut  = ["SSH"];
    ResoIn = [  81,   27 ];            ResoOut = [ 27 ];
    targetout = [0];
else :

    raise ValueError("Create another archi");
#
#----------------------------------------------------------------------
#######################################################################
#           SET RUN EXPERIMENT CONFIGURATION (and other ...)
#######################################################################
#----------------------------------------------------------------------
TEST_ON  = True; # True ou False : usage ou pas d'un ens de test
VALID_ON = 4;    # 0 : Pas d'usage de l'ensemble de validation (run simple)
                 # 1 : Usage de l'ensemble de validation (sans early stopping)
                 # 2 : Caduc old usage
                 # 3 : Early stopping sur l'ensemle de validation. Les poids ou
                 #     le mod�le sauvgardable sont obtenus � la fin du run
                 #     apr�s patience it sans am�lioration).
                 # 4 : Le run va jusqu'au bout. On r�cup�re, par la suite la
                 #     sauvegarde des poids au meilleur de l'ensemble de
                 #     validation pour les r�sultats.
if (TEST_ON and pcentT<=0.0) or (VALID_ON and pcentV<=0.0) :
    raise ValueError("Your pcentT or pcentV is not OK");
#%%
Mdl2dirname = None     # RUN_MODE=="LEARN": repertoire de base pour l'apprentissage
Mdl2savedcase = None  # RUN_MODE=="RESUME": repertoire de base de l'apprentissage pour Test
factoutlbl4train = None
Mdl2reprendre = None
#
if SCENARCHI >= 0 :
    RUN_MODE = "LEARN"
    #RUN_MODE = "REPRENDRE"
    #RUN_MODE = "RESUME"

    ###################################################################
    #"LEARN" : On effectue l'entrainement du reseau ou l'apprentissage
    #          (model fit).
    #
    # "RESUME": Pas d'entrainement. Effectue seulement un test avec les
    #           donnees de Test et affiche diverses figures.  Recharge
    #           les poids d'un modele deja entraine.
    #
    # "REPRENDRE": Reprise de l'apprentissage a partir d'un checkpoint
    #              avec les poids du modele deja entraine.
    # ##################################################################

    # Nom de fichier pour les poids d'un modele
    if RUN_MODE == "LEARN" :
        # Nom du du fichier � sauvegarder
        Mdl2dirname = 'Save_Model'; # Nom du repertoire des cas, par defaut
        Mdl2name = 'kmodel'; # Nom du cas
        #
        # fonction d'activation des sorties:
        #  - fonctions standards dans keras: 'relu', 'tanh', 'sigmoid', 'linear', or
        #  - ou bien celles definies dans recasart.py: 'sig01', 'sig17'
        factoutlbl4train = 'sig01' #
        #factoutlbl4train = 'sig17' #
        #factoutlbl4train = 'relu' #
        #factoutlbl4train = 'linear' #
        #factoutlbl4train = 'swish'
        #
    elif RUN_MODE == "RESUME" :
        # Nom du du fichier � recharger
        if SCENARCHI == 1 :   # RESAC sans SST
            Mdl2reload = 'kmodel/kmodel_ktd3_366';
        elif SCENARCHI == 2 : # RESAC avec SST
            Mdl2olddirname = 'Save_Model'; # Nom du repertoire ou se troube le cas a reprendre
            #---Mdl2savedcase--------------------------------------------------
            # Ecrire ici le nom du cas d'apprentissage a etre utilise pour Tests
            #Mdl2savedcase = 'Net_ICARE_E3-BS15_20210324-165410'  # Tf1, 30 epochs
            #Mdl2savedcase = 'Net_HAL1_E100-BS29_20210324-171057' # 100 epochs
            #Mdl2savedcase = 'Net_HAL4_E7200-BS29_20210324-175434_sig01' # 7200 epochs, sig01
            #Mdl2savedcase = 'Net_HAL4_E7200-BS29_20210324-232654_relu' # 7200 epochs, relu
            #Mdl2savedcase = 'Net_HAL4_E7200-BS29_20210325-074839_linear' # 7200 epochs, linear
            #Mdl2savedcase = 'Net_HAL4_E7200-BS29_20210325-211257_relu' # 7200 epochs, relu (2nd)
            Mdl2savedcase = 'Net_HAL4_E7200-BS29_20210325-233551_linear' # 7200 epochs, linear (2nd)
            #Mdl2savedcase = 'Net_HAL1_E7200-BS29_20210328-214815_sig17' # 7200 epochs, sig17
            #Mdl2savedcase = 'Net_SAMOTHRACE_E100-BS29_20210415-173832_LINEAR'
            #------------------------------------------------------------------
            Mdl2resumedir = os.path.join('RESUME',Mdl2savedcase)
        else :
            print(f"Scenario {SCENARCHI} pas encore prevu") # Nom par d�faut, suppose avoir �t� cr�e avant
            Mdl2reload = 'Save_Model/modelk'; # Nom par d�faut, suppose avoir �t� cr�e avant
        #
    elif RUN_MODE == "REPRENDRE":
        if SCENARCHI == 1: #RESAC sans SST
            print("Ce scenario n'est pas prévu")
        elif SCENARCHI == 2: #RESAC AVEC SST 81 to 09
            Mdl2olddirname = 'Save_Model'; # Nom du repertoire ou se troube le cas a reprendre
            Mdl2savedcase = 'Net_SAMOTHRACE_E100-BS29_20210415-173832_LINEAR'
            Mdl2savedcase = 'Net_kmodel_ICARE_E5-BS15_20210425-181327_SIG01'
            #------------------------------------------------------------------
            Mdl2reprendredir = os.path.join('REPRENDRE',Mdl2savedcase)
    else :
        raise ValueError("Bad RUN MODE");
    #
    #f RUN_MODE == "LEARN" and Mdl2save != "Save_Model/" :
    #    _rep = input("Are you shure ? : ");
    #    if _rep != 1 :
    #        raise ValueError("Are you not shure hein !");
    #
    #Niter, Bsize = (5100, 29);
    #Niter, Bsize = (7200, 29);
    #Niter, Bsize = (10000, 29);
    #Niter, Bsize = (500, 29);
    #Niter, Bsize = (100, 29);
    #Niter, Bsize  = (1900, 32);
    #init_points, n_iter= (10,60)
    #Niter, Bsize  = (10, 29);
    #init_points, n_iter= (8,48)
    init_points, n_iter= (1,1)
    Niter, Bsize=(20, 32)
    #
    #======================================================================
    # ajoute au nom du dossier de sauvegarde un sous-dossier compose de
    # 'Net_' suivie de:
    #  - le nom de l'ordinateur
    #  - le nombre d'epochs
    #  - le batch size
    #
    #======================================================================
    # Param�tre de Visualisation (2) des r�sultats
    # .Flags pour l'�dition et la visualisation des r�sultats sur les donn�es brutes
    # RMS: rms ; SCAT: scatterplot ; HISTO: histogramme ;
    # Pour vecteur UV : COSIM: cosine similarity  ; NRJ: energie ;
    #                   ENS: enstrophie ; CORRPLEX: corr�lation complexe ;
    # - Sur l'ensemble d'APP
    SAVEFIG=True
    FLGA_RES = 0; # Resultats s/donn�es brutes (Conditionne les flags suivants)
    FLGA_RMS, FLGA_SCAT, FLGA_HISTO = \
    (   1,        1,         1);                # pour result
    FLGA_COSIM, FLGA_NRJ, FLGA_ENS, FLGA_CORRPLEX, FLGA_DIV = \
    (   0,         0,        0,            0,          0);  # flag pour esultuv
    # - Sur l'ensemble de TEST
    FLGT_RES = 0; # Result s/donn�es brutes (Conditionne les flags suivants)
    FLGT_RMS, FLGT_SCAT, FLGT_HISTO = \
    (   1,        1,        1);                # pour result
    FLGT_COSIM, FLGT_NRJ, FLGT_ENS, FLGT_CORRPLEX, FLGT_DIV = \
    (   0,         0,        0,           0,          0);  # pour resultuv

    SCATALPHA  = 0.20             # controlling transparency for scatter plots
    SCATLABELS = ["ident", "reg.lin", "scatter"]  # to control labels and order in legend for scatterplots
    #
    # Des figures ou R�sultats qu'on peut vouloir sauter.
    FIGBYPASS = True   # figure de type imshow (showsome, ...)
    FIGBYIMGS = False  # figure de r�sultat par image (type courbe rms, KL, ...)
    RESBYPASS = False  # sauter les r�sultats par variable de sortie (targetout)
    #
    #=====================================================================
    #
