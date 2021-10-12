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
        #data_tmp = np.load(os.path.join(datasets_dir,subdir,f"NATL60_{v.upper()}_R{r:02d}.npy"))
        #dimension_tmp = np.load(os.path.join(datasets_dir,subdir,f"NATL60_coords_R{r:02d}.npz"))
        data_tmp = np.load(os.path.join(datasets_dir,subdir,f"SAT_{v.upper()}_R{r:02d}s.npy"))
        dimension_tmp = np.load(os.path.join(datasets_dir,subdir,f"SAT_coords_R{r:02d}s.npz"))
        dico_dim = { 'time': dimension_tmp['time'],
                     'lat' : dimension_tmp['latitude'],
                     'lon' : dimension_tmp['longitude'],
                     'lat_border' : dimension_tmp['latitude_border'],
                     'lon_border' : dimension_tmp['longitude_border'] }
        V_data_list.append(data_tmp)
        D_dico_list.append(dico_dim)
    #
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
    V_data_list, couple_var_reso_list, D_dico_list = load_resac_by_var_and_resol(varIn,varOut,ResoIn,ResoOut, subdir='Satellite/SatbyVar')# Enlever subdir si données NATL60 classique desirées
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
# NATL60 : Moyennes Journaliï¿½res des pixels pour l'energie et
# l'enstrophie ï¿½ la rï¿½solution de sortie (R09)
if FLAG_DAILY_MOY_EE and IS_UVout :
    iu_ = varOut.index("U")
    iv_ = varOut.index("V")
    Uall_ = np.concatenate((VAout_brute[iu_],VVout_brute[iu_],VTout_brute[iu_]))
    Vall_ = np.concatenate((VAout_brute[iv_],VVout_brute[iv_],VTout_brute[iv_]))
    NI_, NL_, NC_ = np.shape(Uall_)
    NL_lim_ = nl2limlat(NL_, LIMLAT_NORDSUD) # Nombre de ligne jusqu'ï¿½ la latitude limite Nord-Sud
    #1) Energie
    NRJ_    = (Uall_**2 + Vall_**2) / 2
    moyNRJ_ = np.mean(NRJ_, axis=0)
    plt.figure()
    plotavar(moyNRJ_, "", None, None, None)
    plt.title("NATL60: Daily Mean of Pixels Energy (at Rout=R09)")
    # Pourcentatge au Nord et au Sud d'une Latitude donnï¿½e
    XN_, XS_ = splitns(NRJ_, NL_lim_)
    sumNRJ_  = np.sum(NRJ_)
    sumNRJS_ = np.sum(XS_)
    sumNRJN_ = np.sum(XN_)
    pcentS_  = sumNRJS_ / sumNRJ_
    pcentN_  = sumNRJN_ /sumNRJ_
    print("Pcent Energy NATL60 (at Rout=R09): Nord=%.4f  Sud=%.4f  (sum pcent = %.4f (should be 1.)"
          %(pcentN_, pcentS_, pcentN_+pcentS_))
    del NRJ_, moyNRJ_, XN_, XS_, sumNRJ_, sumNRJS_, sumNRJN_
    #
    #2) Enstrophie
    ENS_    = enstrophie2d(Uall_, Vall_, dxRout, dyRout)
    moyENS_ = np.mean(ENS_, axis=0)
    plt.figure()
    plotavar(moyENS_, "", None, None, None)
    plt.title("NATL60: Daily Mean of Pixels Enstrophy (at Rout=R09)")
    # Pourcentatge au Nord et au Sud d'une Latitude donnï¿½e
    XN_, XS_ = splitns(ENS_, NL_lim_)
    sumENS_  = np.sum(ENS_)
    sumENSS_ = np.sum(XS_)
    sumENSN_ = np.sum(XN_)
    pcentS_  = sumENSS_ / sumENS_
    pcentN_  = sumENSN_ /sumENS_
    print("Pcent Enstrophy NATL60 (at Rout=R09) : Nord=%.4f  Sud=%.4f  (sum pcent = %.4f (should be 1.)"
          %(pcentN_, pcentS_, pcentN_+pcentS_))
    del ENS_, moyENS_, XN_, XS_, sumENS_, sumENSS_, sumENSN_
    del Uall_, Vall_
#%%
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
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
#
#======================================================================
# Ajout de bruit sur l'input SSH brute pour l'ensemble de TEST uniquement
if SIGT_NOISE > 0 :
    for i in np.arange(NvarIn) :
        if varIn[i] == "SSH" :
            STAT_ON_NOISE = True
            if STAT_ON_NOISE :
                X0_ = VTin_brute[i]  # H avant bruitage
            #
            np.random.seed(0)
            N_,M_,L_,C_ = np.shape(VTin_brute[i])
            #sigma = 0.050 #1.=8ï¿½, 10=42ï¿½, 100=45ï¿½(0ï¿½-180ï¿½)
            noise_ = np.random.randn(N_,M_,L_,C_) * SIGT_NOISE
            VTin_brute[i] = VTin_brute[i] + noise_     # TEST set only
            #
            if STAT_ON_NOISE :
                noiseabs_ = np.abs(noise_)
                print("SSH : noise %.3f (min=%.3f, max=%.3f, moy=%.4e, std=%.3f moyabs=%.4e, stdabs=%.3f)"
                      %(SIGT_NOISE, np.min(noise_), np.max(noise_), np.mean(noise_),
                        np.std(noise_), np.mean(noiseabs_), np.std(noiseabs_)))
                rmsi, Nnan, inan = nanrms(VTin_brute[i], X0_)
                print("noiseRMSE : %f "%(rmsi))
                #
                print("noiseErrRelAbs : %f "%( np.sum(noiseabs_) / np.sum(np.abs(X0_))))
                print("Stat noiseAbs perturbation (sur H en m): min=%.4e, max=%.4f, moy=%.4f, std=%.4f"
                      %(np.min(noiseabs_), np.max(noiseabs_),np.mean(noiseabs_), np.std(noiseabs_)))
                del noiseabs_, X0_
            del noise_
#======================================================================
if VisuBA+VisuBV+VisuBT > 0 :
    print("# Visualisation des donnees brutes")

#%%
#----------------------------------------------------------------------
def visuB (X_brute, varIO, Resolst, D_dicolst, VisuB, Ndon, strset, inout, im2show,
           qmask=None, qscale=None, qmode=None, calX0=None, 
           fsizeimg=None, fsizesome=None, fsizehquiv=None,
           figdir='.', savefig=False) :
    Nvar = len(varIO)
    calX0_= None
    if calX0 is not None :
        calX0_ = calX0[im2show]
    if (VisuB==1 or VisuB==3) :
        for i in np.arange(Nvar) :
            wk_ = tvwmm.index(varIO[i])
            showimgdata(X_brute[i],cmap=wcmap[wk_], n=Ndon,fr=0, vmin=wbmin[wk_],
                        vmax=wbmax[wk_], vnorm=wnorm[wk_], origine='lower', fsize=fsizeimg)
            titre = "%s: %s_brute(%s%d) R%02d, min=%f, max=%f, mean=%f, std=%f"%(
                strset, varIO[i], inout, i+1, Resolst[i], np.min(X_brute[i]),np.max(X_brute[i]),
                np.mean(X_brute[i]),np.std(X_brute[i]))
            plt.suptitle(titre, fontsize=x_figtitlesize)
            if savefig :
                printfilename = strconv_for_title(titre)
                plt.savefig(os.path.join(figdir,f"{printfilename}.png"))

    if (VisuB==2 or VisuB==3) and len(im2show) > 0 :
        if inout=="in" or (inout=="out" and 1) :
            for i in np.arange(Nvar) :
                wk_ = tvwmm.index(varIO[i])
                suptitre="some %s brute(%s%d) %s R%02d, m%s %s"%(strset, inout,i+1, varIO[i], Resolst[i], im2show, calX0_)
                showsome(X_brute[i][im2show,0,:,:], Resolst[i], D_dicolst[i], wmin=wbmin[wk_], wmax=wbmax[wk_],
                         wnorm=wnorm[wk_], cmap=wcmap[wk_], fsize=fsizesome, calX0=calX0_,
                         varlib=varIO[i], suptitre=suptitre, Xtit=None,
                         figdir=figdir, savefig=savefig)

        if (VisuB==2 or VisuB==3) and len(im2show) > 0  and inout=="out" :
            if IS_HUVout and 1 : # Vecteurs UV on H
                ih_ = len(varIO) - 1 - varIO[::-1].index("SSH") # le dernier varbg (en charchant le premier de la liste inversée!)

                suptitre="some %s brute+UV(%s) %s R%02d, m%s %s"%(
                    strset, inout, varIO[ih_], Resolst[ih_], im2show, calX0_)
                ih_ = showhquiv(X_brute, Resolst, D_dicolst, im2show, qscale=qscale,
                                qmask=qmask, qmode=qmode, calX0=calX0, suptitre=suptitre,
                                fsize=fsizehquiv)
                if savefig :
                    printfilename = strconv_for_title(suptitre)
                    plt.savefig(os.path.join(figdir,f"{printfilename}.png"))
#--------              ResoIn Din_dico_list, ResoOut, Dout_dico_list,
#%%
#======================================================================
# Preparation des repertoires du cas
#======================================================================
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
        
elif RUN_MODE == "RESUME" :
    print("\nRESUME MODE ...");
    try:
        images_dir      = os.path.join(Mdl2resumedir,"Images")
        # Creation du repertoire de sauvegarde (pour RESUME, images, ...)
        os.makedirs(Mdl2resumedir,   exist_ok = True)
        os.makedirs(images_dir,      exist_ok = True)
        
        Mdl2reloadcase     = os.path.join(Mdl2olddirname,Mdl2savedcase)
        Mdl2reloadArchi    = os.path.join(Mdl2reloadcase,"Archi")
        Mdl2reloadWeights  = os.path.join(Mdl2reloadcase,'Weights','modelkvalid.ckpt')
        Mdl2reloadHistLoss = os.path.join(Mdl2reloadcase,'history.pkl')

    except:
        print(f"Unexpected error when creating '{RUN_MODE}' directories in {Mdl2resumedir}/", sys.exc_info()[0])
        raise

elif RUN_MODE == "REPRENDRE" :
    print("\nREPRENDRE MODE ...");
    try:
        #Mdl2reprendre = Mdl2reprendredir+f"_{current_hostname.upper()}_E{Niter}-BS{Bsize}_{nowstr}"
        Mdl2reprendre = Mdl2reprendredir
        print(f'Mdl2reprendre: "{Mdl2reprendre}"')
        if os.path.exists(os.path.join(Mdl2reprendre,"Archi","saved_model.pb")) :
            Mdl2reloadcase = Mdl2reprendre
            print(f" --> Reprendre from already created REPRENDRE case:\n     '{Mdl2reloadcase}'")
        else:
            Mdl2reloadcase = os.path.join(Mdl2olddirname,Mdl2savedcase)
            print(f" --> Reprendre from trained LEARN case:\n     '{Mdl2reloadcase}'")

        Mdl2reloadArchi    = os.path.join(Mdl2reloadcase,"Archi")
        Mdl2reloadWeights  = os.path.join(Mdl2reloadcase,'Weights','modelkvalid.ckpt')
        Mdl2reloadHistLoss = os.path.join(Mdl2reloadcase,'history.pkl')

        historique_dir  = os.path.join(Mdl2reprendre,"Historique_Loss")
        archi_train_dir = os.path.join(Mdl2reprendre,"Archi")
        weights_dir     = os.path.join(Mdl2reprendre,"Weights")
        logs_fit_dir    = os.path.join(Mdl2reprendre,'logs','fit')
        images_dir      = os.path.join(Mdl2reprendre,"Images")
        
        # Creation du repertoire de sauvegarde du produit de l'apprentissage (modele, poids, historique, images)
        os.makedirs(Mdl2reprendre,   exist_ok = True)
        os.makedirs(historique_dir,  exist_ok = True)
        os.makedirs(archi_train_dir, exist_ok = True)
        os.makedirs(weights_dir,     exist_ok = True)
        os.makedirs(logs_fit_dir,    exist_ok = True)
        os.makedirs(images_dir,      exist_ok = True)

    except:
        print(f"Unexpected error when creating '{RUN_MODE}' directories in {Mdl2reprendredir}/", sys.exc_info()[0])
        raise
    
else : #=> Unknown RUN_MODE ...
    print(f"Unknown RUN_MODE {RUN_MODE}, should be one between [ 'LEARN', 'RESUME', 'REPRENDRE' ]")
    raise
#%%
if VisuBA > 0 :
    # visu des donnï¿½es brutes d'APP en ENTREE
    visuB(VAin_brute, varIn, ResoIn, Din_dico_list, VisuBA, NdonA, "APP","in", im2showA,
          calX0=calA[0], fsizeimg=(10,12), fsizesome=(10,12), fsizehquiv=(10,12),
          figdir=images_dir, savefig=SAVEFIG)
    # Visu des donnï¿½es brutes d'APP en SORTIE.
    visuB(VAout_brute, varOut, ResoOut, Dout_dico_list, VisuBA, NdonA, "APP","out", im2showA,
          calX0=calA[0], fsizeimg=(10,12), fsizesome=(10,12), fsizehquiv=(10,12),
          figdir=images_dir, savefig=SAVEFIG)
if VisuBT > 0 and TEST_ON :
    # visu des donnï¿½es brutes de TEST en entrï¿½e
    visuB(VTin_brute, varIn, ResoIn, Din_dico_list, VisuBT, NdonT, "TEST","in", im2showT,
          calX0=calT[0], fsizeimg=(10,12), fsizesome=(10,12), fsizehquiv=(10,12),
          figdir=images_dir, savefig=SAVEFIG)
    # Visu des donnï¿½es brutes de TEST en sortie.
    visuB(VTout_brute, varOut, ResoOut, Dout_dico_list, VisuBT, NdonT, "TEST","out", im2showT,
          calX0=calT[0], fsizeimg=(10,12), fsizesome=(10,12), fsizehquiv=(10,12),
          figdir=images_dir, savefig=SAVEFIG)
if VisuBV > 0 and VALID_ON :
    # visu des donnï¿½es brutes de VALIDATION en entrï¿½e
    visuB(VVin_brute, varIn, ResoIn, Din_dico_list, VisuBV, NdonV, "VALID","in", im2showV,
          calX0=calV[0], fsizeimg=(10,12), fsizesome=(10,12), fsizehquiv=(10,12),
          figdir=images_dir, savefig=SAVEFIG)
    # Visu des donnï¿½es brutes de VALIDATION en sortie.
    visuB(VVout_brute, varOut, ResoOut, Dout_dico_list, VisuBV, NdonV, "VALID","out", im2showV,
          calX0=calV[0], fsizeimg=(10,12), fsizesome=(10,12), fsizehquiv=(10,12),
          figdir=images_dir, savefig=SAVEFIG)
if VisuBstop or SCENARCHI==0 :
    print("STOP after visu donnï¿½es brutes")
    #plt.show()
    sys.exit(0)
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
#%%
#======================================================================
#######################################################################
#                       THE ARCHITECTURE
#######################################################################
#======================================================================
print("# Build and compile Architecture")
if KERASBYTENSORFLOW :
    from tensorflow.keras.layers    import Input #, Dense, Flatten, Reshape, AveragePooling2D, Dropout
    from tensorflow.keras.layers    import Conv2D, UpSampling2D, BatchNormalization #, Deconvolution2D, MaxPooling2D
    from tensorflow.keras.layers    import concatenate #, Concatenate
    from tensorflow.keras.models    import Model
    from tensorflow.keras.models    import load_model
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
    from tensorflow.keras.layers.experimental.preprocessing import Resizing
    from math import ceil
    from keras import regularizers, optimizers # Regularizer factout function
else:
    if PLAIDMLKERASBACKEND : # backend pour cartes graphiques non NVIDIA
        os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    #
    from keras.layers    import Input #, Dense, Flatten, Reshape, AveragePooling2D, Dropout
    from keras.layers    import Conv2D, UpSampling2D #, Deconvolution2D, MaxPooling2D
    from keras.layers    import concatenate #, Concatenate
    from keras.models    import Model
    from keras.models    import load_model


    #import keras.callbacks
    from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
#
#----------------------------------------------------------------------
if 1 : # Avoir sa propre fonction d'activation
    if KERASBYTENSORFLOW :
        from tensorflow.keras.layers import Activation
        from tensorflow.keras import backend as K
        #from keras.utils.generic_utils import get_custom_objects
        from tensorflow.keras.utils import get_custom_objects
    else:
        from keras.layers import Activation
        from keras import backend as K
        from keras.utils.generic_utils import get_custom_objects
    def sig010(x) : # sigmoï¿½d dans l'intervalle [-0.10, 1.10]
        return  (K.sigmoid(x) * 1.20) - 0.10
    def sig017(x) : # sigmoï¿½d dans l'intervalle [-0.35, 1.35]
        return  (K.sigmoid(x) * 1.70) - 0.35
    get_custom_objects().update({'sig01': Activation(sig010)})
    get_custom_objects().update({'sig17': Activation(sig017)})
#----------------------------------------------------------------------
# Dï¿½termination de la prise en compte (ou pas) de la SST en entrï¿½e
# de l'archi selon les rï¿½solutions indiquï¿½es (dans resacparm.py).
all_Kinput_img = []
IS_SSTR81 = IS_SSTR27 = IS_SSTR09 = IS_SSTR03 = False
for ii in np.arange(NvarIn) :
    #Ncan_, NL_, NC_ = np.shape(x_train[ii][0]) # CHANNEL FIRST
    #all_Kinput_img.append(Input(shape=(Ncan_,NL_,NC_)))
    NL_, NC_, Ncan_  = np.shape(x_train[ii][0]) # CHANNEL LAST
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
        else :
            raise ValueError("Other resolution of SST not prevue")
#%%
from tensorflow.nn import swish, gelu
if RUN_MODE=="LEARN":
    print("# Build and compile Architecture")
    # Paramï¿½tres par dï¿½faut (a re-adapter si besoin)
    np.random.seed(acide)      # Choose a random (or not) for reproductibilitie
    init       = 'he_normal'   #'glorot_normal'+ #'orthogonal' #'normal'
    factiv     = 'swish'        # 'relu', 'tanh', 'sigmoid', 'linear'
    # Fonctions d'activation des couches de sortie SSH (R27, R09, ...) et U et V
    factoutlbl = None
    if Mdl2savedcase is not None :
        # pour remettre la bonne fonction au reseau pour RESUME
        if Mdl2savedcase in [ 'Net_HAL4_E7200-BS29_20210324-175434' ] : # liste de cas 'sig01'
            factoutlbl = 'sig01'
        elif Mdl2savedcase in [ 'Net_HAL4_E7200-BS29_20210324-232654_relu',
                               'Net_HAL4_E7200-BS29_20210325-211257_relu']: # liste de cas 'relu'
            factoutlbl = 'relu'
        elif Mdl2savedcase in [ 'Net_HAL4_E7200-BS29_20210325-074839_linear',
                               'Net_HAL4_E7200-BS29_20210325-233551_linear' ]: # liste de cas 'linear'
            factoutlbl = 'linear'
        elif Mdl2savedcase in [ 'Net_HAL1_E7200-BS29_20210328-214815_sig17' ] : # liste de cas 'sig17'
            factoutlbl = 'sig17'
        else:
            factoutlbl = 'sig01'  # 'relu', 'tanh', 'sigmoid', 'linear'
        print(f"\nCas {RUN_MODE} '{Mdl2savedcase}':")
        print(f"fonctions d'activation des couches de sortie: '{factoutlbl}'")
    else:
        if factoutlbl4train is None:
            factoutlbl = 'sig01'       # 'relu', 'tanh', 'sigmoid', 'linear'
            #factoutlbl = 'sig17'       # 'relu', 'tanh', 'sigmoid', 'linear'
            #factoutlbl = 'relu'        # 'relu', 'tanh', 'sigmoid', 'linear'
            #factoutlbl = 'linear'      # 'relu', 'tanh', 'sigmoid', 'linear'
        else:
            factoutlbl = factoutlbl4train
        print(f"\nFonctions d'activation des couches de sortie: '{factoutlbl}'")
    factout = factoutlbl
    factout = 'sig01' #sig01
    upfactor   =  3            # facteur d'upsampling
    #
    ArchiOut   = []            # Liste des sorties de l'Archi
    #
    # 1er ï¿½tage : ... to SSH_R27
    if IS_SSTR81 : # SSH_R81 + SST_R81 to SSH_R27
        ISSHreso = all_Kinput_img[0] # Input SSH reso
        #ArchiA   = concatenate([ISSHreso, ISSTR81], axis=1) # CHANNEL FIRST
        ArchiA   = concatenate([ISSHreso, ISSTR81], axis=3) # CHANNEL LAST
        ArchiA   = UpSampling2D((upfactor, upfactor), interpolation='bilinear')(ArchiA)
        for i in np.arange(4) : #4
            
            ArchiA = Conv2D(32,(3,3), activation=factiv, padding='same',kernel_initializer=init)(ArchiA)
            ArchiA = Conv2D(32,(3,3), activation=factiv, padding='same',kernel_initializer=init)(ArchiA)
            ArchiA = BatchNormalization(axis=3)(ArchiA)
        ArchiA = Conv2D(8,(3,3), activation=factiv, padding='same',kernel_initializer=init)(ArchiA)
        ArchiA = Conv2D(1,(1,1), activation=factout, padding='same',kernel_initializer=init,name='archiA')(ArchiA)


    elif IS_SSTR27: # # SSH_R81 + SST_R09 to SSH_R27
        ISSHreso = all_Kinput_img[0] # Input SSH reso 81
        ArchiA   = UpSampling2D((upfactor, upfactor), interpolation='bilinear')(ISSHreso)
        #rchiA = Resizing(3*16,3*17,interpolation='bicubic')(ISSHreso)
        #ArchiA   = concatenate([ArchiA, ISSTR27], axis=1) # CHANNEL FIRST
        ArchiA   = concatenate([ArchiA, ISSTR27], axis=3) # CHANNEL LAST
        for i in np.arange(5) : #3
           
            ArchiA = Conv2D(32,(3,3), activation=factiv, padding='same',kernel_initializer=init)(ArchiA)
            ArchiA = Conv2D(32,(3,3), activation=factiv, padding='same',kernel_initializer=init)(ArchiA)#32
            ArchiA = BatchNormalization(axis=3)(ArchiA)
        ArchiA = Conv2D(8,(3,3), activation=factiv, padding='same',kernel_initializer=init)(ArchiA)
        if factout == 'linear':
          ArchiA = BatchNormalization(axis=3)(ArchiA)
        ArchiA = Conv2D(1,(1,1), activation=factout, padding='same',kernel_initializer=init,name='archi81-27')(ArchiA)


    else : # assumed Sans SST : # SSH_R81 to SSH_R27
        ISSHreso = all_Kinput_img[0] # Input SSH reso
        ArchiA   = UpSampling2D((upfactor, upfactor), interpolation='bilinear')(ISSHreso)
        for i in np.arange(7) :
            ArchiA = Conv2D(36,(6,6), activation=factiv, padding='same',kernel_initializer=init)(ArchiA)
        ArchiA = Conv2D(1,(1,1), activation=factout, padding='same',kernel_initializer=init,name='archi81-27')(ArchiA)
    ArchiOut.append(ArchiA)
    
    # 2ï¿½me ï¿½tage H_R27+SST_R09 to SSH_R09
    ArchiB = UpSampling2D((upfactor, upfactor), interpolation='bilinear')(ArchiA)
    #ArchiB = Resizing(3*48,3*51,interpolation='bicubic')(ArchiA)
    if IS_SSTR09 and 1 : # SST R09 utilisï¿½ pour trouver SSH-R09
        #ArchiB   = concatenate([ArchiB, ISSTR09], axis=1) # CHANNEL FIRST
        ArchiB   = concatenate([ArchiB, ISSTR09], axis=3) # CHANNEL LAST
    for i in np.arange(3) : #3
        
        ArchiB = Conv2D(23,(3, 3), activation=factiv, padding='same',kernel_initializer=init)(ArchiB)
        ArchiB = Conv2D(23,(3, 3), activation=factiv, padding='same',kernel_initializer=init)(ArchiB)#16
        ArchiB = BatchNormalization(axis=3)(ArchiB)
    ArchiB = Conv2D(8,(3,3), activation=factiv, padding='same',kernel_initializer=init)(ArchiB)
    ArchiB = Conv2D(1,(1, 1), activation=factout, padding='same',kernel_initializer=init,name='archi27-09')(ArchiB)
    ArchiOut.append(ArchiB);
    
    ArchiC = UpSampling2D((upfactor, upfactor), interpolation='bilinear')(ArchiB)
    #ArchiC = Resizing(3*144,3*153,interpolation='bicubic')(ArchiB)
    if IS_SSTR09 and 1 : # SST R09 utilisï¿½ pour trouver SSH-R09
        #ArchiC   = concatenate([ArchiC, ISSTR03], axis=1) # CHANNEL FIRST
        ArchiC   = concatenate([ArchiC, ISSTR03], axis=3) # CHANNEL LAST
    for i in np.arange(5) : #3
        
        ArchiC = Conv2D(10,(3, 3), activation=factiv, padding='same',kernel_initializer=init)(ArchiC)
        ArchiC = Conv2D(10,(3, 3), activation=factiv, padding='same',kernel_initializer=init)(ArchiC)#16
        ArchiC = BatchNormalization(axis=3)(ArchiC)
    ArchiC = Conv2D(8,(3,3), activation=factiv, padding='same',kernel_initializer=init)(ArchiC)
    ArchiC = Conv2D(1,(1, 1), activation=factout, padding='same',kernel_initializer=init,name='archi09-03')(ArchiC)
    ArchiOut.append(ArchiC);
    # 3ï¿½me ï¿½tage : H_R09+SST_R09 to U_R09 and V_R09
    
    
    nc, ml, mc, nl = (4,3,3,7)#(16, 5, 5, 1);
    if IS_SSTR09 : # SST R09 utilisï¿½ pour trouver U R09 et V R09
        ArchiUV  = concatenate([ArchiB, ISSTR09], axis=1) # CHANNEL FIRST
        #ArchiUV  = concatenate([ArchiC, ISSTR03], axis=3) # CHANNEL LAST
        ArchiUV  = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiUV) ##
    else :
        ArchiUV = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiB) ##
    for i in np.arange(nl) :
        ArchiUV = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiUV)
        ArchiUV = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiUV)
        ArchiUV = BatchNormalization(axis=3)(ArchiUV)
    ArchiUV = Conv2D(ceil(nc/2.0),(ml, mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiUV)
    ArchiUV = Conv2D(1,(1, 1), activation=factout, padding='same',kernel_initializer=init)(ArchiUV)
    #
    ArchiU = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiUV)
    for i in np.arange(nl) :
        ArchiU = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiU)
        ArchiU = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiU)
        ArchiU = BatchNormalization(axis=3)(ArchiU)
    ArchiU = Conv2D(ceil(nc/2.0),(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiU)
    ArchiU = Conv2D(1,(1, 1), activation=factout, padding='same',kernel_initializer=init,name='archiU')(ArchiU)
    ArchiOut.append(ArchiU)
    #
    ArchiV = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiUV)
    for i in np.arange(nl) :
        ArchiV = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiV)
        ArchiV = Conv2D(nc,(ml,mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiV)
        ArchiV = BatchNormalization(axis=3)(ArchiV)
    ArchiV = Conv2D(ceil(nc/2.0),(ml, mc), activation=factiv, padding='same',kernel_initializer=init)(ArchiV)
    ArchiV = Conv2D(1,(1, 1), activation=factout, padding='same',kernel_initializer=init,name='archiV')(ArchiV)
    ArchiOut.append(ArchiV)
    
    Mdl   = Model(all_Kinput_img, ArchiOut)
    Mdl.summary();
    Mdl.compile(loss='logcosh', optimizer=optimizers.Adam(learning_rate=2*(10**(-2.89637961))))
    print("Architecture completed")
    #
else : # --> RUN_MODE "RESUME" ou
       #              "REPRENDRE"
    print("Lecture du ficher Modele d'un apprentissage passé: ",Mdl2savedcase)
    np.random.seed(acide)
    Mdl = load_model(Mdl2reloadArchi)   # Chargement du modele (de l'archi)
    Mdl.summary();
#
#%%
print("SHAPE X TRAIN", "\n")
for k in range(len(x_train)):
   print(len(x_train), x_train[k].shape)

print("SHAPE Y TRAIN", "\n")
for k in range(len(y_train)):
   print(len(y_train), y_train[k].shape)
#======================================================================
#                LEARNING (ou reprendre)
#======================================================================
if RUN_MODE=="RESUME" :
    print("Reload des poids d'un model préalablement sauvegardé",Mdl2savedcase)
    Mdl.load_weights(Mdl2reloadWeights);
    print("reload weights, done that")
    #
    #----------------------------------------------------------------------
    try : # loss history
        with open(os.path.join(Mdl2reloadcase,'history.pkl'), 'rb') as file:
            LH = pickle.load(file)
    except :
        print("failed when reloading file history to replot loss curves")
    #
    plot_history(LH, title=f"Loss curves R{ResoIn[0]:02d}-R{ResoIn[-1]:02d} [{Mdl2savedcase}]",
                 linewidth=1, losslw=2, linestyle='--', valls='-', alpha=0.7,
                 log=True, rndcolors=False, fsize=(18,8),
                 figdir=images_dir, savefig=True)
#
elif RUN_MODE=="REPRENDRE":
    print("Reprise d'un apprentissage passé: ",Mdl2savedcase)
    Mdl.load_weights(Mdl2reloadWeights)   # Chargement des poids à partir desquel on reprend l'apprentissage
    print("Chargement des poids a repreprendre effectue")
    with open(Mdl2reloadHistLoss, 'rb') as file:
        HistoryFromReprendre=pickle.load(file)      #Chargement de l'historique des loss
    print("Chargement de l'historique des loss precedentes effectue")
    #
    t0=time()
    if VALID_ON==3: # Early stopping sur l'ensemle de validation. Les poids ou
                    # le mod�le sauvgardable sont obtenus � la fin du run
                    # apr�s patience it sans am�lioration).
        earlystop= EarlyStopping(monitor='val_loss',
                patience=int(Niter/4), verbose=2, mode='auto')
        H_reprendre = Mdl.fit(x_train, y_train, verbose=2, epochs=Niter,batch_size=Bsize,
                shuffle=True, callbacks=[earlystop], validation_data=(x_valid, y_valid))
    elif VALID_ON==4 or VALID_ON==5:
        log_dir = logs_fit_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        # 4 : Le run va jusqu'au bout. On r�cup�re, par la suite la sauvegarde
        #     des poids au meilleur de l'ensemble de validation pour les r�sultats.
        checkpoint_reprendre= ModelCheckpoint(os.path.join(weights_dir,"modelkvalid.ckpt"),
                                              monitor='val_loss', verbose=2,
                                              save_best_only=True, save_weights_only=True,
                                              mode='auto')

        callback_history = LossHistory(file=os.path.join(historique_dir,"modelkhistory.pkl"))
        try:
            os.remove(callback_history.file_name())
        except OSError:
            print("Le fichier Historique n'existe pas encore, il n'a pas pu être supprimé")
            pass

        callbacks_list=[checkpoint_reprendre, callback_history,tensorboard_callback] # Sauvegarde de H.history à chaque epoch

        H_reprendre = Mdl.fit(x_train, y_train, verbose=2, epochs=Niter, batch_size=Bsize,
                shuffle=True, callbacks=callbacks_list, validation_data=(x_valid, y_valid));
    print("Le run du mode REPRENDRE est allé jusqu'au bout des %d itérations"%(Niter))
    print("learning time : %f secondes" %(time()-t0))
    print("Temps moyen par itération: %f secondes " %((time()-t0)/Niter))
    #
    print("La longueur du dictionnaire de l'historique de l'apprentissage 1 est ",len(HistoryFromReprendre['val_loss']))
    print("La longueur du dictionnaire de l'historique de l'apprentissage 2 est ",len(H_reprendre.history['val_loss']))
    temp_hist=HistoryFromReprendre
    for key in HistoryFromReprendre.keys():
        HistoryFromReprendre[key].extend(H_reprendre.history[key])

    print("#Visualisation de la courbes d'erreur en apprentissage et de performances")
    plot_history(HistoryFromReprendre,title=f"REPRENDRE Loss curves R{ResoIn[0]:02d}-R{ResoIn[-1]:02d}",
                 linewidth=1.5, losslw=2.5, linestyle='--', valls='-', alpha=0.7,
                 log=True, rndcolors=False, fsize=(18,8),
                 figdir=images_dir, savefig=True)
    # plt.figure()
    # LH = []
    # kh = HistoryFromReprendre.keys()
    # LH.append(kh)
    # if VALID_ON :
    #     valid_err_hist = HistoryFromReprendre['val_loss']
    #     itminval = np.argmin(valid_err_hist)
    #     print("La val_loss n'a pas évolué après l'itération n° "+ str(itminval))

    # for kle in HistoryFromReprendre.keys() :
    #     hloss = HistoryFromReprendre[kle]
    #     plt.plot(np.log(hloss),alpha=0.7)
    #     hloss = np.reshape(hloss,(1,len(hloss)))
    #     LH.append(hloss[0])
    # plt.legend(HistoryFromReprendre.keys(), framealpha=0.3, loc=0)
    # plt.axvline(itminval,c="black")
    # plt.xlabel('epoch')
    # plt.ylabel('log(loss)')
    # plt.savefig("REPRENDRE/logloss_reprendre.png")
    if VALID_ON :
        valid_err_hist = HistoryFromReprendre['val_loss']
        itminval = np.argmin(valid_err_hist)
        print("La val_loss n'a pas évolué après l'itération n° "+ str(itminval))
    #
    with open(os.path.join(Mdl2reprendre,'history.pkl'), 'wb') as file:
        pickle.dump(HistoryFromReprendre, file)
    #
    Mdl.save(archi_train_dir)
    print('Architecture du modèle sauvegardée')
#
elif RUN_MODE=="LEARN":
    np.random.seed(acide);
    #
    # Sauvegarde de l'architecture du modele
    #Mdl.save_weights(Mdl2save+"wei");
    Mdl.save(archi_train_dir)
    print('Architecture du modèle sauvegardée')

    # Les param�tres de codage de l'ens d'App
    np.save(os.path.join(Mdl2save,"coparm"), [coparmAin, coparmAout]);
    print("Paramètres du codage sauvegardés");
    #
    t0 = time();
    if VALID_ON == 0 : # Pas d'usage de l'ensemble de validation (run 'simple)
        H = Mdl.fit(x_train, y_train, verbose=2,
                    epochs=Niter,batch_size=Bsize,shuffle=True);
    elif VALID_ON == 1 : # Usage de l'ensemble de validation (sans early stopping)
        H = Mdl.fit(x_train, y_train, verbose=2, epochs=Niter, batch_size=Bsize,
                    shuffle=True, validation_data=(x_valid, y_valid));
    elif VALID_ON == 3 : # Early stopping sur l'ensemle de validation. Les poids ou
                         # le modï¿½le sauvgardable sont obtenus ï¿½ la fin du run
                         # aprï¿½s patience it sans amï¿½lioration).
        earlystop = EarlyStopping(monitor='val_loss',
                patience=20, verbose=2, mode='auto');
        H = Mdl.fit(x_train, y_train, verbose=2, epochs=Niter,batch_size=Bsize,
                shuffle=True, callbacks=[earlystop], validation_data=(x_valid, y_valid));
    elif VALID_ON==4 or VALID_ON==5 :
        # 4 : Le run va jusqu'au bout. On rï¿½cupï¿½re, par la suite la sauvegarde
        #     des poids au meilleur de l'ensemble de validation pour les rï¿½sultats.
        tensorboard_callback = TensorBoard(log_dir=logs_fit_dir, histogram_freq=1)
        checkpoint= ModelCheckpoint(os.path.join(weights_dir,"modelkvalid.ckpt"), monitor='val_loss',
                                    verbose=2, save_best_only=True, save_weights_only=True,
                                    mode='auto')
        callback_history = LossHistory(file=os.path.join(historique_dir,"modelkhistory.pkl"))
        #checkpoint= ModelCheckpoint(Mdl2save+"valid.ckpt", monitor='val_loss', verbose=2,
        #        save_best_only=True, save_weights_only=True, mode='auto')
        #
        try:
            os.remove(callback_history.file_name())
        except OSError:
            print("Le fichier n'existe pas encore, il n'a pas pu être supprimé")
            pass
        #
        callbacks_list=[checkpoint, callback_history,tensorboard_callback] # Sauvegarde de H.history à chaque epoch
        #
        # Model Fit()
        #
        H = Mdl.fit(x_train, y_train, verbose=2, epochs=Niter, batch_size=Bsize,
                shuffle=True, callbacks=callbacks_list, validation_data=(x_valid, y_valid));
    #
    print("learning time : %f" %(time()-t0))
    print("temps moyen des" +str(Niter)+ " iterations:"+str((time()-t0)/Niter))
    with open(os.path.join(Mdl2save,'history.pkl'), 'wb') as file:
        pickle.dump(H.history, file)
    print("L'entrainement est allé jusqu'au bout des "+str(Niter)+" itérations" )
    #
    # if Mdl2save is not None :
    #     # Sauvegarde des poids du model
    #     #Mdl.save_weights(Mdl2save+"wei");
    #     Mdl.save("test_save_husseine/saved_wei.h5")
    #     #
    #     # Les paramï¿½tres de codage de l'ens d'App
    #     np.save(Mdl2save+"coparm", [coparmAin, coparmAout]);
    #     print("model %s  saved"%Mdl2save);
    #
    print("#Visualisation de la courbes d'erreur en apprentissage et de performances")
    # plt.figure()
    # LH = []
    # kh = H.history.keys()
    # LH.append(kh);
    # for kle in H.history.keys() :
    #     hloss = H.history[kle]
    #     plt.plot(np.log(hloss),alpha=0.8)
    #     hloss = np.reshape(hloss,(1,len(hloss)))
    #     LH.append(hloss[0])
    # plt.legend(H.history.keys(), framealpha=0.3, loc=0)
    # plt.xlabel('epoch')
    # plt.ylabel('log(loss)')
    if VALID_ON :
        valid_err_hist = H.history['val_loss']
        itminval = np.argmin(valid_err_hist)
        print("itminval =",itminval)
#
#%%
#======================================================================
#                       RESULTS ON MODEL LEARNED
#======================================================================
if VALID_ON == 4 :
    if RUN_MODE == "LEARN" :
        # ON prend les poids au min de la validation (sauvegardï¿½s pendant le run)
        Mdl.load_weights(os.path.join(weights_dir,"modelkvalid.ckpt"));
        #
    elif RUN_MODE == "REPRENDRE":
        #Si le resultat obtenu precedement est meilleur, on charge les poids precedents
        #Par contre si les poids après réentrainement sont meilleurs c'est eux que l'on charge
        if temp_hist['val_loss'][np.argmin(temp_hist['val_loss'])] < H_reprendre.history['val_loss'][np.argmin(H_reprendre.history['val_loss'])]:
            #Les anciens resultats sont meilleurs
            Mdl.load_weights(os.path.join(Mdl2olddirname,Mdl2savedcase,'Weights','modelkvalid.ckpt'))
            print("On recharge les poids précédents car la reprise n'a pas amélioré la val_loss")
        else: # Le modele reprendre a eu de meilleures performances
            Mdl.load_weights(Mdl2reloadWeights)
            print("On charge les nouveaux poids après reprise car la val_loss s'est améliorée")
        #
    else : # suppose="RESUME" : on reprend des poids d'un run prï¿½cedent
        print("Mode "+ RUN_MODE+" les poids ont déjà été chargés")
    #
    print("load weights min valid, done this");
#----------------------------------------------------------------------
#
if RUN_MODE=="LEARN":
    plot_history(H.history, title=f"Loss curves R{ResoIn[0]:02d}-R{ResoIn[-1]:02d}",
                 linewidth=1, losslw=2, linestyle='--', valls='-', alpha=0.7,
                 log=True, rndcolors=False, fsize=(18,8),
                 figdir=images_dir, savefig=True)

    print("Affichage des performances")

def setresult(Mdl, x_set, strset, VXout_brute, calX,
              ResIn, ResOut, Din_diclst, Dout_diclst,
              flag_rms=0, flag_scat=0, flag_histo=0, nbins=NBINS, flag_cosim=0,
              flag_nrj=0, flag_ens=0, flag_corrplex=0, flag_div=0, 
              dx=dxRout, dy=dyRout, wdif='log', im2show=[],
              fsizesome=(10,12),
              fsizehquiv=(10,12),
              fsizerms=(12,8), szxlblrms=8,
              fsizehisto=(12,8),
              fsizescat=(10,10), squarescatax=False, 
              figdir='.', savefig=False) :

    # Predicted coded data
    y_scale = Mdl.predict(x_set);
    for i in np.arange(NvarOut):
        y_scale[i] = y_scale[i].transpose(0,3,1,2)
    if len(varOut)==1 : # Si une seule sortie mettre y_scale en forme de list comme l'est y_train
        y_scale = [y_scale];
    #BACK TO BRUTE

    print("%s BACK TO BRUTE"%strset);

    for i in targetout : # boucle sur les variables cibles en sortie
        wk_ = tvwmm.index(varOut[i]);
        y_scale[i] = decodage(y_scale[i], coparmAout[i]);

        if not RESBYPASS :
            if LIMLAT_NORDSUD <= 0 :
                result("brute scalled", strset, varOut[i], VXout_brute[i], y_scale[i], ResOut[i],
                       flag_rms=flag_rms, flag_scat=flag_scat, flag_histo=flag_histo,
                       nbins=nbins, calX=calX, fsizerms=fsizerms, szxlblrms=szxlblrms, 
                       fsizehisto=fsizehisto,
                       fsizescat=fsizescat, squarescatax=squarescatax, 
                       figdir=figdir, savefig=savefig);
            else : # PLM, je choisis de faire que tout, ou que Nord-Sud, pour ï¿½viter
                   # de crouler sous les images
                pipo, pipo, Nlig_, pipo = np.shape(VXout_brute[i]);
                # Nombre de ligne jusqu'ï¿½ la latitude limite Nord-Sud
                NL_lim = nl2limlat (Nlig_, LIMLAT_NORDSUD);
                # Split Nord-Sud
                youtN_,  youtS_ = splitns(VXout_brute[i], NL_lim);
                yscalN_, yscalS_= splitns(y_scale[i], NL_lim);
                result("brute scalled", strset+"-Nord", varOut[i], youtN_, yscalN_, ResOut[i],
                       flag_rms=flag_rms, flag_scat=flag_scat, flag_histo=flag_histo,
                       nbins=nbins, calX=calX, fsizerms=fsizerms, szxlblrms=szxlblrms,
                       fsizehisto=fsizehisto,
                       fsizescat=fsizescat, squarescatax=squarescatax, 
                       figdir=figdir, savefig=savefig);
                result("brute scalled", strset+"-Sud", varOut[i], youtS_, yscalS_, ResOut[i],
                       flag_rms=flag_rms, flag_scat=flag_scat, flag_histo=flag_histo,
                       nbins=nbins, calX=calX, fsizerms=fsizerms, szxlblrms=szxlblrms,
                       fsizehisto=fsizehisto,
                       fsizescat=fsizescat, squarescatax=squarescatax, 
                       figdir=figdir, savefig=savefig);

        min_= np.min(y_scale[i]);  max_= np.max(y_scale[i])
        moy_= np.mean(y_scale[i]); std_=np.std(y_scale[i])
        print("%s scalled : %s_decodee(out9), min=%f, max=%f, mean=%f, std=%f"
                    %(strset, varOut[i], min_, max_, moy_, std_));

        if not FIGBYPASS and len(im2show) > 0 :
            suptitre="some %s %s_Yscalled decodee %s"%(strset, varOut[i],im2show);
            showsome(y_scale[i][im2show,0,:,:], ResOut[i], Dout_diclst[i], wmin=wbmin[wk_], wmax=wbmax[wk_],
                     wnorm=wnorm[wk_], cmap=wcmap[wk_], calX0=calX[0][im2show],
                     Xref=VXout_brute[i][im2show,0,:,:], wdif="", wdifr=None,
                     varlib=varOut[i], suptitre=suptitre, fsize=fsizesome,
                     figdir=figdir, savefig=savefig);
            print();

    # Rï¿½sultat sur vecteur uv ; hors de la boucle parce qu'on en a besoin que
    # d'une fois, et qu'il faut attendre la fin de la boucle pour que
    # y_scale soit dï¿½codï¿½; ATTENTION, cela n'est fait que pour les targetout !!!
    resultuv(VXout_brute, y_scale, ResOut, Dout_diclst, strset, flag_cosim=flag_cosim,
             flag_nrj=flag_nrj, flag_ens=flag_ens, flag_corrplex=flag_corrplex,
             flag_div=flag_div, dx=dxRout, dy=dyRout, flag_scat=flag_scat,
             flag_histo=flag_histo, calX=calX, wdif='log');
    #
    if not FIGBYPASS :
        if IS_HUVout and 1 : # Vecteurs UV on H
            suptitre = "some %s(out) H brute scalled(+uv) %s"%(strset,im2show); #! (H en dure ?)
            ih_ = showhquiv(y_scale, ResOut, Dout_diclst, im2show, calX0=calX[0], Yref=VXout_brute, wdif="",
                            wdifr=None, suptitre=suptitre, fsize=fsizehquiv,
                            figdir=figdir,  savefig=savefig);
#
#%%
#------------
if FLGA_RES : # sur APP
    setresult(Mdl, x_train, "APP", VAout_brute, calA,
              ResoIn, ResoOut, Din_dico_list, Dout_dico_list,
              flag_rms=FLGA_RMS,
              flag_scat=FLGA_SCAT, flag_histo=FLGA_HISTO, nbins=NBINS,
              flag_cosim=FLGA_COSIM, flag_nrj=FLGA_NRJ, flag_ens=FLGA_ENS,
              flag_corrplex=FLGA_CORRPLEX, flag_div=FLGA_DIV,
              dx=dxRout, dy=dyRout, wdif='log', im2show=im2showA,
              fsizesome=(10,12),   # taille des figures de cartes de SSSH, SST...
              fsizehquiv=(10,12),  # taille des figures de cartes avec quiver ...
              fsizerms=(26,6), szxlblrms=4, # taille des figures de plots rms par image (ou jour), fontsize xtick labels
              fsizehisto=(12,8),   # taille des figures des histogrammes
              fsizescat=(10,10), squarescatax=True, # taille des figures de scatter-plots, flag axis carre ...
              figdir=images_dir, savefig=SAVEFIG)
#%%
#------------
if TEST_ON and FLGT_RES : # sur TEST
    setresult(Mdl, x_test, "TEST", VTout_brute, calT,
              ResoIn, ResoOut, Din_dico_list, Dout_dico_list,
              flag_rms=FLGT_RMS,
              flag_scat=FLGT_SCAT, flag_histo=FLGT_HISTO, nbins=NBINS,
              flag_cosim=FLGT_COSIM, flag_nrj=FLGT_NRJ, flag_ens=FLGT_ENS,
              flag_corrplex=FLGT_CORRPLEX, flag_div=FLGT_DIV,
              dx=dxRout, dy=dyRout, wdif='log',  im2show=im2showT,
              fsizesome=(10,12),   # taille des figures de cartes de SSSH, SST...
              fsizehquiv=(10,12),  # taille des figures de cartes avec quiver ...
              fsizerms=(18,6), szxlblrms=8, # taille des figures de plots rms par image (ou jour), fontsize xtick labels
              fsizehisto=(12,8),   # taille des figures des histogrammes
              fsizescat=(10,10), squarescatax=True, # taille des figures de scatter-plots, flag axis carre ...
              figdir=images_dir, savefig=SAVEFIG);
#------------
#plt.show();
#======================================================================
#%%
