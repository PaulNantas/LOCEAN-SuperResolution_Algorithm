# -*- coding: utf-8 -*-
"""
 **************************************************************************
 resacartdef.py
 
 Librerie de fonctions de ResacNet.

 Historique:
    2021-06-06 ResacNet - Changing scatplot() function in resacartdef.py for
                          bug corrections. Correcting axis limits when plotting
                          identity diagonal. Controling axis limits.
                          Adding new parameters SCATALPHA and SCATLABELS in
                          resacartparm.py to be used in scatplot() function.
    2021-05-21 ResacNet - adding data_prefix and data_suffix options to
                          load_resac_by_var_and_resol data loading function.
 **************************************************************************
"""
from __future__ import print_function
import os
import sys
import pickle
import random
import math
from   time  import time
import numpy as     np
import matplotlib as mpl #see: ../matplotlib/rcsetup.py
import matplotlib.pyplot as plt
from   matplotlib import cm
import matplotlib.colorbar as cb
from   matplotlib.colors import LogNorm
from   scipy.stats import norm
from   resacartparm import *
#

if KERASBYTENSORFLOW :  
    
    from tensorflow.keras.callbacks import Callback
else:
    if PLAIDMLKERASBACKEND :  # backend pour cartes graphiques non NVIDIA
        os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    #
    from keras.callbacks import Callback

#
#=====================================================================
# Traitement post param�trage
#---------------------------------------------------------------------
# D�duit du param�trage
def lidex(lin, idx) : # extrait de la list lin les �l�ments d'index idx
    lout = []
    for i in idx :
        lout.append(lin[i]);
    return lout;
if SCENARCHI>0 :
    varoutcible = lidex(varOut, targetout);
NvarIn    = len(varIn);     NvarOut= len(varOut);
IS_UVout  = False;
IS_HUVout = False;
if "U" in varOut and "V" in varOut :
    IS_UVout  = True;    
    if "SSH" in varOut :
        IS_HUVout = True;   
#
# Delta x et y des donn�es de sortie (R09 pour resac*) :
dxRout, dyRout = (ResoOut[-1]*dxR01, ResoOut[-1]*dyR01);
#
#----------------------------------------------------------------------
# Calendrier had-oc
# if not CALENDAR_FROM_DATA : 
#     mois  = ["oct","nov","dec","jan","feb","mar","apr","may","jun,","jul","aug","sep","oct"]; 
#     nday  = [  31,   30,   31,   31,   28,   31,   30,   31,   30,    31,   31,   30,    1];
#     annee = 2012;
#     calendrier = [];
#     for i in np.arange(len(mois)) :
#         if mois[i]=="jan" :
#             annee = 2013;
#         for j in np.arange(nday[i]) :
#             #jourj = "%d/%s/%d"%(j+1,mois[i],annee)
#             jourj = "%d%s%d"%(j+1,mois[i],annee)
#             calendrier.append(jourj);
#     calendrier = np.array(calendrier);
#----------------------------------------------------------------------
# Calculer les Coordonn�es G�ographique en fonction des r�solutions une fois pour toute
# lonfr=40.; lonto=65.; latfr=26.; latto=45.; # Coordonn�e :26�N, 45�N; 40�W, 65�W
# def coordgeo(NL_,NC_,unsn) :
#     lxticks = np.arange(lonto, lonfr-1, (lonfr-lonto)/(NC_-1));
#     lxticks = lxticks.astype(int);      # l comme label ; arrondi au degr� le plus proche
#     xxticks = np.arange(len(lxticks));  # coordonn�e en x
#     #iunsn   = np.arange(0,NC_,unsn);    # prendre un point sur n - P2
#     iunsn   = np.arange(0,NC_,unsn).astype(int);    # prendre un point sur n  - P3
#     xxticks = xxticks[iunsn]
#     lxticks = lxticks[iunsn];
#
#     lyticks = np.arange(latfr, latto+1, (latto-latfr)/(NL_-1)); #NL valeurs de 25�->45� (de bas en haut)
#     lyticks = lyticks.astype(int);      # l comme label ; arrondi au degr� le plus proche
#     yyticks = np.arange(len(lyticks));  # coordonn�e en y
#     #iunsn   = np.arange(0,NL_,unsn);    # prendre un point sur n - P2
#     iunsn   = np.arange(0,NL_,unsn).astype(int);    # prendre un point sur n  - P3
#     yyticks = yyticks[iunsn]
#     lyticks = lyticks[iunsn];
#     lxticks = -lxticks; # <- Signe moins pours les degr�s West
#     return xxticks, lxticks, yyticks, lyticks;
# CoordGeo = [];
# ResoInterp2 = [1]; # [1] pour Interp2reso
# ResoAll     = ResoIn + ResoOut + ResoInterp2;
# ResoAll     = list(np.unique(ResoAll));
# for i in np.arange(len(ResoAll)) :
#     # D�duire NL, NC � partir de la r�solution sachant que R01 : 1296x1377
#     NL_ = NLigR01 / ResoAll[i];
#     NC_ = NColR01 / ResoAll[i];
#     # D�terminer un nombre de points de coordonn�es
#     unsn = NL_ / 8; #2      # prendre un point sur n (8 choisi en DURE)
#     rc_ = coordgeo(NL_, NC_, unsn); # coordonn�es pour cette r�solution
#     CoordGeo.append(rc_);
#----------------------------------------------------------------------
#
def build_all_resol_dic(dic_r1, all_r, native_resol=1) :
    dico_all_r = {}
    for r in all_r :
        if r == native_resol:
            dico_all_r[f'R{r:02d}'] = dic_r1
        elif r > native_resol:
            lat_x, lon_x = dic_r1['lat'], dic_r1['lon']
            lat_r, lon_r, lat_border_r, lon_border_r = build_lower_resol_vectors(lat_x, lon_x, r//native_resol, borders=True)
            diccoordtmp = {}
            for k in dic_r1.keys():
                if k == 'lat':
                    diccoordtmp[k] = lat_r
                elif k == 'lon':
                    diccoordtmp[k] = lon_r
                elif k == 'lat_border':
                    diccoordtmp[k] = lat_border_r
                elif k == 'lon_border':
                    diccoordtmp[k] = lat_border_r
                else:
                    diccoordtmp[k] = dic_r1[k]
            dico_all_r[f'R{r:02d}'] = diccoordtmp
        else:
            print(f"resolution {r} smaller then native data resolution of {native_resol}. Not generated")
    #
    return dico_all_r
#
def build_lower_resol_vectors(lat, lon, r, borders=False):
    print(f'genere lat/lon pour resolution {r}')
    lat_rLow = lat[(r//2)::r]
    lon_rLow = lon[(r//2)::r]
    if borders :
        delta_lat_rLow = (lat_rLow[1]-lat_rLow[0])
        delta_lon_rLow = (lon_rLow[1]-lon_rLow[0])
        lat_border_rLow = np.concatenate((lat_rLow - delta_lat_rLow/2,[lat_rLow[-1] + delta_lat_rLow/2])); 
        lon_border_rLow = np.concatenate((lon_rLow - delta_lon_rLow/2,[lon_rLow[-1] + delta_lon_rLow/2])); 
        #
        return lat_rLow, lon_rLow, lat_border_rLow, lon_border_rLow
    else:
        return lat_rLow, lon_rLow
#
def select_data_by_dim_by_list(data, set_of_index, dim_axis=0):
    if dim_axis == 0:
        data = data[set_of_index]
    elif dim_axis == 1:
        data = data[:,set_of_index]
    elif dim_axis == 2:
        data = data[:,:,set_of_index]
    elif dim_axis == 3:
        data = data[:,:,:,set_of_index]
    elif dim_axis == 4:
        data = data[:,:,:,:,set_of_index]
    else:
        assert False, f'Invalid dimension index: {dim_axis}. Edit this function to allow it'
    #
    return data
#
def select_data_by_dim(data, index_pattern, dim_lbl=None, dim_axis=0):
    if np.isscalar(index_pattern) or len(index_pattern) == 2:
        if np.isscalar(index_pattern):
            i0, step = 0, index_pattern
        else:
            i0, step = index_pattern
        if i0 >= 0 and step > 1 :
            list_of_index = np.arange(i0,data.shape[dim_axis],step)
            data = select_data_by_dim_by_list(data, list_of_index, dim_axis=dim_axis)
            if dim_lbl is not None :
                new_dim_lbl = dim_lbl[list_of_index]

        elif i0 < 0 or step < 0 or (i0 + step) > data.shape[dim_axis] :
            assert False, f'Invalid i0 or step for dimension: {dim_axis}'
        else:
            # vraisemblamblement index_pattern est (0, 1) alors, rien ne change
            if dim_lbl is not None :
                new_dim_lbl = dim_lbl
    #
    if dim_lbl is None :
        return data
    else:
        return data, new_dim_lbl
#
def select_by_coords(data, dim_dic, lat_limits=None, lon_limits=None,
                     lat_axis=1, lon_axis=2, epsilon=1e-4):
    if lat_limits is not None :
        latmin,latmax = lat_limits
        if latmin is not None :
            latmin -= epsilon
        if latmax is not None :
            latmax += epsilon
        lat_limits = latmin,latmax
        # selectionne lat_border d'abord
        if latmin is not None and latmax is not None :
            bool_lat_border = [d >= latmin and d <= latmax for d in dim_dic['lat_border']]
        elif latmin is not None :
            bool_lat_border = dim_dic['lat_border'] >= latmin
        elif latmax is not None :
            bool_lat_border = dim_dic['lat_border'] <= latmax
        dim_dic['lat_border'] = dim_dic['lat_border'][bool_lat_border]
        # selectionne ensuite lat selon les limites de lat_border
        bool_lat = [d > min(dim_dic['lat_border']) and d < max(dim_dic['lat_border']) for d in dim_dic['lat'] ]
        dim_dic['lat'] = dim_dic['lat'][bool_lat]
        data = select_data_by_dim_by_list(data, bool_lat, dim_axis=lat_axis)
    #
    if lon_limits is not None :
        lonmin,lonmax = lon_limits
        if lonmin is not None :
            lonmin -= epsilon
        if lonmax is not None :
            lonmax += epsilon
        lon_limits = lonmin,lonmax
        # selectionne lon_border d'abord
        if lonmin is not None and lonmax is not None :
            bool_lon_border = [d >= lonmin and d <= lonmax for d in dim_dic['lon_border']]
        elif lonmin is not None :
            bool_lon_border = dim_dic['lon_border'] >= lonmin
        elif lonmax is not None :
            bool_lon_border = dim_dic['lon_border'] <= lonmax
        dim_dic['lon_border'] = dim_dic['lon_border'][bool_lon_border]
        # selectionne ensuite lon selon les limites de lon_border
        bool_lon = [d > min(dim_dic['lon_border']) and d < max(dim_dic['lon_border']) for d in dim_dic['lon'] ]
        dim_dic['lon'] = dim_dic['lon'][bool_lon]
        data = select_data_by_dim_by_list(data, bool_lon, dim_axis=lon_axis)
            
    return  data, dim_dic
#
#----------------------------------------------------------------------
def get_real_lat_lon_limits(lat_list, lon_list, zone=None, 
                            lat_limits=None, lon_limits=None, verbose=False):
    if zone is not None:
        if zone.lower() in ["north","nord","south","sud"] :
            # convert l'option ZONE North ou South en option LAT [LMin, LMax]
            if verbose:
                print(f'ZONE: passe de {zone.upper()}')
            half_latitude = min(lat_list) + (max(lat_list) - min(lat_list)) / 2
            if zone.lower() in ["north","nord"] :
                lat_limits = [half_latitude, None]
            elif zone.lower() in ["south","sud"] :
                lat_limits = [None, half_latitude]
            if verbose:
                print(f' ... a: {lat_limits}')
        else:
            assert False, f'Invalid ZONE specification: {zone}'
    else:
        if lat_limits is not None :
            latmin,latmax = lat_limits
            print(f'LAT: passe de: {lat_limits}',end='')
            if latmin is not None :
                latmin = min(lat_list[lat_list >= latmin]) # nouvel LATMIN selon les lat_border
            if latmax is not None :
                latmax = max(lat_list[lat_list <= latmax]) # nouvel LATMIN selon les lat_border
            lat_limits = latmin,latmax
            print(f' ... a: {lat_limits}')
        if lon_limits is not None :
            lonmin,lonmax = lon_limits
            print(f'LON: passe de: {lon_limits}',end='')
            if lonmin is not None :
                lonmin = min(lon_list[lon_list >= lonmin]) # nouvel lonMIN selon les lon_border
            if lonmax is not None :
                lonmax = max(lon_list[lon_list <= lonmax]) # nouvel lonMIN selon les lon_border
            lon_limits = lonmin,lonmax
            print(f' ... a: {lon_limits}')
    #
    return lat_limits, lon_limits
#
#----------------------------------------------------------------------
def get_zonelatlon_label(zone=None, lat_limits=None, lon_limits=None):
    label = ''
    if zone is not None:
        label += f'_{zone.upper()}'
    if lat_limits is not None:
        label += f'_LAT'+\
            f'{"-NA" if lat_limits[0] is None else (str(lat_limits[0]) if lat_limits[0] < 0 else "+"+str(lat_limits[0]))}'+\
                f'{"-NA" if lat_limits[1] is None else (str(lat_limits[1]) if lat_limits[1] < 0 else "+"+str(lat_limits[1]))}'                    
    if lon_limits is not None:
        label += f'_LON'+\
            f'{"-NA" if lon_limits[0] is None else (str(lon_limits[0]) if lon_limits[0] < 0 else "+"+str(lon_limits[0]))}'+\
                f'{"-NA" if lon_limits[1] is None else (str(lon_limits[1]) if lon_limits[1] < 0 else "+"+str(lon_limits[1]))}'
    return label
#
#======================================================================
# D�finitions
#----------------------------------------------------------------------
def fit01(X, gap01=0.0, coparm=None, verbose=False) : 
    ''' Ram�ne les valeurs de X dans l'intervalle [0, 1] + gap01
        On retourne les valeurs (d=max(X)-min(X) et min(X/d) qui
        permettront � la fonction fit01 (ci-dessous) de
        r�aliser l'op�ration inverse.
    '''
    if verbose:
        print("       FIT01 -> size: %s:\n   n,min/Max/mean/std Avant: "%(','.join(str(x) for x in X.shape)),
              X.min(),X.max(),X.mean(),X.std())
    if coparm is None :
        a = np.min(X);
        b = np.max(X)
        deltax = b-a; 
        Y = X / deltax;
        miny = np.min(Y)
        Y = Y - miny;
        Y = Y + gap01;
        coparm = ("fit01", miny, deltax, gap01);
        if verbose:
            print("   [FIT01] n,min/Max/mean/std Apres: ",Y.min(),Y.max(),Y.mean(),Y.std())
        return Y, coparm
    elif coparm[0] == "fit01" :
        nom,miny,deltax,gap01 = coparm
        Y = X / deltax;
        Y = Y - miny;
        Y = Y + gap01;
        return Y
    else:
        print(f" ** fit01: nom de codage {coparm[0]} inattendu. Devrait etre 'fit01' ...")
        sys.exit(1)
#----------------------------------------------------------------------
def inv_fit01(Y,coparm) :
    ''' R�alise l'op�ration inverse de la fonction fit01 ci-dessus.
    miny et d sont les param�tres qui ont �t� retourn�s par fit01.
    '''
    miny,d,gap01 = coparm[1:];
    #
    Y = Y - gap01;
    Y = Y + miny
    X = Y * d
    return X
#----------------------------------------
def refit01(X, minx0, maxx0, gap01=0.0) :
    # minx0, maxx0 : le min et le max des donn�es initiales avec
    # lesquelles le fit01 a �t� fait
    delta = maxx0 - minx0;
    Y = (X / delta) - (minx0/delta)
    Y = Y + gap01;
    return Y
#----------------------------------------------------------------------
def inv_refit01(Y, minx0, maxx0, gap01=0.0) :
    Y = Y - gap01;
    delta = maxx0 - minx0;
    X = (delta*Y) + minx0;
    return X
#----------------------------------------
def centrereduc(X, coparm=None, verbose=False) :
    ''' Centre-reduction: transforme les valeurs de X pour avoir une moyenne a 
        zero et un ecart-type de 1.
        On retourne les valeurs (moyenne et ecart-type) qui
        permettront a la fonction decentrereduc (ci-dessous) de
        realiser l'operation inverse.
    '''
    if verbose:
        print("       CENTREREDUC -> size: %s:\n   n,min/Max/mean/std Avant: "%(','.join(str(x) for x in X.shape)),
              X.min(),X.max(),X.mean(),X.std())
    if coparm is None :
        mean = np.mean(X);
        std = np.std(X)
        Y = (X - mean) / std;
        coparm = ("cenred", mean, std);
        if verbose:
            print("   [CENTREREDUC] n,min/Max/mean/std Apres: ",Y.min(),Y.max(),Y.mean(),Y.std())
        return Y, coparm
    elif coparm[0] == "cenred" :
        nom, mean, std = coparm
        Y = (X - mean) / std;
        return Y
    else:
        print(f" ** centrereduc: nom de codage '{coparm[0]}' inattendu. Devrait etre 'cenred' ...")
        sys.exit(1)
#----------------------------------------------------------------------
def decentrereduc(Y,coparm) :
    ''' R�alise l'op�ration inverse de la fonction fit01 ci-dessus.
    miny et d sont les param�tres qui ont �t� retourn�s par fit01.
    '''
    nom,mean,std = coparm;
    X = (Y * std) + mean;
    return X
#----------------------------------------------------------------------
def codage(X,CODAGE,gap01=0.0, verbose=False) : 
    if verbose:
        print("     CODAGE:\n")
    if CODAGE=="fit01" :
        X, coparm = fit01(X, verbose=verbose);
    elif CODAGE=="cenred" :
        X, coparm = centrereduc(X, verbose=verbose);
    elif CODAGE=="cr+fit01" :
        Xcr, coparm_cr = centrereduc(X, verbose=verbose)
        n_cr, mn_cr, st_cr = coparm_cr
        X, coparm_f01 = fit01(Xcr, verbose=verbose);
        n_f01, miny_f01, d_f01, gap01_f01 = coparm_f01
        coparm = ( "cr+fit01", mn_cr, st_cr, miny_f01, d_f01, gap01_f01) 
    else :
        raise ValueError(f"codage: bad code: '{CODAGE}'");
    return X, coparm
#----------------------------------------------------------------------
def decodage(Y,coparm) :
    CODAGE = coparm[0];
    if CODAGE=="fit01" :
        X = inv_fit01(Y,coparm);
    elif CODAGE=="cenred" :
        X = decentrereduc(Y,coparm);
    elif CODAGE=="cr+fit01" : # de-normalisation inv-fit01 puis de-centre-redcuction
        nom, mn_cr, st_cr, miny_f01, d_f01, gap01_f01 = coparm
        X01 = inv_fit01(Y,("fit01",miny_f01, d_f01, gap01_f01))
        X = decentrereduc(X01,("cenred", mn_cr, st_cr))
    else :
        raise ValueError("decodage: code %s is unknown"%CODAGE);
    return X
#----------------------------------------------------------------------
def recodage(X,coparm) :
    CODAGE = coparm[0];
    if CODAGE=="fit01" :
        X = fit01(X, coparm=coparm);
    elif CODAGE=="cenred" :
        X = centrereduc(X, coparm=coparm);
    elif CODAGE=="cr+fit01" :
        nom, mn_cr, st_cr, miny_f01, d_f01, gap01_f01 = coparm
        Xcr = centrereduc(X, coparm=("cenred", mn_cr, st_cr))
        X   = fit01(Xcr, coparm=("fit01",miny_f01, d_f01, gap01_f01))
    else :
        raise ValueError("recodage: code %s is unknown"%CODAGE);
    return X
#----------------------------------------------------------------------
def codage_multivar(Xlist, codefunc="fit01", verbose=False, verboseplus=False):
    V = []; coparm_list = []
    if np.isscalar(codefunc) :
        codefunc = [codefunc]*len(Xlist)
    for i,Z in enumerate(zip(Xlist,codefunc)):
        X,cfunc = Z
        if verboseplus:
            print("\n-> CODAGE_MULTIVAR(%d) [%s] -> size: %s:"%(i+1,cfunc,','.join(str(x) for x in X.shape)))
            print("   n,min/Max/mean/std Avant: ", X.min(),X.max(),X.mean(),X.std())
        V_, c_= codage(X, cfunc, verbose=verboseplus)
        if verboseplus:
            print("   [CODAGE_MULTIVAR] n,min/Max/mean/std Apres: ",V_.min(),V_.max(),V_.mean(),V_.std())
        V.append(V_)
        coparm_list.append(c_)
        if verbose:
            print("   ",c_)
    return V, coparm_list
#----------------------------------------------------------------------
def recodage_multivar(Xlist,coparm_list):
    V = []
    for X,coparm in zip(Xlist,coparm_list):
        V_ = recodage(X, coparm)
        V.append(V_)
    return V
def decodage_multivar(Ylist,coparm_list):
    V = []
    for Y,coparm in zip(Ylist,coparm_list):
        Z_ = decodage(Y, coparm)
        V.append(V_)
    return V
#
#======================================================================
def showimgdata(X, Labels=None, n=1, fr=0, interp=None, cmap=CMAP_DEF, nsubl=None, 
                vmin=None, vmax=None, facecolor='w', vnorm=None, origine='lower',
                U=None, V=None, ticks=None, qscale=30, fsize=(12,16)) :
    if nsubl == None :
        nbsubc = np.ceil(np.sqrt(n));
        nbsubl = np.ceil(1.0*n/nbsubc);
    else :
        nbsubl = nsubl;
        nbsubc = np.ceil(1.0*n/nbsubl);
    nbsubl=int(nbsubl); #nbsubl.astype(int)
    nbsubc=int(nbsubc); #nbsubc.astype(int)
  
    if vmin is None :
        vmin = np.min(X);
    if vmax is None :
        vmax = np.max(X);
        
    N, M, P, Q = np.shape(X);      

    ISUV=False; 
    if U is not None and V is not None :
        ISUV=True;
    ISTICKS=False;
    if ticks is not None :
        ISTICKS=True;
        xxticks, lxticks, yyticks, lyticks = ticks;

    #M, P, Q = np.shape(X[0]);      
    fig, axes = plt.subplots(nrows=nbsubl, ncols=nbsubc,
                        sharex=True, sharey=True, figsize=fsize,facecolor=facecolor)
    fig.subplots_adjust(wspace=0.1, hspace=0.3, bottom=0.0)
    ifig = 0;
    for ax in axes.flat :
        if ifig < N and ifig < n : 
            img  = X[ifig+fr]; #img = X[i];               
            if M == 1 :
                img  = img.reshape(P,Q)
            elif M != 3 :
                print("showimgdata: Invalid data dimensions image : must be : 1xPxQ or 3xPxQ")
                sys.exit(0);
            else : #=> = 3
                img  = img.transpose(1,2,0);
            if vnorm=="LogNorm" :
                ims = ax.imshow(img, norm=LogNorm(vmin=vmin, vmax=vmax),
                                interpolation=interp, cmap=cmap, origin=origine);
            else : 
                ims = ax.imshow(img, interpolation=interp, cmap=cmap, vmin=vmin,
                                vmax=vmax, origin=origine);
            if ISUV :
                u = U[ifig+fr].reshape(P, Q);
                v = V[ifig+fr].reshape(P, Q);
                ax.quiver(u, v, scale=qscale);
            if Labels is not None :
                ax.set_title(Labels[ifig+fr],fontsize=x_figtitlesize)
            if ISTICKS :
                ax.set_xticks(xxticks); ax.set_xticklabels(lxticks);
                ax.set_yticks(yyticks); ax.set_yticklabels(lyticks);
            ifig += 1;
    cbar_ax,kw = cb.make_axes([ax for ax in axes.flat],orientation="horizontal",
                             pad=0.05,aspect=30)
    fig.colorbar(ims, cax=cbar_ax, **kw);
#
#----------------------------------------------------------------------
def showdiff (Xref, Xest, resol, dim_dic, relatif=False, wdif="", cmap=CMAP_DEF,
              vdmin=None, vdmax=None) :
    # les diff�rence ne sont pas (forcement) dans les m�mes echelles que les donn�es
    # Une seule image � la fois
    #print("-- in showdiff --");
    origine = ORIGINE;
    if relatif == False : 
        if wdif=="" :
            Xdif = Xref - Xest; 
            plt.imshow(Xdif, interpolation='none', vmin=vdmin, vmax=vdmax,
                       cmap=cmap, origin=origine);
        elif wdif=='log' :
            Xdif = np.log(Xref) - np.log(Xest); 
            plt.imshow(Xdif, interpolation='none', vmin=vdmin, vmax=vdmax,
                       cmap=cmap, origin=origine);
        elif wdif=='dlog' :
            Xdif = np.log(Xref - Xest); 
            plt.imshow(Xdif, interpolation='none', vmin=vdmin, vmax=vdmax, 
                       cmap=cmap, origin=origine);
        elif wdif=='dalog' :
            Xdif = np.log(np.abs(Xref - Xest)) 
            plt.imshow(Xdif, interpolation='none', vmin=vdmin, vmax=vdmax,
                       cmap=cmap, origin=origine);
    else : # Assumed relatif == True : 
        if wdif=="" :
            Xdif = (Xref - Xest) / Xref; 
            plt.imshow(Xdif, interpolation='none', vmin=vdmin, vmax=vdmax,
                       cmap=cmap, origin=origine);
        elif wdif=='log' :
            lxref = np.log(Xref);
            Xdif = (lxref - np.log(Xest)) / lxref; 
            plt.imshow(Xdif, interpolation='none', vmin=vdmin, vmax=vdmax,
                       cmap=cmap, origin=origine);
        elif wdif=='dlog' :
            Xdif = np.log((Xref - Xest)/ Xref); 
            plt.imshow(Xdif, interpolation='none', vmin=vdmin, vmax=vdmax,
                       cmap=cmap, origin=origine);
        elif wdif=='dalog' :
            Xdif = np.log(np.abs((Xref - Xest)/ Xref)); 
            plt.imshow(Xdif, interpolation='none', vmin=vdmin, vmax=vdmax,
                       cmap=cmap, origin=origine);

    # Set coord geo as ticks ...
    NL_, NC_ = np.shape(Xref);
    xxticks, lxticks, yyticks, lyticks = getrticks_from_dic(dim_dic);
    plt.xticks(xxticks, lxticks); plt.xlabel("longitude");
    plt.yticks(yyticks, lyticks); plt.ylabel("latitude");
    titres  = "R%02d(%dx%d)"%(resol,NL_,NC_); # titre resolution
    plt.colorbar(orientation='horizontal');
    return titres;
#
#----------------------------------------------------------------------
def plotavar (Xi, resol, dim_dic, titre, wnorm, wmin, wmax, cmap=CMAP_DEF, calX0i=None, Xiref=None) :
    origine = ORIGINE;
    if wnorm=="Log" :       
        wbinf=np.log(wmin); wbsup=np.log(wmax);
        plt.imshow(np.log(Xi), vmin=wbinf, vmax=wbsup, interpolation='none', cmap=cmap, origin=origine);
        titre = titre+"Log ";
    elif wnorm=="log" : # sans indiquer vmin, vmax 
        plt.imshow(np.log(Xi), interpolation='none', cmap=cmap, origin=origine);
        titre = titre+"log ";
    elif wnorm=="LogNorm" : 
        plt.imshow(Xi, norm=LogNorm(vmin=wmin, vmax=wmax), interpolation='none', cmap=cmap, origin=origine);
        titre = titre+"LogNorm ";
    else : 
        plt.imshow(Xi, interpolation='none', cmap=cmap, vmin=wmin, vmax=wmax, origin=origine);
    #
    # Set coord geo as ticks ...
    NL_, NC_ = np.shape(Xi);
    xxticks, lxticks, yyticks, lyticks = getrticks_from_dic(dim_dic);
    plt.xticks(xxticks, lxticks); plt.xlabel("longitude");
    plt.yticks(yyticks, lyticks); plt.ylabel("latitude");
    titre = titre + "R%02d(%dx%d)"%(resol,NL_,NC_);
    #
    if calX0i is not None :
        titre = titre + " " + calX0i;
    if Xiref is not None :
        rmsi, Nnan, inan = nanrms(Xiref, Xi);    
        titre = titre + "\nrms=%.4e "%(rmsi);    
    plt.colorbar(orientation='horizontal');
    return titre;
#--------------------------------------------------
def showsome(X, resol, dim_dic, wmin=None,wmax=None,wnorm=None, cmap=CMAP_DEF,
             fsize=None, calX0=None, Xref=None, wdif=None, wdifr=None, varlib="",
             suptitre="", Xtit="Est.", vdmin=None, vdmax=None,
             figdir='.', savefig=False) :
    #print("-- in showsome --");
    if wmin==None :
        wmin = np.min(X);
    if wmax==None :
        wmax = np.max(X);
    n = len(X);
    if SUBP_ORIENT == "horiz" :
        if fsize is None :
            fsize=(16,12)
        subl=1; subc=n;
    else :
        if fsize is None :
            fsize=(12,16);
        subl=n; subc=1;
    
    plt.figure(figsize=fsize);
    for i in np.arange(n) : # VALEURS ESTIMEE
        titre = ""
        if Xtit is not None:
            titre += Xtit+" "
        titre += varlib+" ";
        plt.subplot(subl,subc,i+1);
        if Xref is not None :
            titre = plotavar (X[i], resol, dim_dic, titre, wnorm, wmin, wmax, cmap=cmap,
                              calX0i=calX0[i], Xiref=Xref[i]); 
        else :
            titre = plotavar (X[i], resol, dim_dic, titre, wnorm, wmin, wmax, cmap=cmap,
                              calX0i=calX0[i]); 
        plt.title(titre, fontsize=x_figtitlesize);
        #titre= "estimation"+varlib
    plt.suptitle(suptitre,fontsize=x_figsuptitsize);
    if savefig==True:
        printfilename = strconv_for_title(f"{suptitre}-Est")
        plt.savefig(os.path.join(figdir,f"{printfilename}.png"))
        
    if Xref is not None : # TRUE VALUE (valeurs de r�f�rence)
        plt.figure(figsize=fsize);
        for i in np.arange(n) :
            titre="Ref. : "+varlib+" ";
            plt.subplot(subl,subc,i+1);
            titre = plotavar (Xref[i], resol, dim_dic, titre, wnorm, wmin, wmax, cmap=cmap,
                              calX0i=calX0[i]);
            plt.title(titre, fontsize=x_figtitlesize);
        plt.suptitle(suptitre,fontsize=x_figsuptitsize);
        if savefig==True:
            printfilename = strconv_for_title(f"{suptitre}-Ref")
            plt.savefig(os.path.join(figdir,f"{printfilename}.png"))
    #
    #wdif : "": ref-est ; "log": log(ref)-log(est) ; "dlog": log(ref-est) ; "dalog" : log(abs(ref-est))
    #
    if Xref is not None : # DIFFERENCES ...
        if wdif is not None : #DIFFERENCES
            plt.figure(figsize=fsize);            
            for i in np.arange(n) :
                titre = "Dif. : "+varlib;
                plt.subplot(subl,subc,i+1);                       
                titres = showdiff(Xref[i], X[i], resol, dim_dic, relatif=False, wdif=wdif,
                                  cmap=cmap, vdmin=vdmin, vdmax=vdmax);
                titre  = titre + " " + wdif + " " + titres;
                if calX0 is not None :
                    titre = titre + " " + calX0[i];
                plt.title(titre, fontsize=x_figtitlesize);
            plt.suptitle(suptitre,fontsize=x_figsuptitsize);
            if savefig==True:
                printfilename = strconv_for_title(f"{suptitre}-Diff")
                plt.savefig(os.path.join(figdir,f"{printfilename}.png"))
        if wdifr is not None : #DIFFERENCES RELATIVES
            plt.figure(figsize=fsize);
            for i in np.arange(n) :
                titre = "Dif. Rel. : "+varlib;
                plt.subplot(subl,subc,i+1);           
                titres = showdiff(Xref[i], X[i], resol, dim_dic, relatif=True, wdif=wdifr,
                                  cmap=cmap, vdmin=vdmin, vdmax=vdmax);
                titre  = titre + " " + wdifr + " " + titres;                
                if calX0 is not None :
                    titre = titre + " " + calX0[i];
                plt.title(titre, fontsize=x_figtitlesize);
            plt.suptitle(suptitre,fontsize=x_figsuptitsize);
            if savefig==True:
                printfilename = strconv_for_title(f"{suptitre}-DiffRel")
                plt.savefig(os.path.join(figdir,f"{printfilename}.png"))
#----------------------------------------------------------------------
def showquivmask (Ui, Vi, resol, qscale=None, qmask=None, qmode=None) :
    print("-- in showquivmask --");
    Nlig, Ncol = np.shape(Ui); # Une seul image a la fois, m�me shape assumed
    UU_ = np.ones((Nlig, Ncol)) * np.nan;
    VV_ = np.ones((Nlig, Ncol)) * np.nan;
    iquv  = quvreso.index(resol);
    if qmask is None :
        qmask  = quvmask[iquv];
    if qscale is None :
        qscale = quvscale[iquv];
    if qmode is None :
        qmode  = quvmode[iquv];
    if qmode==1 : # 'step'
        qstep = qmask;
        UU_[0:Nlig:qstep, 0:Ncol:qstep] = Ui[0:Nlig:qstep, 0:Ncol:qstep];
        VV_[0:Nlig:qstep, 0:Ncol:qstep] = Vi[0:Nlig:qstep, 0:Ncol:qstep];
    else : # moyenne' assumed
        UM_ = makemoy(Ui.reshape(1,Nlig,Ncol),qmask,qmask);
        VM_ = makemoy(Vi.reshape(1,Nlig,Ncol),qmask,qmask);
        UU_[0:Nlig:qmask , 0:Ncol:qmask] = UM_[0];
        VV_[0:Nlig:qmask , 0:Ncol:qmask] = VM_[0];
    plt.quiver(UU_,VV_, scale=qscale);
#--------------------------------------------------
def showquivmaskdiff (Uiref, Viref, Uiest, Viest, resol, qscale=None,
                      qmask=None, qmode=None, relatif=False, wdif="") :
    if relatif==False :
        if wdif=="" :
            Uidif = Uiref - Uiest;
            Vidif = Viref - Viest;
        elif wdif=='log' :
            Uidif = np.log(Uiref) - np.log(Uiest);
            Vidif = np.log(Viref) - np.log(Viest);           
        elif wdif=='dlog' :
            Uidif = np.log(Uiref - Uiest);
            Vidif = np.log(Viref - Viest);
        elif wdif=='dalog' :
            Uidif = np.log(np.abs(Uiref - Uiest));
            Vidif = np.log(np.abs(Viref - Viest));
    else : # Assumed relatif == True : 
        if wdif=="" :
            Uidif = (Uiref - Uiest) / Uiref;
            Vidif = (Viref - Viest) / Uiest;
        elif wdif=='log' :
            luref = np.log(Uiref);     lvref = np.log(Viref);
            Uidif  = (luref - np.log(Uiest)) / luref;
            Vidif  = (lvref - np.log(Viest)) / lvref;           
        elif wdif=='dlog' :
            Uidif = np.log((Uiref - Uiest) / Uiref);
            Vidif = np.log((Viref - Viest) / Viref);
        elif wdif=='dalog' :
            Uidif = np.log(np.abs((Uiref - Uiest) / Uiref));
            Vidif = np.log(np.abs((Viref - Viest) / Viref));
    showquivmask(Uidif, Vidif, resol, qscale=qscale, qmask=qmask, qmode=qmode);              
#--------------------------------------------------
def showxquiv(X, U, V, resol, dim_dic, qscale=None, qmask=None, qmode=None, xvmin=None,
              xvmax=None, xvnorm=None, cmap=CMAP_DEF, fsize=None, calX0=None, 
              Xref=None, Uref=None, Vref=None, wdif=None, wdifr=None, suptitre="",
              figdir='.', savefig=False) :
    print("-- in showxquiv --");
    # qmode : 1: 'step sinon 'moyenne'
    # qmask : Si qmode!=1, en attendant de revoir la fonction makemoy,
    #         ce doit etre un diviseur de Nlig et Ncol - not checked
    Nimg, Nlig, Ncol = np.shape(X); # same for U et V not checked
    xxticks, lxticks, yyticks, lyticks = getrticks_from_dic(dim_dic);
    origine = ORIGINE;
    #
    if SUBP_ORIENT == "horiz" :
        if fsize is None:
            fsize=(20,9)
        subl=1; subc=Nimg;
    else :
        if fsize is None:
            fsize=(9, 18);
        subl=Nimg; subc=1;
    #    
    # VALEURS ESTIMEES (h+quiv(uv))
    plt.figure(figsize=fsize);
    for i in np.arange(Nimg) :
        titre="Est. ";
        plt.subplot(subl,subc,i+1);
        plt.imshow(X[i], interpolation='none', cmap=cmap, vmin=xvmin, vmax=xvmax, origin=origine);
        plt.colorbar(orientation='horizontal');
        showquivmask(U[i], V[i], resol, qscale=qscale, qmask=qmask, qmode=qmode);        
        if 1 : # Avoir d�finit les coordonn�es g�ographique par r�solution une fois pour toute
            plt.xticks(xxticks, lxticks); plt.xlabel("longitude");
            plt.yticks(yyticks, lyticks); plt.ylabel("latitude");
            titre = titre + "R%02d(%dx%d)"%(resol,Nlig,Ncol);
        if calX0 is not None :
            titre = titre + " " + calX0[i];
        #titre="estimation"+varlue[i]+"R%02d(%dx%d)"%(resol,Nlig,Ncol)
        if Xref is not None : # On suppose que c'est idem pour Uref et Vref
            RMSX, Nnan, inan = nanrms(Xref[i], X[i]);
            RMSU, Nnan, inan = nanrms(Uref[i], U[i]);
            RMSV, Nnan, inan = nanrms(Vref[i], V[i]);
            plt.title(titre + " rmsH=%.4e, rmsU=%.4e, rmsV=%.4e"%(RMSX,RMSU,RMSV),
                      fontsize=x_figtitlesize);
        else:
            #plt.title(titre + " rmsH=%.4e, rmsU=%.4e, rmsV=%.4e"%(RMSX,RMSU,RMSV),
            plt.title(titre,
                      fontsize=x_figtitlesize);
        if savefig==True:
            printfilename = strconv_for_title(f"{suptitre}-Est")
            plt.savefig(os.path.join(figdir,f"{printfilename}.png"))
        
    plt.suptitle(suptitre,fontsize=x_figsuptitsize);
    if Xref is not None : # On suppose que c'est idem pour Uref et Vref
        # VALEUR de REFERENCE (True) (h+quiv(uv))
        plt.figure(figsize=fsize);
        for i in np.arange(Nimg) :
            titre="Ref. ";
            plt.subplot(subl,subc,i+1);
            plt.imshow(Xref[i], interpolation='none', cmap=cmap, vmin=xvmin, vmax=xvmax, origin=origine);
            plt.colorbar(orientation='horizontal');
            showquivmask(Uref[i], Vref[i], resol, qscale=qscale, qmask=qmask, qmode=qmode);

            # Set coord geo as ticks ...
            plt.xticks(xxticks, lxticks); plt.xlabel("longitude");
            plt.yticks(yyticks, lyticks); plt.ylabel("latitude");
            titre = titre + "R%02d(%dx%d)"%(resol,Nlig,Ncol);
                
            if calX0 is not None :
                titre = titre + " " + calX0[i];
            plt.title(titre, fontsize=x_figtitlesize);
        plt.suptitle(suptitre,fontsize=x_figsuptitsize);
        if savefig==True:
            printfilename = strconv_for_title(f"{suptitre}-Ref")
            plt.savefig(os.path.join(figdir,f"{printfilename}.png"))
        #
        if wdif is not None :   # DIFFERENCES (h+quiv(uv))
            ivar_ = tvwmm.index("SSH");
            vdmin = wdmin[ivar_];
            vdmax = wdmax[ivar_];
            plt.figure(figsize=fsize);
            for i in np.arange(Nimg) :
                titre="Dif. ";
                plt.subplot(subl,subc,i+1);
                titres = showdiff(Xref[i], X[i], resol, dim_dic, cmap=cmap, relatif=False,
                                  wdif=wdif, vdmin=vdmin, vdmax=vdmax);
                showquivmaskdiff (Uref[i], Vref[i], U[i], V[i], resol,
                          qscale=qscale, qmask=qmask, qmode=qmode, relatif=False, wdif=wdif);
                titre = titre + " " + titres;
                if calX0 is not None :
                    titre = titre + " " + calX0[i];
                plt.title(titre, fontsize=x_figtitlesize);
            plt.suptitle(suptitre,fontsize=x_figsuptitsize);
            if savefig==True:
                printfilename = strconv_for_title(f"{suptitre}-Diff")
                plt.savefig(os.path.join(figdir,f"{printfilename}.png"))

        if wdifr is not None :  # DIFFERENCES RELATIVES (h+quiv(uv))
            plt.figure(figsize=fsize);
            for i in np.arange(Nimg) :
                titre="Dif. Rel. SSH +(U,V) ";
                plt.subplot(subl,subc,i+1);
                titres = showdiff(Xref[i], X[i], resol, dim_dic, cmap=cmap, relatif=True,
                                  wdif=wdifr, vdmin=vdmin, vdmax=vdmax);
                showquivmaskdiff (Uref[i], Vref[i], U[i], V[i], resol, qscale=qscale,
                                  qmask=qmask, qmode=qmode, relatif=True, wdif=wdifr);
                titre = titre + " " + titres;
                if calX0 is not None :
                    titre = titre + " " + calX0[i];
                plt.title(titre, fontsize=x_figtitlesize);
            plt.suptitle(suptitre,fontsize=x_figsuptitsize);
            if savefig==True:
                printfilename = strconv_for_title(f"{suptitre}-DiffRel")
                plt.savefig(os.path.join(figdir,f"{printfilename}.png"))
#--------------------------------------------------
def showhquiv(Voutb, Resolst, D_dicolst, im2show, varbg="SSH", varu="U", varv="V",
              qscale=None, qmask=None, qmode=None, calX0=None, Yref=None, 
              wdif=None, wdifr=None, fsize=(20,9), suptitre="",
              figdir='.', savefig=False) : 
    #print("-- in showhquiv --");
    if calX0 is not None :
        calX0_ = calX0[im2show];
    iu_ = varOut.index(varu);
    iv_ = varOut.index(varv);
    ih_ = len(varOut) - 1 - varOut[::-1].index(varbg) # le dernier varbg (en charchant le premier de la liste inversée!)
    #ih_ = varOut.index(varbg,np.min([iu_,iv_])-1); # le dernier h avant u et v        
    resol = Resolst[iu_]
    dim_dic = D_dicolst[iu_]
    if resol != Resolst[iv_] or resol != Resolst[ih_] :
        raise ValueError('resultuv: {varu} and {varv} or {varu} and {varbg} doesn''t have same resolution');
    H_  = Voutb[ih_][im2show,0,:,:];
    U_  = Voutb[iu_][im2show,0,:,:];
    V_  = Voutb[iv_][im2show,0,:,:];
    wk_ = tvwmm.index(varOut[ih_]);
    Href_=None; Uref_=None; Vref_=None;
    if Yref is not None :
        Href_ = Yref[ih_][im2show,0,:,:];
        Uref_ = Yref[iu_][im2show,0,:,:];
        Vref_ = Yref[iv_][im2show,0,:,:];        
    showxquiv(H_, U_, V_, resol, dim_dic, qscale=qscale, qmask=qmask, qmode=qmode, 
              xvmin=wbmin[wk_], xvmax=wbmax[wk_], xvnorm=wnorm[wk_], cmap=wcmap[wk_],
              fsize=fsize, calX0=calX0_, Xref=Href_, Uref=Uref_, Vref=Vref_,
              wdif=wdif, wdifr=wdifr, suptitre=suptitre, figdir=figdir, savefig=savefig);
    return ih_;
#----------------------------------------------------------------------
def rms(Xref, Xest) :
    '''Calcule et retourne la RMS entre Xref et Xest
    '''
    return np.sqrt(np.sum((Xref-Xest)**2) / np.prod(np.shape(Xref)));
def rmsrel(Xref, Xest) :
    '''Calcule et retourne la RMS relative entre Xref et Xest '''
    Nall   = np.prod(np.shape(Xref));
    Errrel = (Xref - Xest) / Xref;   
    Errrel = np.sqrt(np.sum(Errrel**2)/Nall); 
    return Errrel;
def nanrms(X, Y) :
    Nitem_ = np.prod(np.shape(X))
    X_     = np.reshape(X, Nitem_);
    Y_     = np.reshape(Y, Nitem_);
    ixnan  = np.where(np.isnan(X_))[0];
    iynan  = np.where(np.isnan(Y_))[0];
    inan   = np.union1d(ixnan, iynan);
    Nnan   = len(inan)
    Nitem_ = Nitem_- Nnan;       
    RMS    = np.sqrt(np.nansum((X_-Y_)**2) / Nitem_ );
    return RMS, Nnan, inan
#----------------------------------------------------------------------
def makemoy(XB, ml=3, mc=3) :
    N,nl,nc = np.shape(XB);
    xm      = np.zeros((N,nl//ml,nc//mc));
    ii = 0;
    for i in np.arange(0,nl,ml) :
        jj = 0;
        for j in np.arange(0,nc,mc) :
            if 0 :
                moy = np.mean(XB[:,i:i+ml,j:j+mc]);
                xm[0,ii,jj] = moy; 
            else :              
                moy = np.mean(XB[:,i:i+ml,j:j+mc], axis=(1,2));
                xm[:,ii,jj] = moy;
            jj=jj+1;
        ii=ii+1;
    return xm
#-------------------------------------------------------------
def isetalea (Nimg, pcentSet) :
    pcentA, pcentV, pcentT = pcentSet;
    Ialea = np.arange(Nimg);
    np.random.seed(0); # Pour reproduire la m�me chose � chaque fois
    np.random.shuffle(Ialea);
    indA = Ialea[0:int(Nimg*pcentA)];
    indV = Ialea[int(Nimg*pcentA): int(Nimg*(pcentA+pcentV))];
    indT = Ialea[int(Nimg*(pcentA+pcentV)): int(Nimg*(pcentA+pcentV+pcentT))];
    return indA, indV, indT;    
#----------------------------------------
def splitset (Vin_brute, Vout_brute, pcentSet) :
    Nimg_ = len(Vin_brute[0]); # == len(Vou_brute[0]); not checked
    indA, indV, indT = isetalea(Nimg_, pcentSet);
    #
    VAout_brute = []; VVout_brute = []; VTout_brute = [];
    for i in np.arange(len(Vout_brute)) : # Pour chaque variable (i.e. liste)
        VAout_brute.append(Vout_brute[i][tuple([indA])]);               
        VVout_brute.append(Vout_brute[i][tuple([indV])]);               
        VTout_brute.append(Vout_brute[i][tuple([indT])]);               
    VAin_brute = []; VVin_brute = []; VTin_brute = [];
    for i in np.arange(len(Vin_brute)) : # Pour chaque variable (i.e. liste)
        VAin_brute.append(Vin_brute[i][tuple([indA])]);               
        VVin_brute.append(Vin_brute[i][tuple([indV])]);               
        VTin_brute.append(Vin_brute[i][tuple([indT])]);
    return VAin_brute, VAout_brute, VVin_brute, VVout_brute, VTin_brute, VTout_brute;
#-------------------------------------------------------------
def setresolution(VA_brute,VV_brute,VT_brute,varlue,ResoIn,ResoOut,
                  native_resol=1) :           
    # Make resolution for IN and OUT
    print("... making V*out_Brute");
    VAout_brute = []; VVout_brute = []; VTout_brute = [];
    for i in np.arange(NvarOut) : #varOut ['SSH', 'SSH', 'U', 'V']
        reso_rel = ResoOut[i] // native_resol
        idvar = varlue.index(varOut[i])
        dvar  = VA_brute[idvar];
        if reso_rel > 1 :
            dvar  = makemoy(dvar, reso_rel, reso_rel);
        VAout_brute.append(dvar);            
        dvar  = VV_brute[idvar];
        if reso_rel > 1 :
            dvar  = makemoy(dvar, reso_rel, reso_rel);
        VVout_brute.append(dvar);            
        dvar  = VT_brute[idvar];
        if reso_rel > 1 :
            dvar  = makemoy(dvar, reso_rel, reso_rel);
        VTout_brute.append(dvar);            
    print("... making V*in_Brute");
    VAin_brute = []; VVin_brute = []; VTin_brute = [];
    for i in np.arange(NvarIn) : #varIn['SSH', 'SST', 'SST', 'SST']
        reso_rel = ResoIn[i] // native_resol
        idvar = varlue.index(varIn[i])
        dvar  = VA_brute[idvar];
        if reso_rel > 1 :
            dvar  = makemoy(dvar, reso_rel, reso_rel);
        VAin_brute.append(dvar);            
        dvar  = VV_brute[idvar];
        if reso_rel > 1 :
            dvar  = makemoy(dvar, reso_rel, reso_rel);
        VVin_brute.append(dvar);            
        dvar  = VT_brute[idvar];
        if reso_rel > 1 :
            dvar  = makemoy(dvar, reso_rel, reso_rel);
        VTin_brute.append(dvar);
    return VAout_brute, VVout_brute, VTout_brute, VAin_brute, VVin_brute, VTin_brute;
#-------------------------------------------------------------
def data_repartition(V_brute, couple_var_reso_list, var_list, reso_list, indA, indV, indT) :
    # Make resolution for IN and OUT
    print("... making V*out_Brute");
    VA_brute = []; VV_brute = []; VT_brute = [];
    for v,r in zip(var_list,reso_list) : #varOut ['SSH', 'SSH', 'U', 'V']
        idvar = couple_var_reso_list.index((v,r))
        print(v,r,idvar)
        VA_brute.append(V_brute[idvar][indA,:])
        VV_brute.append(V_brute[idvar][indV,:])
        VT_brute.append(V_brute[idvar][indT,:])
    return VA_brute, VV_brute, VT_brute
#-------------------------------------------------------------
def dic_dimension_repartition(D_dico_list, couple_var_reso_list, var_list, reso_list,
                              isets=None, n_var='time') :
    # Make resolution for IN and OUT
    DAselect_dico_list = [];
    if isets is not None:
        DVselect_dico_list = [];
        DTselect_dico_list = [];
    for v,r in zip(var_list, reso_list) :
        idvar = couple_var_reso_list.index((v,r))
        tmpA_dico = D_dico_list[idvar].copy()
        if isets is None:
            DAselect_dico_list.append(tmpA_dico)
        else:
            ia, iv, it = isets
            tmpV_dico = tmpA_dico.copy()
            tmpT_dico = tmpA_dico.copy()
            # APP
            tmpA_dico[n_var] = tmpA_dico[n_var][ia]
            DAselect_dico_list.append(tmpA_dico)
            # VAL
            tmpV_dico[n_var] = tmpV_dico[n_var][iv]
            DVselect_dico_list.append(tmpV_dico)
            # TEST
            tmpT_dico[n_var] = tmpT_dico[n_var][it]
            DTselect_dico_list.append(tmpT_dico)
    if isets is None:
        return DAselect_dico_list
    else:
        return DAselect_dico_list, DVselect_dico_list, DTselect_dico_list
#-------------------------------------------------------------
def statibase(Xi, fmt=None) : # stat de base
    if fmt is None : 
        print("min=%.4f, max=%.4f, moy=%.4f, std=%.4f"
                  %(np.min(Xi), np.max(Xi), np.mean(Xi), np.std(Xi)));            
    else : 
        print("min=%.4e, max=%.4e, moy=%.4e, std=%.4e"
                  %(np.min(Xi), np.max(Xi), np.mean(Xi), np.std(Xi)));            
def stat2base (X, nmvar) : 
    for i in np.arange(len(nmvar)) :           
        print("%s : "%nmvar[i], end=''); statibase(X[i])
#-------------------------------------------------------------
def linr2 (x, y) :
    ''' b0,b1,s,R2,sigb0,sigb1 = .linreg(x,y)
    | Calcule la r�gression lin�aire de x par rapport a y.
    | En sortie :
    | b0, b1 : Coefficients de la droite de regression lineaire : y=b0+b1*x
    | R2     : Coefficient de d�termination
    '''
    N       = x.size;
    xmean   = np.mean(x);
    xmean2  = xmean*xmean;
    xcentre = x-xmean;
    ymean   = np.mean(y);
    ycentre = y-ymean;
    b1 = np.sum(ycentre*xcentre) / (np.sum(x*x) - N*xmean2);
    b0 = ymean - b1*xmean;  
    yc = b0 + b1*x;
    R2 = np.sum(pow((yc-ymean),2))/np.sum(pow(ycentre,2));  
    return b0,b1,R2
#--------------------------------------------------
def histodbl(Xref, Xest, legende, nbins=NBINS) :
    # Histogrammes double (supperpos�s) pour comparaison 
    mmin = np.min([Xref, Xest]); 
    mmax = np.max([Xref, Xest]); 
    epsi  = abs(mmax)/100000;
    mmaxX = mmax + epsi; 
    bins  = np.arange(mmin, mmaxX, (mmax-mmin)/nbins);
    bins[-1] = bins[-1] + epsi;
    plt.figure();
    plt.hist(Xref, bins=bins);
    plt.hist(Xest, bins=bins, color=[0.60,1.00,0.60,0.7]);        
    plt.legend(legende, framealpha=0.5);
    return
#--------------------------------------------------
def scatplot (Xref, Xest, ident=True, regr2=True, mksz=3.0, alpha=SCATALPHA, 
              linewidthident=1.0, linewidthregr=2.0,
              fsize=(10,10), squareax=False):
    # Scatter plot
    mpl.rcParams['agg.path.chunksize'] = 1000000;
    f, ax = plt.subplots(figsize=fsize)
    #plt.figure(figsize=fsize);
    #plt.plot(Xref, Xest, '.g', markersize=mksz); 
    ax.scatter(Xref, Xest, marker='.', color='g', s=mksz, alpha=alpha); 
    plt.xlabel("NATL60"); plt.ylabel("RESAC");
    xymin=np.nanmin([*Xref,*Xest])  # min et
    xymax=np.nanmax([*Xref,*Xest])  # max absoluts pour ref et est
    lax = ax.axis()
    if squareax :
        #plt.axis('square')
        plt.axis('scaled')
    if ident : # ligne de l'identit�
        if False:
            plt.plot([xymin, xymax], [xymin, xymax], '-m',linewidth=linewidthident);
        else:
            ax.plot([0, 1], [0, 1], color='m',linewidth=linewidthident,
                    transform=ax.transAxes)
        lax = [xymin, xymax, xymin, xymax]
    if regr2 : # r�gression + R2
        b0, b1, R2 = linr2(Xref, Xest);
        xr = np.arange(xymin, xymax,(xymax-xymin)/200); #[xmin:(xmax-xmin)/200:xmax]';  
        yr = b1*xr + b0;
        plt.plot(xr, yr, '-b',linewidth=linewidthregr);
        #plt.legend(["", "id", "rl"], numpoints=1, loc='upper left',framealpha=0.5);
        plt.legend(labels=SCATLABELS, numpoints=1, loc='upper left',framealpha=0.5);
        ax.axis(lax) # pour preserver les limites des axes
        #
        return b0, b1, R2;
    ax.axis(lax) # pour preserver les limites des axes
    #
    return
#--------------------------------------------------
def setbins(mmax, mmin, nbins) :
    epsi  = abs(mmax)/100000;
    mmaxX = mmax + epsi; 
    bins  = np.arange(mmin, mmaxX, (mmax-mmin)/nbins);
    bins[-1] = bins[-1] + epsi;
    return bins
#
# def getrticks (Nlig) :
#     # get r�solution et ticks
#     reso    = NLigR01//Nlig; #=NColR01/NC_    (division entiere en Python 3)
#     icoord  = ResoAll.index(reso);
#     xxticks, lxticks, yyticks, lyticks = CoordGeo[icoord];
#     return reso, xxticks, lxticks, yyticks, lyticks
#
def getrticks_from_dic(dim_dic,nx=9,ny=8):
    xxticks = np.linspace(0,len(dim_dic['lon'])-1,nx)
    yyticks = np.linspace(0,len(dim_dic['lat'])-1,ny)
    lxticks = [ f"{c:.1f}" for c in np.linspace(dim_dic['lon'][0],dim_dic['lon'][-1],nx)]
    lyticks = [ f"{c:.1f}" for c in np.linspace(dim_dic['lat'][0],dim_dic['lat'][-1],ny)]
    return xxticks, lxticks, yyticks, lyticks
#
def strconv_for_title(str, upper=False, lower=False, capitalize=False) :
    #print(f"\nstrconv_for_title ... '{str}' ...... ", end='')
    # regres de subtitution de carateres qui peuvent etre genants dans le nom d'un fichier
    str = str.replace('"', '').replace("'", '')
    str = str.replace(':', ' ').replace(';', ' ').replace(',', ' ')
    str = str.replace('/', '-').replace('(', '_').replace(')', '_').replace('&', '-')
    str = str.replace('[', '_').replace(']', '_').replace('{', '_').replace('}', '_')
    # substitution de certains caracteres qui peuvent etre en exces
    str = str.replace('.-','-').replace('._', '_').replace('. ', ' ').replace('\n','-')
    str = str.replace('  ', ' ').replace('  ', ' ').replace(' ', '_')
    str = str.replace('- ', '-').replace(' -', '-').replace('_ ', '_').replace(' _', '_')
    str = str.replace('--', '-').replace('--', '-').replace('__', '_').replace('__', '_')
    str = str.replace('_-','-').replace('-_', '-')
    str = str.replace('-.','.').replace('_.', '.').replace(' .', '.')
    if str[0] in [' '] : # enleve blanc si en premier
        str = str[1:]
    if str[-1] in [' ','_','-'] : # enleve blanc si en dernier
        str = str[:-1]
    if lower:
        str = str.lower()
    elif upper:
        str = str.upper()
    elif capitalize:
        str = str.capitalize()
    #print(f" '{str}'")
    return str
#
def nl2limlat (Nlig, limlat) :
    # Nombre de ligne jusqu'� une latitude limite
    #lonfr=40.; lonto=65.;latfr=26.; latto=45.; # Coordonn�e :26�N, 45�N; 40�W, 65�W
    Nlat   = latto-latfr+1;              # =45-26+1    = 20.0
    Nllat  = Nlig / Nlat;                # =144 / Nlat = 7.20 : nombre de ligne par lattitude
                                         # v�rif : (45 - 26 +1) * 7.20 = 144
    #limlat = 35;                        # La limite de latitude choisie EN DURE
    NL_lim = int((limlat-latfr+1)*Nllat);# =(35-26+1)*7.20 = 72 : nombre de lignes jusqu'� la lattitude limite
    return NL_lim
#
def splitns (X, NL_lim) :
    # Split Nord-Sud
    if len(np.shape(X))==3 : # Cas Enstrophie
        pipo, Nlig, pipo = np.shape(X);
        XS = X[:,0:NL_lim,:];
        XN = X[:,NL_lim:Nlig,:];
    else : # assume==4 (cas Energie)
        pipo, pipo, Nlig, pipo = np.shape(X);
        XS = X[:,:,0:NL_lim,:];
        XN = X[:,:,NL_lim:Nlig,:];
    return XN, XS;
#--------------------------------------------------
def result(lib, strset, varname, Yref, Yest, resol, flag_rms=0, flag_scat=0,
           flag_histo=0, nbins=NBINS, calX=None, fsizerms=(12,8), szxlblrms=8, 
           fsizehisto=(12,8), 
           fsizescat=(10,10), squarescatax=False, dpiscat=180,
           figdir='.', savefig=False) :
    print("-- in result --"); #print(np.shape(Yref)); #(55L, 1L, 144L, 153L)
    # Au moins toujours la RMS qui est affich�e par ailleurs
    Nall_ = np.prod(np.shape(Yest));
    N_ = len(Yref); Allrmsi = [];
    for i in np.arange(N_) :
        rmsi, Nnan, inan = nanrms(Yref[i], Yest[i]);
        Allrmsi.append(rmsi);
    moyAllrmsi = np.mean(Allrmsi);
    titres_ = "%s %s %s R%02d - RMS by image (%d pixels), Mean : %.4f\n min=%.4f, max=%.4f, std=%.4f" \
               %(strset, varname, lib, resol, Nall_, moyAllrmsi, np.min(Allrmsi), np.max(Allrmsi), np.std(Allrmsi));
    nom_save="%s_%s_%s_R%02d_RMS_BY_IMAGE_(%d pixels)"%(strset, varname, lib, resol, Nall_)
    print(titres_);
    #
    if flag_rms and FIGBYIMGS : # figure RMS par image
        f,ax = plt.subplots(nrows=1, ncols=1,figsize=fsizerms)
        plt.subplots_adjust(left=0.06, bottom=0.105, right=0.99, top=0.915) #, wspace=0.2, hspace=0)

        if calX is None : # plot non tri� sur la date
            y_val = Allrmsi
            #plt.figure(figsize=fsizerms)
            plt.plot(y_val, '-*');
            #plt.xticks(np.arange(N_), calX[0], rotation=35, ha='right',fontsize=szxlblrms);
        else : # plot tri� sur la date
            ids = np.argsort(calX[1]); # indice de tri des dates  
            y_val = list(np.array(Allrmsi)[ids])
            x_tick_labels = [f'{d} ({ids[i]})' for i,d in enumerate(calX[0][ids])]         
            #plt.figure(figsize=fsizerms);
            plt.plot(y_val, '-*');
            plt.xticks(np.arange(N_),x_tick_labels, rotation=35, ha='right',fontsize=szxlblrms);
            #horizontalalignment='right', verticalalignment='baseline')          
        plt.plot([0,N_-1],[moyAllrmsi, moyAllrmsi]); # Trait de la moyenne
        y_min,y_max = min(y_val),max(y_val)
        i_min,i_max = y_val.index(y_min),y_val.index(y_max)
        plt.axvline(x=i_min,color='blue',linewidth=1,linestyle=':')
        plt.axvline(x=i_max,color='red',linewidth=1,linestyle=':')
        if calX is not None : # plot non tri� sur la date
            lax = plt.axis()
            delta_y = (lax[3]-lax[2])/100
            plt.text(i_min,y_min-delta_y,f"{x_tick_labels[i_min]}",size='small',ha='center',va='top')
            plt.text(i_max,y_max+delta_y,f"{x_tick_labels[i_max]}",size='x-small',ha='center',va='bottom')
        plt.title(titres_, fontsize=x_figtitlesize);
        if savefig==True:
            printfilename = strconv_for_title(nom_save)
            plt.savefig(os.path.join(figdir,f"{printfilename}.png"))
    #
    Yref_ = Yref.ravel();
    Yest_ = Yest.ravel();
    #                  
    if flag_scat :     # Scatter plot
        pipo, pipo, R2 = scatplot (Yref_, Yest_, fsize=fsizescat, squareax=squarescatax);
        plt.title("%s %s %s R%02d - Scatter plot (%d pixels)\n mean(rms)=%.6f, R2=%f"
                  %(strset, varname, lib, resol, Nall_, moyAllrmsi, R2), fontsize=x_figtitlesize);
        nom_save2="%s_%s_%s_R%02d_scatter_plot_(%d_pixels)"%(strset, varname, lib, resol, Nall_)
        if savefig==True:
            printfilename = strconv_for_title(f"{nom_save2}scatt")
            plt.savefig(os.path.join(figdir,f"{printfilename}.png"))
    #
    if flag_histo : # Histogramme des diff�rences
        Ydif_ = Yref_-Yest_;
        wk_   = tvwmm.index(varname);
        # Positionnement des min et max d'erreur (Forcage des bins sur EXP1)
        mmin = whdmin[wk_];  mmax = whdmax[wk_]; 
        bins = setbins(mmax, mmin, nbins);
        
        if 0 : # Histo non Normalis�
            plt.figure(figsize=fsizehisto);
            Nbmax_= whnmax[wk_];
            BI = plt.hist(Ydif_, bins=bins); #print("max(BI[0])=%d"%(np.max(BI[0])))
            plt.axis([mmin, mmax, 0, Nbmax_]);
            plt.title("%s %s %s R%02d - Histo des differences (Ref.-Est.)\n(%dbins)(%d pixels(%dnans) %dloss), mean(rms)=%.6f"
                      %(resol_strset, varname, lib, nbins, Nall_, Nnan, Nall_-sum(BI[0]), moyAllrmsi), fontsize=x_figtitlesize);
            nom_save3="%s_%s_%s_R%02d_Histo_des_differences"%(strset, varname, lib, resol)
            if savefig==True:
                printfilename = strconv_for_title(f"{nom_save3}histo_non_norm")
                plt.savefig(os.path.join(figdir,f"{printfilename}.png"))
        #
        # Histogramme Normalis�
        plt.figure(figsize=fsizehisto);
        #strset may be : "APP", "TEST", "APP-Nord" ou "APP-Sud" ou "TEST-Nord" ou "TEST-Sud"
        if len(strset)>4 :          # Pour savoir si on est dans un cas Nord-Sud ou pas, on
            NBmax_ = whSmax[wk_];   # utilise un test sur la longueur de strset ; ok c'est pas
        else :                      # top ; p'tet qu'un jour on c�era un param�tre d�di� ...
            NBmax_ = whNmax[wk_];  
        BI = plt.hist(Ydif_, bins=bins,density=True)#, normed=True);
        dbin   = bins[1] - bins[0]; 
        dhbin  = dbin / 2; 
        Centre = bins[0:nbins]+dhbin;
        ymoy   = np.mean(Ydif_);
        ystd   = np.std(Ydif_);
        y      = norm.pdf(Centre, ymoy, ystd);
        plt.plot(Centre, y, '-*r', linewidth=3);
        plt.axis([mmin, mmax, 0, NBmax_]);
        plt.title("%s %s %s R%02d - Histo Normalise des Diff.(Ref.-Est.)\n mean=%f, std=%f, %d pixels, mean(rms)=%.6f"
                  %(strset, varname, lib, resol, np.mean(Ydif_), np.std(Ydif_), Nall_, moyAllrmsi));
        nom_save4="%s_%s_%s_R%02d_Histo_Normalise_des_Diff"%(strset, varname, lib, resol)
        if savefig==True:
            printfilename = strconv_for_title(f"{nom_save4}histo_norm")
            plt.savefig(os.path.join(figdir,f"{printfilename}.png"))
#
#----------------------------------------------------------------------      
def distcosine2D (Wref, West) : # Matrix way
    WW    = np.sum(Wref*West, axis=1);   # les produits scalaires <wref, west>
    Nwref = np.sqrt(np.sum(Wref**2, 1)); # Normes des wref
    Nwest = np.sqrt(np.sum(West**2, 1)); # Normes des west
    costheta  = (WW/(Nwref*Nwest));
    moycosine = np.mean(costheta);
    return 1 - moycosine;
def dcosine2D(Uref,Vref,Uest,Vest) :
    Nall = np.prod(np.shape(Uref)); #idem for others not cheked
    Wref = np.array([Uref.reshape(Nall), Vref.reshape(Nall)]);
    West = np.array([Uest.reshape(Nall), Vest.reshape(Nall)]);
    return distcosine2D(Wref, West);
#----------------------------------------------------------------------
def enstrophie2d (U, V, dx, dy) :
    # U, V de la forme (Nimg, Nlig, Ncol) (16L, 144L, 153L)
    dedx = dx*2; dedy = dy*2;
    lenshape = len(np.shape(U));
    if lenshape == 3 :
        Nimg, Nlig, Ncol = np.shape(U); # same for V; not checked
        U_= U; V_= V;
    elif lenshape == 4 :
        Nimg, Ncan, Nlig, Ncol = np.shape(U); # same for V; not checked
        U_ = U.reshape(Nimg, Nlig, Ncol);
        V_ = V.reshape(Nimg, Nlig, Ncol);
        
    axX = np.arange(Ncol-2)+1;
    axY = np.arange(Nlig-2)+1;
    Enst= np.zeros((Nimg, Nlig, Ncol));
    for l in axY : # lig
        for c in axX : # col          
             dvdxdudy    = ((V_[:,l,c+1] - V_[:,l,c-1]) / dedx) \
                         - ((U_[:,l-1,c] - U_[:,l+1,c]) / dedy);
             Enst[:,l,c] = (dvdxdudy**2) / 2;             
    Enst = Enst[:, 1:Nlig-1, 1:Ncol-1]; 
    return Enst;
def divergence2d (U, V, dx, dy) :
    # U, V de la forme (Nimg, Nlig, Ncol) (16L, 144L, 153L)
    dedx = dx*2; dedy = dy*2;
    lenshape = len(np.shape(U));
    if lenshape == 3 :
        Nimg, Nlig, Ncol = np.shape(U); # same for V; not checked
        U_= U; V_= V;
    elif lenshape == 4 :
        Nimg, Ncan, Nlig, Ncol = np.shape(U); # same for V; not checked
        U_ = U.reshape(Nimg, Nlig, Ncol);
        V_ = V.reshape(Nimg, Nlig, Ncol);
        
    axX = np.arange(Ncol-2)+1;
    axY = np.arange(Nlig-2)+1;
    Div = np.zeros((Nimg, Nlig, Ncol));
    for l in axY : # lig
        for c in axX : # col            
             dudxdvdy    = ((U_[:,l,c+1] - U_[:,l,c-1]) / dedx) \
                         + ((V_[:,l-1,c] - V_[:,l+1,c]) / dedy);            
             Div[:,l,c] = (dudxdvdy**2) / 2;             
    Div = Div[:, 1:Nlig-1, 1:Ncol-1];
    return Div;
#----------------------------------------------------------------------
def phistuff (phy_ref, phy_est, resol, dim_dic, varname, varlib, strset, calX, wdif,
              flag_histo, flag_scat, nbins=NBINS, 
              fsizescat=(10,10), squarescatax=False, figdir='.', savefig=False) :
    print("-- in phistuff --");       
    if len(np.shape(phy_ref))==4 : # cas energie
         Nimg_, Ncan_, Nlig_, Ncol_ = np.shape(phy_ref);
         phi_ref = phy_ref.reshape(Nimg_, Nlig_, Ncol_); 
         phi_est = phy_est.reshape(Nimg_, Nlig_, Ncol_);
    else : # assumed==3 : cas enstrophie
         Nimg_, Nlig_, Ncol_ = np.shape(phy_ref);
         phi_ref = phy_ref 
         phi_est = phy_est    
    #
    wk_   = tvwmm.index(varname);
    #
    if strset[0]=='A' : #"APP" :
        im2show = im2showA;
    elif strset[0]=='T'  : #"TEST" :
        im2show = im2showT;
    else :
        raise ValueError("salperlipopette");
    #
    if not FIGBYPASS : 
        suptitre = "Some %s %s %s"%(strset,varlib,im2show);
        ivar_ = tvwmm.index(varname);
        vdmin = wdmin[ivar_];
        vdmax = wdmax[ivar_];
        if len(im2show) and resol in ResoAll :
            # ps : dans le cas Nord-Sud resol ne sera pas dans ResoAll
            #      et on ne devrait pas avoir besoin des images dans ce cas
            showsome(phi_est[im2show], resol, dim_dic, wmin=wbmin[wk_], wmax=wbmax[wk_],
                     wnorm=wnorm[wk_], cmap=wcmap[wk_], calX0=calX[0][im2show], Xref=phi_ref[im2show],
                     wdif=wdif, varlib=varlib, suptitre=suptitre, vdmin=vdmin, vdmax=vdmax,
                     figdir=figdir, savefig=savefig)

    # Les Erreurs Relatives : toujours car la moyenne est utilis�e apr�s 
    RelErr_ = []; RMRelErr_ = [];                    
    for j in np.arange(Nimg_) :
        Ej_ref   = np.sum(phi_ref[j]);  
        Ej_est   = np.sum(phi_est[j])
        RelErrj_ = np.abs(Ej_ref - Ej_est) / Ej_ref;
        RelErr_.append(RelErrj_);
    MoyRelErr    = np.mean(RelErr_); # Moyenne des erreurs relatives
    titre_err    = "%s %s, ErG (Michel) : mean=%f min=%f, max=%f, std=%f" \
                    %(strset, varlib, MoyRelErr, np.min(RelErr_), \
                    np.max(RelErr_), np.std(RelErr_));
    print(titre_err);
    
    # Les valeurs par image (Daily sum)
    Somimg_ref = np.sum(phi_ref, axis=(1,2));
    Somimg_est = np.sum(phi_est, axis=(1,2));
    MatCor     = np.corrcoef(Somimg_ref, Somimg_est);
    correl     = MatCor[0,1];
    titre_valimg = "%s %s: Daily sum ; Coef. Correl. Ref.vs Est. : %f"%(strset, varlib, correl);
    print(titre_valimg);

    if FIGBYIMGS : #! Par image ordre chrono        
        # Par image dans l'ordre chronologique
        ids = np.argsort(calX[1]); # indice de tri des dates 
        plt.figure();
        plt.plot(np.array(RelErr_)[ids], '-*');
        plt.plot([0,Nimg_-1],[MoyRelErr, MoyRelErr]); # Trait de la moyenne
        plt.xticks(np.arange(Nimg_),calX[0][ids], rotation=35, ha='right');
        plt.axis("tight");
        plt.title(titre_err);
        #
        # Les valeurs par image
        plt.figure();        
        plt.plot(Somimg_ref[ids], '-*');        
        plt.plot(Somimg_est[ids], '-*');
        plt.xticks(np.arange(Nimg_),calX[0][ids], rotation=35, ha='right');
        plt.axis("tight");
        plt.legend(["NATL60", "Resac"], framealpha=0.5);
        plt.title(titre_valimg);
  
    # Somme Temporelle pour chaque pixel (pour une �tude spatiale)
    if not FIGBYPASS and resol in ResoAll : 
        PHI_ = np.sum(phi_ref, axis=(0));
        plt.figure(); plotavar(PHI_, "", None, None, None, cmap=wcmap[wk_]);
        plt.title("%s NATL60: Somme temporelle pour l'%s des Pixels"%(strset,varlib));            
        PHI_ = np.sum(phi_est, axis=(0));
        plt.figure(); plotavar(PHI_, "", None, None, None, cmap=wcmap[wk_]);
        plt.title("%s Resac: Somme temporelle pour l'%s des Pixels"%(strset,varlib));

    # Pourcentatge au Nord et au Sud d'une Latitude donn�e
    if LIMLAT_NORDSUD > 0 :             
        NL_lim   = nl2limlat(Nlig_, LIMLAT_NORDSUD); # Nombre de ligne jusqu'� la latitude limite Nord-Sud
        # for NATL60
        sumphi_ref = np.sum(phi_ref);
        XN_, XS_   = splitns(phi_ref, NL_lim);
        sumS       = np.sum(XS_);           sumN   = np.sum(XN_);
        pcentS     = sumS / sumphi_ref;     pcentN = sumN / sumphi_ref;
        print("%s Pcent %s NATL60: Nord=%.4f ; Sud=%.4f ; (sum pcent = %.4f (should be 1.)"
              %(strset, varlib, pcentN, pcentS, pcentN+pcentS));
        # for RESAC
        sumphi_est = np.sum(phi_est);
        XN_, XS_   = splitns(phi_est, NL_lim);            
        sumS       = np.sum(XS_);           sumN   = np.sum(XN_);
        pcentS     = sumS / sumphi_est;     pcentN = sumN / sumphi_est;
        print("%s Pcent %s RESAC : Nord=%.4f ; Sud=%.4f ; (sum pcent = %.4f (should be 1.)"
              %(strset, varlib, pcentN, pcentS, pcentN+pcentS));
        del XN_, XS_;               
   
    # Pour Histogramme "compar�s" (Ref., Est.) et scatter plot
    if flag_histo or flag_scat :
        Yref_ = phi_ref.ravel()
        Yest_ = phi_est.ravel()
        
    if flag_histo : # Histogramme "compar�s" (Ref., Est.)
        epsi = 0.0;
        min_ = np.min([phi_ref, phi_est]);
        print("Histo %s Ref-Est : min = %f"%(varlib, min_));
        if min_ <= 0 :
            epsi = -min_ + 1e-30;
        # histo en log ...                                                             
        logrefe = np.log(Yref_ + epsi);
        logeste = np.log(Yest_ + epsi);
        histodbl(logrefe, logeste, legende=['Ref','Est'], nbins=NBINS);
        plt.title("%s Histo log(%s+epsi=%e) (%dbins),\nrelerr_mean=%.4e"
                  %(strset,varlib, epsi, nbins, MoyRelErr), fontsize=x_figtitlesize);
    #
    if flag_scat : # Scatter plot not in log 
        pipo, pipo, R2 = scatplot(Yref_, Yest_, fsize=fsizescat, squareax=squarescatax)
        plt.title("%s %s scatter-plot, relerr_mean=%.4e R2=%f"%(strset,varlib,MoyRelErr,R2), fontsize=x_figtitlesize);               
#----------------------------------------------------------------------
def anycorrplex (Urefout, Vrefout, Uestout, Vestout, strset, calX=None) :
    # Elargissement de la Corr�lation Complexe � d'autre variable que u et v ...
    Nimg_ = len(Urefout);
    def corrplexpo (U, Up) :
        import cmath
        U_  = U.ravel();
        Up_ = Up.ravel()
        Upc = Up_.conjugate(); # Up conjug�
        Cc  = np.dot(U_,Upc) / (np.linalg.norm(U_) * np.linalg.norm(Up_)); # Corr�lation complexe
        modulCc, argCc = cmath.polar(Cc); # module->corr�lation, argCc->angle  \ entre les 2 champs complexes
        return modulCc, argCc, Cc, Upc;
    U_  = Urefout + 1j*Vrefout;
    Up_ = Uestout + 1j*Vestout;
    modulCc, argCc, Cc, Upc = corrplexpo (U_, Up_);
            
    # Calcul par image, always
    ModCC = []; ACC = [];
    for k in np.arange(Nimg_) : # Pour chaque image k           
        Uk_  = Urefout[k] + 1j*Vrefout[k];
        Upk_ = Uestout[k] + 1j*Vestout[k];
        modulCck, argCck, Cck, Upck = corrplexpo (Uk_, Upk_);
        ModCC.append(modulCck);
        ACC.append(argCck);
    # Modules
    moymodcc = np.mean(ModCC);
    titre_mod = "%s Correlation complexe: Module : Moy=%f, Min=%f, Max=%f, Std=%f" \
                %(strset, moymodcc, np.min(ModCC), np.max(ModCC), np.std(ModCC));
    print(titre_mod);
    # Angles en radian (no more)          
    # Angles en degr� absolu
    ACC       = np.array(ACC);
    AbsACC    = np.abs(ACC*180/np.pi); 
    moyabsacc = np.mean(AbsACC);
    titre_ang = "%s Correlation complexe: |Angle(degre)| : Moy=%f, Min=%f, Max=%f, Std=%f" \
                %(strset, moyabsacc, np.min(AbsACC),np.max(AbsACC),np.std(AbsACC));
    print(titre_ang);

    if FIGBYIMGS : # Par image
        # Par image ordre chrono
        ids = np.argsort(calX[1]); # indice de tri des dates
        # Module
        ModCC = np.array(ModCC)[ids];
        plt.figure(); plt.plot(ModCC,'.-');          
        plt.plot([0,Nimg_-1],[moymodcc, moymodcc]); 
        plt.axis("tight");
        plt.xticks(np.arange(Nimg_),calX[0][ids], rotation=35, ha='right');       
        plt.title(titre_mod);
        
        # Angles en degr� absolu 
        AbsACC = AbsACC[ids]
        plt.figure(); plt.plot(AbsACC,'.-');
        plt.plot([0,Nimg_-1],[moyabsacc, moyabsacc]); 
        plt.axis("tight");
        plt.xticks(np.arange(Nimg_),calX[0][ids], rotation=35, ha='right');
        plt.title(titre_ang);
#----------------------------------------------------------------------
def UVresult(Urefout, Vrefout, Uestout, Vestout, resol, dim_dic,
             strset, flag_cosim=0, flag_nrj=0, flag_ens=0, flag_corrplex=0, flag_div=0,
             dx=dxRout, dy=dyRout, flag_scat=0, flag_histo=0, calX=None, wdif='log',
             nbins=NBINS, fsizescat=(10,10), squarescatax=False) :
    print("-- in UVresult --"); #print(np.shape(Urefout)); #(55L, 1L, 144L, 153L)
    
    if flag_cosim :# Cosine Similarity
        dcos = 1 - dcosine2D(Urefout, Vrefout, Uestout, Vestout);
        print("%s cosine similarity on brute scalled : %.6f"%(strset,dcos));

    if flag_nrj : # Energie
        nrj_ref = (Urefout**2 + Vrefout**2) / 2; print("nrj shape : ", np.shape(nrj_ref));
        nrj_est = (Uestout**2 + Vestout**2) / 2;
        phistuff (nrj_ref, nrj_est, resol, dim_dic, "NRJ", "Energie", strset,
                  calX, wdif, flag_histo, flag_scat, nbins=nbins,
                  fsizescat=fsizescat, squarescatax=squarescatax);

    if flag_ens : # Enstrophie
        enst_ref = enstrophie2d(Urefout, Vrefout, dx, dy); #print("enstro shape : ", np.shape(enst_ref));
        enst_est = enstrophie2d(Uestout, Vestout, dx, dy);
        phistuff(enst_ref, enst_est, resol, dim_dic, "ENS", "Enstrophie", strset,
                  calX, wdif, flag_histo, flag_scat, nbins=nbins,
                  fsizescat=fsizescat, squarescatax=squarescatax);

    if flag_nrj and flag_ens and FIGBYIMGS :
        # Correlation Energie \ Enstrophie
        SomEimg_ref = np.sum(nrj_ref, axis=(1,2,3));
        SomEimg_est = np.sum(nrj_est, axis=(1,2,3));
        SomSimg_ref = np.sum(enst_ref, axis=(1,2));
        SomSimg_est = np.sum(enst_est, axis=(1,2));
        r_ref_ = np.corrcoef(SomEimg_ref, SomSimg_ref); 
        r_est_ = np.corrcoef(SomEimg_est, SomSimg_est);
        print("Correlation temporelle Energie \ Enstrophie : NATL60:=%.2f ; Resac:=%.2f"
              %(r_ref_[0,1],r_est_[0,1]))

    if flag_div : # Divergence
        div_ref    = divergence2d(Urefout, Vrefout, dx, dy);        
        div_est    = divergence2d(Uestout, Vestout, dx, dy);
        RMS, Nnan, inan = nanrms(div_ref, div_est);
        print("%s Divergence RMS (%dnan) : %.4e" %(strset, Nnan, RMS));

    if flag_corrplex : # Correlation Complexe
        anycorrplex (Urefout, Vrefout, Uestout, Vestout, strset, calX=calX); 
#
#--------------------------------------------------
def resultuv(Yrefout, Yestout, ResOut, Dout_diclst, strset, flag_cosim=0, flag_nrj=0,  
             flag_ens=0, flag_corrplex=0, flag_div=0, dx=dxRout, dy=dyRout,
             flag_scat=0, flag_histo=0, calX=None, wdif='log', nbins=NBINS,
             fsizescat=(10,10), squarescatax=False) :
    print("-- in resultuv --");
    iu_ = varOut.index("U");  Urefout = Yrefout[iu_];  Uestout = Yestout[iu_]; resol=ResOut[iu_]; dim_dic=Dout_diclst[iu_]
    iv_ = varOut.index("V");  Vrefout = Yrefout[iv_];  Vestout = Yestout[iv_];
    if resol != ResOut[iv_] :
        raise ValueError('resultuv: U and V must have same resolution');
    if LIMLAT_NORDSUD <= 0 :
        UVresult(Urefout, Vrefout, Uestout, Vestout, resol, dim_dic,
                 strset, flag_cosim=flag_cosim,
                 flag_nrj=flag_nrj, flag_ens=flag_ens,
                 flag_corrplex=flag_corrplex, flag_div=flag_div, dx=dx, dy=dy,
                 flag_scat=flag_scat, flag_histo=flag_histo, calX=calX,
                 wdif=wdif, nbins=nbins,
                 fsizescat=fsizescat, squarescatax=squarescatax);
    else : # PLM, je choisis de faire que tout, ou que Nord-Sud, sinon on
           # croule sous les images
        pipo, pipo, Nlig_, pipo = np.shape(Urefout);
        # Nombre de ligne jusqu'� la latitude limite Nord-Sud
        NL_lim = nl2limlat (Nlig_, LIMLAT_NORDSUD);
        # Split Nord-Sud
        UrefN_,  UrefS_ = splitns(Urefout, NL_lim);
        VrefN_,  VrefS_ = splitns(Vrefout, NL_lim);
        UestN_,  UestS_ = splitns(Uestout, NL_lim);
        VestN_,  VestS_ = splitns(Vestout, NL_lim);
        # Then
        #strset = strset+"-Nord";
        UVresult(UrefN_, VrefN_, UestN_, VestN_, resol, dim_dic,
                 strset+"-Nord", flag_cosim=flag_cosim,
                 flag_nrj=flag_nrj, flag_ens=flag_ens,
                 flag_corrplex=flag_corrplex, flag_div=flag_div, dx=dx, dy=dy,
                 flag_scat=flag_scat, flag_histo=flag_histo, calX=calX,
                 wdif=wdif, nbins=nbins,
                 fsizescat=fsizescat, squarescatax=squarescatax);
        #strset = strset+"-Sud";
        UVresult(UrefS_, VrefS_, UestS_, VestS_, resol, dim_dic,
                 strset+"-Sud", flag_cosim=flag_cosim,
                 flag_nrj=flag_nrj, flag_ens=flag_ens,
                 flag_corrplex=flag_corrplex, flag_div=flag_div, dx=dx, dy=dy,
                 flag_scat=flag_scat, flag_histo=flag_histo, calX=calX,
                 wdif=wdif, nbins=nbins,
                 fsizescat=fsizescat, squarescatax=squarescatax);
#--------------------------------------------------
def plot_data_histogram(var_list, reso_list, VA_array, VV_array, VT_array, nbins=20,
                        range_ssh=(-0.5, 1.3), range_sst=(1, 29), 
                        range_u=(-1.5, 1.5), range_v=(-1.5, 1.5),
                        stitle=None, figsize=(16,10),
                        figdir='.', savefig=False) :
    fig,axes = plt.subplots(nrows=len(var_list), ncols=3, figsize=figsize)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.98, top=0.915, wspace=0.18, hspace=0.25)

    if stitle is not None :
        plt.suptitle(stitle)

    # Hist de comparaison des distributions par variable et par ensemble (->draft, ppt)
    for irow,zipped in enumerate(zip(var_list, reso_list, VA_array, VV_array, VT_array)) :
        varname,resol,VAtmp_brute,VVtmp_brute,VTtmp_brute = zipped
        if varname == 'SSH' :
            range_var = range_ssh
        elif varname == 'SST':
            range_var = range_sst
        elif varname == 'U' :
            range_var = range_u
        elif varname == 'V' :
            range_var = range_v
        
        H_ = VAtmp_brute
        ax = axes[irow,0]
        ax.hist(H_.ravel(), bins=nbins, range=range_var)
        ax.set_title(f"APP {varname} R{resol:02d}")
        
        H_ = VVtmp_brute
        ax = axes[irow,1]
        ax.hist(H_.ravel(), bins=nbins, range=range_var)
        ax.set_title("VAL")
        
        H_ = VTtmp_brute
        ax = axes[irow,2]
        ax.hist(H_.ravel(), bins=nbins, range=range_var)
        ax.set_title("TEST")
    if savefig:
        printfilename = strconv_for_title(stitle)
        plt.savefig(os.path.join(figdir,f"{printfilename}.png"))

    return
#
#--------------------------------------------------
def plot_history(lhist, fsize=(18,8), log=False, alpha=0.5, title=None, 
                 figdir='.', savename="losses_apprentissage", savefig=False,
                 rndcolors=False, mintattitle=True, mintattitlefmt='{:.4e}',                 
                 linewidth=1.5, losslw=None, linestyle='-', valls=None):
    lstofkeys = list(lhist.keys())
    val_flag = 'val_loss' in lstofkeys
    if 'history' in lhist.keys():
        lhist = lhist.history;
    size_obj=len(lhist[list(lhist.keys())[0]])
    x=np.linspace(0,size_obj-1,size_obj)
    nb_loss=len(lhist)
    color = None
    if title is None :
        title = 'Loss curves'
    if rndcolors :
        # random colors
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(nb_loss)]
    #plt.figure(figsize=fsize)
    f,ax = plt.subplots(nrows=1, ncols=1,figsize=fsize)
    plt.subplots_adjust(left=0.04, bottom=0.105, right=0.99, top=0.915) #, wspace=0.2, hspace=0)
    for i in range(nb_loss):
        lw = linewidth; ls = linestyle
        if losslw is not None and lstofkeys[i] in ['loss','val_loss']:
            lw = losslw
        if valls is not None and lstofkeys[i][:3] == 'val' :
            ls = valls
        if color is None :
            plt.plot(x, lhist[list(lhist.keys())[i]], 
                     label=list(lhist.keys())[i],
                     linewidth=lw, linestyle=ls, alpha=alpha)
        else:
            plt.plot(x, lhist[list(lhist.keys())[i]], c=color[i],
                     label=list(lhist.keys())[i],
                     linewidth=lw, linestyle=ls, alpha=alpha)
    plt.xlabel("Epochs")
    if log:
        plt.ylabel("Log(Loss)")
        plt.yscale('log')
    else:
        plt.ylabel("Loss")
    # si val_loss 
    if val_flag :
        itmin_noval = np.argmin(lhist['loss'])
        if mintattitle :
            minlosslabel = mintattitlefmt.format(lhist['val_loss'][itmin_noval])
            title += f" (min val: {minlosslabel} @ {itmin_noval+1} epochs)"
        plt.axvline(itmin_noval,c="black")
        print("loss n'a pas baisse a partir de l'iteration No. "+str(itmin_noval))
    else:
        itminval = np.argmin(lhist['val_loss'])
        if mintattitle :
            minvallosslabel = mintattitlefmt.format(lhist['val_loss'][itminval])
            title += f" (min val_loss: {minvallosslabel} @ {itminval+1} epochs)"
        plt.axvline(itminval,c="black")
        print("val_loss n'a pas baisse a partir de l'iteration No. "+str(itminval))
    plt.title(title)
    plt.legend(ncol=2)
    if savefig:
        plt.savefig(os.path.join(figdir,f"{savename}.png"))
    #plt.show()
# #--------------------------------------------------
# def plot_history_RESUME(lhist, fsize=(18,8)):
#     size_obj=len(lhist[list(lhist.keys())[0]])
#     x=np.linspace(0,size_obj-1,size_obj)
#     nb_loss=len(lhist)
#     #plt.figure(figsize=fsize)
#     f,ax = plt.subplots(nrows=1, ncols=1,figsize=fsize)
#     plt.subplots_adjust(left=0.04, bottom=0.105, right=0.99, top=0.915) #, wspace=0.2, hspace=0)
#     for i in range(nb_loss):
#         plt.plot(x,np.log(lhist[list(lhist.keys())[i]]),label=list(lhist.keys())[i], alpha=0.7)
#     plt.xlabel("Epochs")
#     plt.ylabel("Log(Loss)")
#     itminval=np.argmin(lhist['val_loss'])
#     plt.axvline(itminval,c="black")
#     print("La val_loss n'a pas baisse a partir de l'iteration No. "+str(itminval))
#     plt.legend()
#     plt.savefig(os.path.join(Mdl2resume,"Images",f"loss.png"))
#     #plt.show()
#--------------------------------------------------
def saveHist(path, history):
    with open(path, 'wb') as file:
        pickle.dump(history, file)
#--------------------------------------------------
def appendHist(h1, h2):
    if h1 == {}:
        return h2
    else:
        dest = {}
        for key, value in h1.items():
            dest[key] = value + h2[key]
        return dest
#--------------------------------------------------

class LossHistory(Callback):
    def __init__(self, file="history.pkl") :
         self.history_filename = file

    def file_name(self) :
         return self.history_filename

    def on_epoch_end(self, epoch, logs = None):
        new_history={}
        for k, v in logs.items(): # compile new history from logs
            new_history[k] = [v] # convert values into lists
        try:
            with open(self.history_filename, 'rb') as file:
                old_history=pickle.load(file)
        except:
            print("Fail au moment du chargement du fichier "+self.history_filename)
            old_history={}

        saveHist(self.history_filename, appendHist(old_history,new_history)) # save history from current training

#
#--------------------------------------------------
# Fonctions de chargement des données:
#   - load_resac_by_var_and_resol() .. nouvelle version chargeant des arrays
#           par variable et resolution.
#       Fichiers:
#           <donnees>/NATL60byVar/NATL60_SSH_R01.npy
#                                /NATL60_SSH_R03.npy
#                                /NATL60_SSH_R09.npy
#                                /NATL60_SSH_R27.npy
#                                /NATL60_SSH_R81.npy
#       et ainsi de suite pour SST, U et V.
#
#   - load_resac_data() .............. vieille version chargeant le grand
#           array R01.
#       Fichiers:
#           <donnees>/natl60_htuv_01102012_01102013.npz   (very big array)
#
#--------------------------------------------------
def get_resac_data_dir() :
    datasets_dir = os.getenv('RESAC_DATASETS_DIR', False)
    if datasets_dir is False:
        print('\n# ATTENTION !!\n#')
        print('#  Le dossier contenant les datasets est introuvable !\n#')
        print('#  Pour que les notebooks et programmes puissent le localiser, vous')
        print("#  devez préciser la localisation de ce dossier datasets via la")
        print("#  variable d'environnement : RESAC_DATASETS_DIR.\n#")
        print('#  Exemple :')
        print("#    Dans votre fichier .bashrc :")
        print('#    export RESAC_DATASETS_DIR=~/mon/datasets/folder\n#')
        print('#')
        print('# Valeurs typiques :')
        print('# - Au Locean, toute machine (acces lent):')
        print('#     export RESAC_DATASETS_DIR="/net/pallas/usr/neuro/com/carlos/Clouds/'+\
              'SUnextCloud/Labo/Stages-et-Projets-long/Resac/donnees"')
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
    return datasets_dir
#
#--------------------------------------------------
# nouvelle version PREFEREZ CETTE METHODE
def load_resac_by_var_and_resol(varIn, varOut, ResoIn, ResoOut, subdir='NATL60byVar',
                                data_prefix='NATL60', data_suffix='',
                                zone=None, lat=None, lon=None, itime=None):
    """
    Exemple d'usage:
        V_data_list, couple_var_reso_list = load_resac_by_var_and_resol(varIn,varOut,ResoIn,ResoOut)

    Lecture des données RESAC par variable/résolution.  La function s'attend à trouver
    le répertoire des données dans la variable d'environnement RESAC_DATASETS_DIR.

    Les données par Variable/résolution se trouvent normalement dans un sous-dossier
    appelé 'NATL60byVar'. Il est posisble de spécifier un autre nom avec l'option 
    subdir='DOSSIER'.
    
    Il est possible de specifier une zone 'North' ou 'South' pour selectionner 
    une des moities des donnees. Exemple: -zone="North", u bien especifier une 
    zone de latitude et/ou longitude en donnant le couple de valeurs Min et Max
    de la coordonnée en question. Si l'in des Min ou Max est specifié comme None
    alors c'est le bout dans le sens Min ou Max. Par exemple, la zone à partir
    de 30 deg. de Latitude Nord et jusqu'à 50 degres de longitude Ouest s'ecrit:
    -lat=[30, None], -lon=[None, -50]
    
    Enfin, il aus aussi possible de specifier un schema d'indices de Time a 
    selectionner en especifiant en une tuple l'index initial et le pas de
    selection. Ainsi, la valeur par de faut qui est (0, 1) selecctinne depuis
    l'element 0 un par un jusque la fin, c'est à dire tous les pas de temps
    disponibles dans les données.
    
    Retourne trois éléments:
        
        - liste d'array 3D ([np.time steps, y size, x size]) des données contenant
          les arrays individuels par variable et résolution nécessaires au cas (selon
          les variables d'entrée varIn, varOut, ResoIn, ResoOut). Un element
          dans cette liste de données par couple (Variable, Résolution) distinct.
          
        - liste de couples (Variable, Résolution) intervenant dans le cas.
        
        - liste de dictionnaires indicant pour chaque couple (Variable, Résolution) 
          les dimensions des variables (keys: time', 'lat', 'lon', 'lat_border',
          'lon_border').
    
    """
    # ---- datasets location
    datasets_dir = get_resac_data_dir();
    #
    couple_var_reso_list = []
    for v,r in zip(varIn+varOut,ResoIn+ResoOut):
        if not (v,r) in couple_var_reso_list :
            couple_var_reso_list.append((v,r))
    #
    if zone is not None or lat is not None or lon is not None:
        # affiner les limites lat/lon pour coller precisement aux données de plus
        # basse resolution et ainsi, au fur et a mesure que l'on traite les plus
        # hautes resolutions, garantir la coherence des dimensions (multiples de 3,9,27,...)
        r = max(ResoIn+ResoOut)  # la plus basse resolution, 81, normalement
        dimension_tmp = np.load(os.path.join(datasets_dir,subdir,f"{data_prefix}_coords_R{r:02d}{data_suffix}.npz"))
        lat,lon = get_real_lat_lon_limits(dimension_tmp['latitude_border'],
                                          dimension_tmp['longitude_border'],
                                          zone=zone,
                                          lat_limits=lat, lon_limits=lon)
        print(f"Selection par zone ou Lan/Lon ({lat}/{lon})")
    #
    # Lecture Des Donnees    
    V_data_list = []; D_dico_list = []
    for i,c in enumerate(couple_var_reso_list):
        v,r  = c
        print(f"loading data: '{v}' at R{r:02d}{data_suffix}")
        data_tmp = np.load(os.path.join(datasets_dir,subdir,f"{data_prefix}_{v.upper()}_R{r:02d}{data_suffix}.npy"))
        dimension_tmp = np.load(os.path.join(datasets_dir,subdir,f"{data_prefix}_coords_R{r:02d}{data_suffix}.npz"))
        # conversion de dimension_tmp, objet 'numpy.lib.npyio.NpzFile', en dico_dim dictionnaire
        dico_dim = { 'time': dimension_tmp['time'],
                     'lat' : dimension_tmp['latitude'],
                     'lon' : dimension_tmp['longitude'],
                     'lat_border' : dimension_tmp['latitude_border'],
                     'lon_border' : dimension_tmp['longitude_border'] }
        if lat is not None or lon is not None or itime is not None:
            print(f" - {v.upper()}_R{r:02d}{data_suffix} - Dim AVANT: {data_tmp.shape}")
            if lat is not None or lon is not None :
                if i == 0:
                    print(f"   (avant) limites lat des donnees: [{dico_dim['lat'][0]},{dico_dim['lat'][-1]}] en {len(dico_dim['lat'])} valeurs,"+\
                          f" lon: [{dico_dim['lon'][0]},{dico_dim['lon'][-1]}] en {len(dico_dim['lon'])} valeurs.")
                # selection par zones de coordonnees
                data_tmp, dico_dim = select_by_coords(data_tmp, dico_dim,
                                                      lat_limits=lat, lat_axis=1,
                                                      lon_limits=lon, lon_axis=2)
                if i == 0:
                    print(f"   (apres) limites lat des donnees: [{dico_dim['lat'][0]},{dico_dim['lat'][-1]}] en {len(dico_dim['lat'])} valeurs,"+\
                          f" lon: [{dico_dim['lon'][0]},{dico_dim['lon'][-1]}] en {len(dico_dim['lon'])} valeurs.")
            if itime is not None:
                currtime = dico_dim['time']
                if i == 0:
                    print(f"   (avant) limites Time des donnees: [{dico_dim['time'][0]},{dico_dim['time'][-1]}] en {len(dico_dim['time'])} valeurs")
                # selection par sous-echantillonnage dans l'axe de Time
                data_tmp, new_time = select_data_by_dim(data_tmp, itime, dim_lbl=currtime,
                                                        dim_axis=0)
                dico_dim['time'] = new_time
                if i == 0:
                    print(f"   (apres) limites Time des donnees: [{dico_dim['time'][0]},{dico_dim['time'][-1]}] en {len(dico_dim['time'])} valeurs")
            print(f" - Dim APRES: {data_tmp.shape}")
        V_data_list.append(data_tmp)
        D_dico_list.append(dico_dim)
    #
    return V_data_list, couple_var_reso_list, D_dico_list
#
#--------------------------------------------------
# Ancienne version chargeant un seul array qui contienne toutes les donnees
# (les 4 variables, tous les pas de temps, ...)
# Le chargement est TRES LENT ET GOURMAND EN MEMOIRE dans le cas de l'array
# "natl60_htuv_01102012_01102013.npz" des données NATL60 qui est en resolution
# tres fine.
def load_resac_data(npz_data_file, 
                    zone=None,             # zone de selection pre-configure: "North", "South".
                    lat=None, lon=None,    # limites lat, lon de la zone de selection
                    itime=None,            # indices de sous-echantillonnage des patterns (axe de Time)
                    time_init=None,
                    nav_lat_xtremes=[ 26.57738495,  44.30360031],
                    nav_lon_xtremes=[-64.41895294, -40.8841095 ]) :
    """
    Exemple d'usage:
        FdataAllVar,varlue = load_resac_data("natl60_htuv_01102012_01102013.npz")
        FdataAllVar,varlue = load_resac_data("natl60_htuv_01102012_01102013.npz",[OPTIONS])

    Options:
        zone=ZONE ..... Zone geographique d'etude (une partie de la zone couverte
                        par les donnees actuelless) "North", "South"
        lat/lon=L**LIMITS ... Limites min et max de la Latitude et/ou de la
                        Longitude pour la selection de zone.
        itime=SUBSAMPLING_DES_PATTTERNS .. Entier ou tuple indiquant l'indice
                        initial et le pas de sous-echantillonnage dans la selection
                        des patterns. 1 ou (0, 1) indique toutes les données (pas
                        de 1) à partir de l'indice 0.
        time_init="ANNEE-MOI-JOUR" ....... Date du premier jour des données (pour
                        etablir la liste de dates, les données ne contenant pas
                        la variable time/date). Par defaut c'est "2012-10-01".
                        On considere les donnees comme journaliers, la date suivante
                        sera donc le lendemain de cette date initiale.
        nav_lat_xtremes/nav_lon_xtremes=[LMIN, LMAX] ....... Limites Min et Max
                        en Latitute et/ou Longitude des donnees. Par defaut ce
                        sont les limites NATL60:
                            nav_lat_xtremes=[ 26.57738495,  44.30360031],
                            nav_lon_xtremes=[-64.41895294, -40.8841095 ],
                        Les coordonnees des chaque pixel sera calculée selon les
                        dimensions des donnees et dans une grille reguliere
                        limitee par ces valeurs extremes.
    
    Lecture des données RESAC.  La function s'attend à trouver le repertoire
    des données dans la variable d'environnement RESAC_DATASETS_DIR.
    
    Il est possible de specifier une zone 'North' ou 'South' pour selectionner 
    une des moities des donnees. Exemple: -zone="North", u bien especifier une 
    zone de latitude et/ou longitude en donnant le couple de valeurs Min et Max
    de la coordonnée en question. Si l'in des Min ou Max est specifié comme None
    alors c'est le bout dans le sens Min ou Max. Par exemple, la zone à partir
    de 30 deg. de Latitude Nord et jusqu'à 50 degres de longitude Ouest s'ecrit:
    -lat=[30, None], -lon=[None, -50]

    Enfin, il aus aussi possible de specifier un schema d'indices de Time a 
    selectionner en especifiant en une tuple l'index initial et le pas de
    selection. Ainsi, la valeur par de faut qui est (0, 1) selecctinne depuis
    l'element 0 un par un jusque la fin, c'est à dire tous les pas de temps
    disponibles dans les données.
    
    Retourne l'array 4D des donnees ([nb.variable, np.time steps, y size, x size])
    et la liste des noms de variables dans l'array.
    """
    import pandas as pd

    if time_init is None :
        time_init = "2012-10-01"
    #
    # ---- datasets location
    datasets_dir = get_resac_data_dir();
    #
    # join dataset dir with filename
    data_set_filename = os.path.join(datasets_dir,npz_data_file)
    #
    # Lecture Des Donnees
    print(f"Lecture Des Donnees du fichier {npz_data_file} ... ", end='', flush=True)
    Data_       = np.load(data_set_filename)
    FdataAllVar = Data_['FdataAllVar']
    varlue      = list(Data_['varlue'])
    # Pour enlever les b devant les chaines de caracteres lors de la lecture et pour la
    # conversion de 'SSU','SSV' en 'U','V'
    varlue      = ['U' if i==b'SSU' else 'V' if i==b'SSV' else i.decode() for i in varlue]
    print(f'\nArray avec {len(varlue)} variables: {varlue}')
    print(f'contenant des images de taille {FdataAllVar.shape[2:]} pixels')
    print(f'et {FdataAllVar.shape[1]} pas de temps (une image par jour).')
    print(f'Dimensions de l\'array: {np.shape(FdataAllVar)}')
    #
    _, Nimg_, Nlig_, Ncol_ = np.shape(FdataAllVar) #(4L, 366L, 1296L, 1377L)
    #
    dimensions = {}
    # Coordonnees : TIME
    dimensions['time'] = pd.date_range(time_init, periods=Nimg_)
    # Limites absoluts de la zone geographique (au centre des pixels)
    # sont données par les arguments en option
    #   - nav_lat_xtremes
    #   - nav_lon_xtremes 
    # les valeurs par defaut ce sont ceux approximés pour la zone NATL60.
    # Coordonnees : Lat / Lon (au centre du pixel)
    all_lat = np.linspace(nav_lat_xtremes[0],nav_lat_xtremes[1],num=Nlig_) # latitude of center of the pixel
    all_lon = np.linspace(nav_lon_xtremes[0],nav_lon_xtremes[1],num=Ncol_)
    delta_lat = (all_lat[1]-all_lat[0])
    delta_lon = (all_lon[1]-all_lon[0])
    all_lat_border = np.concatenate((all_lat - delta_lat/2,[all_lat[-1] + delta_lat/2])); 
    all_lon_border = np.concatenate((all_lon - delta_lon/2,[all_lon[-1] + delta_lon/2])); 
    # Bords inferieur et supperieur des pixels
    dimensions['lat'] = all_lat
    dimensions['lon'] = all_lon
    dimensions['lat_border'] = all_lat_border # latitude border inf and sup for each pixel in the zone
    dimensions['lon_border'] = all_lon_border # longitude border inf and sup for eaxh pixel in the zone
    #
    if itime is not None and itime == 1 and itime == (0, 1):
        # nothing changes then dont do nothing !!
        itime = None
    # selection par zones de coordonnees
    if zone is not None or lat is not None or lon is not None or itime is not None:
        print(f"Selection par zone ou Time:\n - Dim AVANT: {FdataAllVar.shape}")
        if zone is not None or lat is not None or lon is not None :
            print(f" - selection par zone ou Lan/Lon ({lat}/{lon}):")
            print(f"   (avant) limites lat des donnees: [{dimensions['lat'][0]},{dimensions['lat'][-1]}] en {len(dimensions['lat'])} valeurs,"+\
                  f" lon: [{dimensions['lon'][0]},{dimensions['lon'][-1]}] en {len(dimensions['lon'])} valeurs.")
            # Reproduit les coordonnees des pixels (du centre et bords) pour la plus
            # basse ressolution pour effectuer la selection de zone sur ces coordonnées.
            r = max(ResoIn+ResoOut)  # la plus basse resolution, 81, normalement
            all_lat_rLow = all_lat[(r//2)::r]
            all_lon_rLow = all_lon[(r//2)::r]
            delta_lat_rLow = (all_lat_rLow[1]-all_lat_rLow[0])
            delta_lon_rLow = (all_lon_rLow[1]-all_lon_rLow[0])
            all_lat_border_rLow = np.concatenate((all_lat_rLow - delta_lat_rLow/2,[all_lat_rLow[-1] + delta_lat_rLow/2])); 
            all_lon_border_rLow = np.concatenate((all_lon_rLow - delta_lon_rLow/2,[all_lon_rLow[-1] + delta_lon_rLow/2])); 
            #
            # corrige les limites de zone selon les coordonnees de basse resolution 
            lat,lon = get_real_lat_lon_limits(all_lat_border_rLow,
                                              all_lon_border_rLow,
                                              zone=zone,
                                              lat_limits=lat, lon_limits=lon)
            #
            # effectue la selection de zone 
            FdataAllVar, dimensions = select_by_coords(FdataAllVar, dimensions,
                                                       lat_limits=lat, lat_axis=2,
                                                       lon_limits=lon, lon_axis=3)
            print(f"   (apres) limites lat des donnees: [{dimensions['lat'][0]},{dimensions['lat'][-1]}] en {len(dimensions['lat'])} valeurs,"+\
                  f" lon: [{dimensions['lon'][0]},{dimensions['lon'][-1]}] en {len(dimensions['lon'])} valeurs.")
        if itime is not None:
            currtime = dimensions['time']
            print(f" - selection par Time selon pattern: {itime}:")
            print(f"   (avant) limites Time des donnees: [{dimensions['time'][0]},{dimensions['time'][-1]}] en {len(dimensions['time'])} valeurs")
            # selection par sous-echantillonnage dans l'axe de Time
            FdataAllVar, new_time = select_data_by_dim(FdataAllVar, itime,
                                                       dim_lbl=currtime,
                                                       dim_axis=1)
            dimensions['time'] = new_time
            print(f"   (apres) limites Time des donnees: [{dimensions['time'][0]},{dimensions['time'][-1]}] en {len(dimensions['time'])} valeurs")
        print(f" - Dim APRES: {FdataAllVar.shape}")
    #
    return FdataAllVar,varlue,dimensions
#
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
                suptitre="some %s brute(%s%d) %s R%02d, m%s %s"%(
                    strset, inout,i+1, varIO[i], Resolst[i], im2show, calX0_)
                showsome(X_brute[i][im2show,0,:,:], Resolst[i], D_dicolst[i],
                         wmin=wbmin[wk_], wmax=wbmax[wk_],
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
    #
    return
#
#----------------------------------------------------------------------
def setresult(Mdl, x_set, strset, VXout_brute, calX, coparm,
              ResIn, ResOut, Din_diclst, Dout_diclst,
              flag_rms=0, flag_scat=0, flag_histo=0, nbins=NBINS, flag_cosim=0,
              flag_nrj=0, flag_ens=0, flag_corrplex=0, flag_div=0, 
              dx=dxRout, dy=dyRout, wdif='log', im2show=[],
              fsizehisto=(12,8),
              fsizesome=(10,12),
              fsizehquiv=(10,12),
              fsizerms=(12,8), szxlblrms=8,
              fsizescat=(10,10), squarescatax=False, 
              figdir='.', savefig=False) :

    imshow_lbl = '-'.join([str(n) for n in im2show])
    # Predicted coded data
    y_scale = Mdl.predict(x_set);
    if len(varOut)==1 : # Si une seule sortie mettre y_scale en forme de list comme l'est y_train
        y_scale = [y_scale];
    for i in np.arange(NvarOut):
        y_scale[i] = y_scale[i].transpose(0,3,1,2)
    
    #BACK TO BRUTE

    print("%s BACK TO BRUTE"%strset);

    for i in targetout : # boucle sur les variables cibles en sortie
        wk_ = tvwmm.index(varOut[i]);
        y_scale[i] = decodage(y_scale[i], coparm[i]);

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
            suptitre="some %s %s_Yscalled decodee %s"%(strset, varOut[i],imshow_lbl);
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
            suptitre = "some %s(out) H brute scalled(+uv) %s"%(strset,imshow_lbl); #! (H en dure ?)
            ih_ = showhquiv(y_scale, ResOut, Dout_diclst, im2show, calX0=calX[0], Yref=VXout_brute, wdif="",
                            wdifr=None, suptitre=suptitre, fsize=fsizehquiv,
                            figdir=figdir,  savefig=savefig);
    #
    return
#

#**********************************************************************
#----------------------------------------------------------------------
#
