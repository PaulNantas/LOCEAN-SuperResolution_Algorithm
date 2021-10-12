#!/bin/bash

## mode batch sans ecran
## Lancement d'une serveur X11 virtuel en arriere plan avec DISPLAY unique sur le noeud
## ($$ represente le numero de process ou s'execute ce scipt)
Xvfb :$$ -screen 0 1600x1200x24 & pid=$!
##positionnement de la variable DISPLAY pour ce serveur
export DISPLAY=:$$

SERVERNM="$( uname -n )"
DATELABEL="$( date +%Y%m%dT%H%M%S )"
DATEDEBUT="$( date )"

# ******************************************************************************
# Addresse Mail pour mails de debut et de fin du processus
# Ca marche si mail du 
# ******************************************************************************
MAILADDR="${USER}@locean.ipsl.fr"
# ******************************************************************************


# ******************************************************************************
# Execution du script - PARTIE A ADAPTER
# ******************************************************************************
# repertoire de travail et de lancement, la ou se trouvent le script Python a lancer:
cd ResacNet09/

#/Clouds/SUnextCloud/Labo/Stages-et-Projets-long/Resac/code_HK/ResacNet
# sous-repertoire ou sera cree le fichier contenant les messages de sortie et d'erreurs d'execution:
TRACEDIR="../Traces"

mkdir "${TRACEDIR}" 2> /dev/null

# Nom du script Python a executer
NOMSCRIPT="resacart.py"

# chargement de l'environnement python:
source /data/opt/Anaconda/miniconda3-2020-03-04a/bin/activate py37tfgpu
#
# ******************************************************************************

cat <<EOF | mail -s " ## DEBUT PROCESSUS $$ <${NOMSCRIPT}> sur ${SERVERNM} ${DATELABEL}" ${MAILADDR}
Script........ ${NOMSCRIPT}
X11 PID ...... $pid
Script PID ... $$
Execute sur .. ${SERVERNM}
PWD .......... ${PWD}/
Debut ........ $DATEDEBUT
EOF

# commade a executer:
python ${NOMSCRIPT} -v > "${TRACEDIR}"/Batch_Python_TRACE_${DATELABEL}.out 2>&1

cat <<EOF | mail -s "## FIN PROCESSUS $$ <${NOMSCRIPT}> sur ${SERVERNM} ${DATELABEL}" ${MAILADDR}
Script........ ${NOMSCRIPT}
X11 PID ...... $pid
Script PID ... $$
Execute sur .. ${SERVERNM}
PWD .......... ${PWD}/
Debut ........ $DATEDEBUT
Fin .......... $( date )
Trace dans ... "${TRACEDIR}/Batch_Python_TRACE_${DATELABEL}.out"
CONDA_PREFIX . ${CONDA_PREFIX}

*** derniere 15 lignes du fichier Trace: *************************
$( tail -n 15  "${TRACEDIR}"/Batch_Python_TRACE_${DATELABEL}.out )
******************************************************************
EOF

# Fin process Xfvb
kill -n 9 $pid
