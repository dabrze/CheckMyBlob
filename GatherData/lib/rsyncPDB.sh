#!/bin/sh

############################################################################
#
# Script for mirroring PDB FTP archive using rsync
#
############################################################################

# This script is being provided to PDB users as a template for using rsync 
# to mirror the FTP archive from an anonymous rsync server. You may want 
# to review rsync documentation for options that better suit your needs.
#
# Author: Thomas Solomon
# Date:   November 1, 2002

# Last Updated on: May 14, 2014

# Comment out the following line if you no longer want to see this info
echo "Prior to first use, you must edit this script to choose a server name, port number, and rsync option!"

############################################################################
# You should CHANGE THE NEXT THREE LINES to suit your local setup
############################################################################

MIRRORDIR=/home/przemek/PDB             # your top level rsync directory
LOGFILE=/home/przemek/PDB/logs               # file for storing logs
RSYNC=rsync                             # location of local rsync

##########################################################################################
#
#        YOU MUST UNCOMMENT YOUR CHOICE OF SERVER AND CORRESPONDING PORT BELOW
#
SERVER=rsync.wwpdb.org::ftp                                   # RCSB PDB server name
PORT=33444                                                    # port RCSB PDB server is using
#
#SERVER=rsync.ebi.ac.uk::pub/databases/rcsb/pdb-remediated     # PDBe server name
#PORT=873                                                      # port PDBe server is using
#
#SERVER=pdb.protein.osaka-u.ac.jp::ftp                         # PDBj server name
#PORT=873                                                      # port PDBj server is using
#
##########################################################################################


############################################################################
#                                                                          #
#                                                                          #
#     YOU MUST UNCOMMENT THE RYSNC OPTION BELOW THAT MEETS YOUR NEEDS!     #
#                                                                          #
#                                                                          #
############################################################################


############################################################################
#--------------------------------------------------------------------------#
# Targets to mirror the entire/parts of the FTP Tree 
#--------------------------------------------------------------------------#
############################################################################

############################################################################
# Rsync the entire FTP archive /pub/pdb (Aproximately 250 GB)
############################################################################
${RSYNC} -rlpt -v -z --delete --port=$PORT ${SERVER}/ $MIRRORDIR > $LOGFILE 2>/dev/null


############################################################################
# Rsync only the data directory /pub/pdb/data (Aproximately 192 GB)
############################################################################
#${RSYNC} -rlpt -v -z --delete --port=$PORT ${SERVER}/data/ $MIRRORDIR/data > $LOGFILE 2>/dev/null


############################################################################
#  Rsync only the derived data directory /pub/pdb/derived_data (Aproximately 163 MB)
############################################################################
#${RSYNC} -rlpt -v -z --delete --port=$PORT ${SERVER}/derived_data/ $MIRRORDIR/derived_data > $LOGFILE 2>/dev/null


############################################################################
#  Rsync only the doc directory /pub/pdb/doc (Aproximately 360 MB)
############################################################################
#${RSYNC} -rlpt -v -z --delete --port=$PORT ${SERVER}/doc/ $MIRRORDIR/doc > $LOGFILE 2>/dev/null



############################################################################
#--------------------------------------------------------------------------#
# Targets to rsync only specific coordinate formats
#--------------------------------------------------------------------------#
############################################################################

############################################################################
# Rsync only the PDB format coordinates  /pub/pdb/data/structures/divided/pdb (Aproximately 20 GB)
############################################################################
#${RSYNC} -rlpt -v -z --delete --port=$PORT ${SERVER}/data/structures/divided/pdb/ $MIRRORDIR > $LOGFILE 2>/dev/null


############################################################################
# Rsync only the mmCIF format coordinates  /pub/pdb/data/structures/divided/mmCIF (Aproximately 24 GB)
############################################################################
#${RSYNC} -rlpt -v -z --delete --port=$PORT ${SERVER}/data/structures/divided/mmCIF/ $MIRRORDIR > $LOGFILE 2>/dev/null


############################################################################
# Rsync only the XML format coordinates  /pub/pdb/data/structures/divided/XML (Aproximately 35 GB)
############################################################################
#${RSYNC} -rlpt -v -z --delete --port=$PORT ${SERVER}/data/structures/divided/XML/ $MIRRORDIR > $LOGFILE 2>/dev/null
