#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import os, sys
import nibabel
import re
import numpy as np
import networkx as nx
from optparse import OptionParser, OptionGroup

usage ="""
connectome2graphml.py -s stats.nii.gz -p prefix

Program to convert the 2D correlation matrix to a graphml file which can be used with graph-theory packages.
    --type is the statistics type ( ie: pvalue, rvalue, zrvalue )
    --labels and --text default to aal116 if no input is provided ( ie: aal116, raichle36 or full paths )
    --AC point defaults to 45,63,36 if undefined ( assuming MNI152_T1_2mm_brain )
    --thresh is the threshold to use for extracting edges. default is 0. ( ie: .99 for p > .01 )
"""
parser = OptionParser(usage=usage)
parser.add_option("-s", "--stats",  action="store", type="string", dest="stats",help="input correlation matrix", metavar="FILE")
parser.add_option("--type",  action="store", type="string", dest="stattype",help="type of statistics of input matrix", metavar="STRING")
parser.add_option("--thresh",  action="store", type="string", dest="threshold",help="threshold for edge extraction", metavar="STRING", default=0)
parser.add_option("-l", "--labels",  action="store", type="string", dest="label",help="label file used to produce the correlation matrix", metavar="FILE",default="aal116")
parser.add_option("-t","--text",  action="store",type="string", dest="labeltext",help="delimited text file associated with correlation labels ( contains region index and names ) ", metavar="FILE", default="aal116")
parser.add_option("--ac", action="store", type="string", dest="acpoint",help="anterior commissure point for coordinate conversion", metavar="STRING",default="45,63,36")
parser.add_option("-p","--prefix",  action="store",type="string", dest="prefix",help="prefix to name your output graphml", metavar="STRING")
parser.add_option("--above",  action="store_true", dest="above",help="your stats are above the diagonal in the connectome instead of below ( below is default )")

options, args = parser.parse_args()

if '-h' in sys.argv:
	parser.print_help()
	raise SystemExit()
if not (options.stats and options.prefix and options.stattype) or '-help' in sys.argv:
	print "The statistics file, statistics type and output prefix are required. Try --help "
	raise SystemExit()
if options.label is not None:
    if re.search('^aal116$',options.label):
        options.label = os.path.join('/usr','local','packages','biacpython','data','aal_MNI_V4.nii')
    elif re.search('^raichle36$',options.label):
        options.label = os.path.join('/usr','local','packages','biacpython','data','raichle_network_mask.nii')
    else:
        if not (os.path.isfile(options.label)):
            print "File does not exist: " + options.label
            raise SystemExit()
if options.labeltext is not None:
    if re.search('^aal116$',options.labeltext):
        options.labeltext = os.path.join('/usr','local','packages','biacpython','data','aal_MNI_V4.txt')
    elif re.search('^raichle36$',options.label):
        options.labeltext = os.path.join('/usr','local','packages','biacpython','data','raichle_network_mask.txt')
    else:
        if not (os.path.isfile(options.labeltext)):
            print "File does not exist: " + options.labeltext
            raise SystemExit()
for fname in [ options.stats, options.label, options.labeltext ]:
    if not ( os.path.isfile(fname)):
        print "File does not exist: " + fname
        raise SystemExit()






def grab_labels(thispath):
    mylabs = open(thispath,'r').readlines()
    labs = []

    for line in mylabs:
        if len(line.split('\t')) == 2:
            splitstuff = line.split('\t')
            splitstuff[0] = int(splitstuff[0])
            splitstuff[-1] = splitstuff[-1].strip()
            labs.append(splitstuff)

    return labs

#aalcenter = np.array([45,63,36])

aalcenter = options.acpoint.split(',')
for x in range(len(aalcenter)):
    aalcenter[x] = int(float(aalcenter[x]))


if __name__ == "__main__":
    labels = grab_labels(options.labeltext)

    thisnii = nibabel.load(options.label)
    niidata = thisnii.get_data()
    niihdr = thisnii.get_header()
    zooms = np.array(niihdr.get_zooms())

    G=nx.Graph(atlas=str(options.label))
    for lab in labels:
        #grab indices equal to label value
        x,y,z = (niidata == lab[0]).nonzero()
        centroid = np.array([int(x.mean()),int(y.mean()),int(z.mean())])
        c_cent_str = str((centroid - aalcenter)*(zooms.astype('int')))[1:-1].strip()
        lab.append( (centroid - aalcenter)*zooms.astype('int') )
        lab.append( 0 )
        G.add_node(lab[0],label=str(lab[1]),centroid=c_cent_str,intensityvalue=lab[0] )


    
    statobj = nibabel.load(options.stats)
    stats = statobj.get_data()
    statx,staty = statobj.shape
    #only grab results above a threshold
    sigx,sigy = (stats > float(options.threshold) ).nonzero()

    for idx in range(len(sigx)):
        if options.above is not None:
            if sigx[idx] < sigy[idx]:
                pval = stats[sigx[idx]][sigy[idx]]
                xidx = sigx[idx] + 1
                yidx = sigy[idx] + 1
                G.add_edge(xidx,yidx)
                G.edge[xidx][yidx][options.stattype]=str(pval)
        else:
            if sigx[idx] > sigy[idx]:
                pval = stats[sigx[idx]][sigy[idx]]
                xidx = sigx[idx] + 1
                yidx = sigy[idx] + 1
                G.add_edge(xidx,yidx)
                G.edge[xidx][yidx][options.stattype]=str(pval)



    B = nx.Graph.to_undirected(G)
    nx.write_graphml(B,options.prefix + '.graphml',encoding='utf-8', prettyprint=True)



