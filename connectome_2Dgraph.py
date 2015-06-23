#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import numpy as np
import networkx as nx
import community  #( http://perso.crans.org/aynaud/communities/index.html )
import os, sys
import matplotlib.pyplot as plt
import pylab
from matplotlib.widgets import RadioButtons

from optparse import OptionParser, OptionGroup
from xml.dom import minidom


usage ="""
connectome_2Dgraph.py --graphml /path/to/connectome.graphml --edgeval pvalue --thresh 0.9998

Generates a graphy-theroy 2D graph with nodes/edges above a certain threshold.
    Communities are colored by a Louvian algorythm.
    Clicking a specific node with produce another graph containing only nodes that are directly connected to the area selected.

"""

parser = OptionParser(usage=usage)
parser.add_option("-g", "--graphml",  action="store", type="string", dest="graphml",help="graphml file containing connectome information", metavar="/path/to/graphml")
parser.add_option("-t", "--thresh",  action="store", type="string", dest="thresh",help="threshold to use for displaying significant connection", metavar="0.9998")
parser.add_option("-e", "--edgeval",  action="store", type="string", dest="edgeval",help="the statistic that is represented by the connections in the graphml", metavar="pvalue", default='pvalue')

options, args = parser.parse_args()

if '-h' in sys.argv:
    parser.print_help()
    raise SystemExit()
if not (options.graphml or options.edgeval ) or '-help' in sys.argv:
    print "Input file ( --graphml ) and value type ( --edgeval ) are required to begin. Try --help "
    raise SystemExit()


if options.graphml is not None:
    if not ( os.path.isfile( options.graphml )):
        print "Input file ( --graphml ) does not exist: ", options.graphml
        raise SystemExit()
    
    (gmlhead, gmltail) = os.path.split(str(options.graphml))
    if not ( gmltail.endswith('.graphml') ):
        print "Input file ( --graphml ) is not a '.grapml' file: ", options.graphml
        raise SystemExit()

    thisgraphml = options.graphml

if options.thresh is not None:
    thresh = float(options.thresh)

hascentroid = None
haslabel = None

if options.edgeval is not None:
    dom = minidom.parse(options.graphml)
    keys = dom.getElementsByTagName('key')
    for k in keys:
        if k.hasAttribute('attr.name'):
            if ( options.edgeval == k.getAttribute('attr.name') ):
                edgeval = str(options.edgeval)
            elif ( k.getAttribute('attr.name') in ['centroid','dn_position'] ):
                hascentroid = k.getAttribute('attr.name')
            elif ( k.getAttribute('attr.name') in ['label','dn_fsname'] ):
                haslabel = k.getAttribute('attr.name')


def radiofunc(radiolabel):
    global radioval
    radioval = radiolabel


#define the picker function
def onpick(event):
    thisline = event.artist
    mouseevent = event.mouseevent
    ind = event.ind[0]

    if ( radioval == 'subgraph'):
        neigh = G.neighbors(nodes[ind])
        N = nx.ego_graph(G,nodes[ind],center=True)

        #create new figure, set size
        fig2=plt.figure(figsize=(12,10))
        ax2 = fig2.add_subplot(111)
        ax2.set_axis_off()
        fig2.set_facecolor('w')
        ax2.set_title("ego center: " + str(G.node[nodes[ind]][haslabel]))

        labs={}
        for v in N:
            labs[v]=str(N.node[v][haslabel])

        nx.draw_networkx_nodes(N,pos,ax=ax2,
            node_size=[float(N.degree(v))*50 for v in neigh],
            node_shape='o',
            node_color='blue',
            alpha = .5)

        nx.draw_networkx_nodes(N,pos,ax=ax2,
            nodelist=[ nodes[ind] ],
            node_size=[float(N.degree(nodes[ind])*50)],
            node_color='r',
            alpha = 1)

        nx.draw_networkx_edges(N, pos, ax=ax2,
            alpha=.4,
            width=1,
            style='dashed')

        nx.draw_networkx_labels(N, pos, ax=ax2,
            font_color='black',
            font_size='16',
            font_family='sans-serif',
            labels=labs)

        plt.show()

        return True
    
    if ( radioval == 'lesion' ):
        CG = G.copy()
        neigh = G.neighbors(nodes[ind])

        CG.remove_nodes_from(neigh)
        CG.remove_node(nodes[ind])
        
        #create new figure, set size
        fig2=plt.figure(figsize=(12,10))
        ax2 = fig2.add_subplot(111)
        ax2.set_axis_off()
        fig2.set_facecolor('w')
        #ax2.set_title("ego center: " + str(G.node[nodes[ind]]['label']))

        #get the colors
        node_colors = []
        for thisnode in CG.nodes():
            for idx in range(len(mycomm)):
                if ( thisnode in mycomm[idx] ):
                    node_colors.append( idx )

        lnodes = list(CG.nodes());
        #draw the graph, plot size by degree*40
        artist = nx.draw_networkx_nodes(CG,pos,ax=ax2,
            nodelist=lnodes,
            node_color=node_colors,
            node_size=[float(CG.degree(v))*50 for v in CG],
            node_shape='o',
            alpha = .7)

        nx.draw_networkx_labels(CG, pos, ax=ax2,
            font_color='white')

        nx.draw_networkx_edges(CG, pos, ax=ax2,
            alpha=.9)

        plt.show()


        return True



if ( thisgraphml or thresh or edgeval ) is None:
    print "Can not continue, invalid options given. Check threshold and the edgevalue defined."
    raise SystemExit()

if ( haslabel or hascentroid ) is None:
    print "Centroids or labels are undefined. Please check the creation of your graphml ( connectome2graphml.py )."
    raise SystemExit()
    




#read the graphml
G = nx.read_graphml(thisgraphml)

#convert unicode to floats
for here in [e for e in G.edges_iter(data=True)]:
    here[-1][edgeval] = float(here[-1][edgeval])
    
#remove values less than thresh
for here in [e for e in G.edges_iter(data=True)]:
    if (here[-1][edgeval] < thresh):
        G.remove_edge(here[0],here[1])

#clip out isolates ( nodes without neighbors/connections )
G.remove_nodes_from(nx.isolates(G))

#calculate the communities
partition = community.best_partition( G )

#get the positions of each node
pos = {}
for node in G.nodes():
    #split the txt on whitespace, then grab X,Y positions
    if hascentroid == 'centroid':
        pos[node] = np.array([ float(str(G.node[node][hascentroid]).split()[0]), float(str(G.node[node][hascentroid]).split()[1])])
    else:
        pos[node] = np.array([ float(str(G.node[node][hascentroid]).split(',')[0]), float(str(G.node[node][hascentroid]).split(',')[1])])


#set the figure size
fig=plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)
ax.set_axis_off()
fig.set_facecolor('w')
ax.set_title("p < " + str(1 - thresh) )

mycomm = {}
memnames = {}

#get the community members
for i in set(partition.values()):
    members = list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == i]
    mycomm[i] = members
    these = []
    for mem in members:
        these.append( str(G.node[mem][haslabel]) )
    memnames[i] = these


#get the colors
node_colors = []
for thisnode in G.nodes():
    #for idx in range(len(mycomm)):
    for idx,v in enumerate(sorted(mycomm, key=lambda k: len(mycomm[k]), reverse=True)):
        if ( thisnode in mycomm[idx] ):
            node_colors.append( idx )

#get names
nodes = list(G.nodes());

#draw the graph, plot size by degree*50
artist = nx.draw_networkx_nodes(G,pos,ax=ax,
    nodelist=nodes,
    node_color=node_colors,
    node_size=[float(G.degree(v))*50 for v in G],
    node_shape='o',
    alpha = .7)

#draw the labels
nx.draw_networkx_labels(G, pos, ax=ax,
    font_color='white')

#draw the edges
nx.draw_networkx_edges(G, pos, ax=ax,
    alpha=.9)

#add the picker radius ( ability to click )
artist.set_picker(5)

ax2 = fig.add_subplot(111)
axcolor = 'white'
paxes = pylab.axes([0.05, 0.05, 0.1, 0.1], axisbg=axcolor)
radio = RadioButtons(paxes,('subgraph','lesion'))
radioval = str('subgraph')
radio.on_clicked(radiofunc)


#turn on clicker
fig.canvas.mpl_connect('pick_event', onpick)

#show it
plt.show()







