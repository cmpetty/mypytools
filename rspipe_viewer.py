#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import numpy as np
import os, sys
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor, Button
from optparse import OptionParser, OptionGroup

usage ="""
rspipe_viewer.py --graphml /path/to/subject.graphml --stat zrvalue

Program to display graphml output from resting_pipeline
 --stat is the statistic to view ( zrvalue, rvalue )
 --notimecourse is a flag to skip the timecourse display when clicking
 """
 
parser = OptionParser(usage=usage)
parser.add_option("-g","--graphml",  action="store", type="string", dest="graphml",help="full path to graphml file to display", metavar="/path/to/subject.graphml")
parser.add_option("-s","--stat",  action="store", type="string", dest="stat",help="statistic type to display ( zrvalue or rvalue ). default zr", metavar="zrvalue", default="zrvalue")
parser.add_option("--notimecourse",  action="store_true", dest="notime",help="do not display the seperate timecourse figures")

options, args = parser.parse_args()

if len(args) > 0:
    system.stderr.write("Too many arguments!  Try --help.")
    raise SystemExit()

if '-h' in sys.argv:
    parser.print_help()
    raise SystemExit()

if not (options.graphml) or '-help' in sys.argv:
    print "Input file ( --graphml ) is required to begin. Try --help "
    parser.print_help()
    raise SystemExit()

if not os.path.isfile(options.graphml):
    print "graphml does not exist!"
    parser.print_help()
    raise SystemExit()
    
    
class Cursor:
    def __init__(self, ax):
        self.ax = ax
        self.lx = ax.axhline(color='k',linewidth=2)  # the horiz line
        self.ly = ax.axvline(color='k',linewidth=2)  # the vert line

        # text location in axes coords
        self.txt = ax.text( 0, -0.05, '', transform=ax.transAxes)

    def mouse_move(self, event):
        if not event.inaxes: return

        x, y = event.xdata, event.ydata
        n1 = str(int(event.xdata) + 1)
        n2 = str(int(event.ydata) + 1)

        # update the line positions
        self.lx.set_ydata(y )
        self.ly.set_xdata(x )

        self.txt.set_text( '%s / %s'%((G.node[n1]['label']),str(G.node[n2]['label'])) )
        event.canvas.draw()

def onpress(event):
    thismaxX, thismaxY = (arr == arr.max()).nonzero()[0];    
    print thismaxX, thismaxY
    print event.mouseevent

def onclick(event):    
    n1 = str(int(event.xdata) + 1)
    n2 = str(int(event.ydata) + 1)

    if G.has_edge(n1,n2):
        #print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        #    event.button, event.x, event.y, event.xdata, event.ydata)
        #print 'intXdata=%f, intYdata=%f'%(int(event.xdata),int(event.ydata))
    
        fig2 = plt.figure(figsize=(10,4))
        ax = plt.subplot(111)
        arr1 = np.array([float(s) for s in G.node[n1]['timecourse'].split()])
        arr2 = np.array([float(s) for s in G.node[n2]['timecourse'].split()])
        ax.plot(arr1 - arr1.mean())
        ax.plot(arr2 - arr2.mean())
        ax.set_xlabel('time (tr)')
        ax.set_ylabel('de-meaned signal')
        ax.set_title("Timecourse")
        leg = ax.legend((str(G.node[n1]['label']), str(G.node[n2]['label'])),
                   loc='upper right', shadow=True, title=str("weight: " + str(G.edge[n1][n2][str(options.stat)])))
        plt.show()


#read the graphml
G = nx.read_graphml(options.graphml)

fig = plt.figure(figsize=(9,8))

arr = np.zeros((G.number_of_nodes(),G.number_of_nodes()))
for x,y,d in G.edges_iter(data=True):
    arr[int(x) - 1][int(y) - 1] = float(d[str(options.stat)])
    arr[int(y) - 1][int(x) - 1] = float(d[str(options.stat)])

plt.matshow(arr,cmap=plt.cm.RdBu_r,fignum=fig.number)
plt.title("Functional Connectome")
plt.colorbar()
#cursor = Cursor(plt.gca(), useblit=True, color='black', linewidth=2 )
cursor = Cursor(plt.gca())

#axmax = plt.axes([.05, 0.05, 0.05, 0.05])
#button = Button(axmax, label="max", color='w', hovercolor='blue')
#button.on_clicked(onpress)

if not options.notime:
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

cid2 = fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)

plt.show()



