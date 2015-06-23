#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import os
import cfflib
from Tkinter import *
import tkFileDialog
import tkSimpleDialog
import tkMessageBox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import re

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
        n1 = int(event.xdata + 1)
        n2 = int(event.ydata + 1)
        
        # update the line positions
        self.lx.set_ydata(y )
        self.ly.set_xdata(x )
        
        try:
            self.txt.set_text( '%s / %s' % (str(G.node[n1]['dn_fsname']),str(G.node[n2]['dn_fsname'])) )
        except KeyError as e:
            self.txt.set_text( '%s / %s' % (str(G.node[str(n1)]['dn_fsname']),str(G.node[str(n2)]['dn_fsname'])) )
        else:
            self.txt.set_text( '%s / %s' % (str(n1),str(n2)) )

        event.canvas.draw()

    def onclick(self, event):    
        n1 = int(event.xdata) + 1
        n2 = int(event.ydata) + 1
 
        #print 'intXdata=%f, intYdata=%f'%(n1,n2)

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
            leg = ax.legend((str(G.node[n1]['dn_fsname']), str(G.node[n2]['dn_fsname'])),
                       loc='upper right', shadow=True)
            plt.show()



class Viewer(Tk):
    def __init__(self,parent):
        Tk.__init__(self,parent)
        self.parent = parent
        self.initialize()
    #self.testthis()
	
    def initialize(self):
        self.frame = Frame(self)
        self.frame.pack(side=TOP, expand=YES, fill=BOTH, padx=20, pady=5)
        
        self.cffpath = StringVar()
        self.l1 = Label(self.frame, text='Input file:')
        self.l1.grid(column=0,row=0,sticky='E',pady=10)
        self.e1 = Entry(self.frame,width=40,textvariable=self.cffpath)
        self.cffpath.set(u'Input Name')
        self.e1.grid(column=1,row=0,sticky='EW')
        
        self.b1 = Button(self.frame, text='Browse...',width=10,command=self.selectopen_callback)
        self.b1.grid(column=2,row=0)

        self.b2 = Button(self.frame, text="Plot", width=10,command=self.showconnectome)
        self.b2.grid(column=2, row=1)
        
        #resizeing options
        self.grid_columnconfigure(0,weight=1)
        #resize 
        self.resizable(True,True)
        self.update()
        self.geometry(self.geometry())


    def selectopen_callback(self):
        fname = tkFileDialog.askopenfilename(filetypes=[('cff files','.cff'),('graphml files','.graphml')]) #browse to file
        self.cffpath.set( fname ) #set the display
        self.infile = fname
        if re.search('\.graphml',fname):
            self.graph = nx.read_graphml(fname)
        elif re.search('\.cff',fname):
            self.cfile = cfflib.load(fname)
            self.cnet = self.cfile.get_by_name('connectome_freesurferaparc')
            self.cnet.load()
            self.graph = self.cnet.data
        self.addoptions()
            
    def addoptions(self):
        x,y = self.graph.edges()[0]
        keys = self.graph.edge[x][y].keys()
        keys.sort()
        keys.insert(0, "choose one")
        self.dvar = StringVar(self.frame)
        self.dvar.set(keys[0])
        self.drop = OptionMenu(self.frame, self.dvar, *keys)
        self.drop.grid(column=1,row=1)

    def showconnectome(self):
        fig = plt.figure(figsize=(9,8))
        global G
        G = self.graph
        arr = np.zeros((G.number_of_nodes(),G.number_of_nodes()))
        for x,y,d in G.edges_iter(data=True):
            arr[int(x) - 1][int(y) - 1] = float(d[str(self.dvar.get())])
            arr[int(y) - 1][int(x) - 1] = float(d[str(self.dvar.get())])

        plt.matshow(arr,cmap=plt.cm.RdBu_r,fignum=fig.number)
        plt.colorbar()
        cursor = Cursor(plt.gca())
        cid = fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)
        if re.search('\w+(rval)$',str(self.dvar.get())):
            plt.title("Functional Connectome")
            cid = fig.canvas.mpl_connect('button_press_event', cursor.onclick)
        else:
            plt.title("Structural Connectome")

        plt.show()


if __name__ == "__main__":
    app = Viewer(None)
    app.title('Connectome Viewer')
    app.mainloop()

