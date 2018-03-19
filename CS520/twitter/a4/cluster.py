"""
cluster.py
"""
# Imports you'll need.
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
import os
import time
from TwitterAPI import TwitterAPI
import configparser
from operator import itemgetter, attrgetter
from collections import Counter, defaultdict
import copy
import pickle

def read_pickle_dump(filename):
    
    file = open(filename,'rb')
    fl = pickle.load(file)
    file.close()

    #print(fl[:2])
    return(fl)

    pass

 

def girvan_newman (G,num_clust):

    
    if (G.order() == 1) :
        return [G]

    # finding highest betweenenss edge 
    def find_betweenness(G1):
        betweenness = nx.edge_betweenness_centrality(G1)
        eb = sorted(betweenness.items(), key=lambda x: (-x[1]))
        #print(eb[:5])
        print('Highest approximate_betweenness-',eb[0][0])
        return(eb[0][0])

    # Each component is a separate community. We cluster each of these.
    
    components = [c for c in nx.connected_component_subgraphs(G)]
    
    # finding clusters
    while len(components) < num_clust: #just want 4 components     
        print('1.Length of component-',len(components))             
        G.remove_edge(*(find_betweenness(G)))        
        components = [c for c in nx.connected_component_subgraphs(G)]

    result = [c for c in components]

    
    return (result)

    
    pass
 


def find_clusters(H,num_clust):
    #print('Nodes = ',H.nodes())
    #print('Edges = ',H.edges())

    comp = girvan_newman(H,num_clust)
    return (comp)

    pass

def draw_network(G, users, filename):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).

    Methods you'll need include networkx.draw_networkx, plt.figure, and plt.savefig.

    Your figure does not have to look exactly the same as mine, but try to
    make it look presentable.
    """
    ###TODO


    Screen_names= ([u['screen_name'] for u in users])
    # print(Screen_names)
    
    label={}   


    for node in G.nodes():
        for name in Screen_names :
           if node==name :
              label[node]=name 

    #print(label)

    plt.figure(2)
    graph_pos = nx.spring_layout(G)

    nx.draw_networkx_edges(G, graph_pos,edge_color='black')
    nx.draw_networkx_nodes(G, graph_pos,label,node_size=200, node_color='blue', alpha=0.3)
    nx.draw_networkx_nodes(G, graph_pos,node_size=50, node_color='red', alpha=0.3)    
    nx.draw_networkx_labels(G, graph_pos,label, font_size=10, font_family='sans-serif')


    plt.savefig(filename)

    plt.show()
    
    pass

def find_compInfo(components):
    
    num_clust = len(components)
    print('Number of cluster = ',num_clust)

    i = 0
    while i < num_clust :
          print('Cluster %d Size = %d'%(i,components[i].order()))     
          i += 1
    pass

def create_pickle_dump(components):
    print('Creating Dump File')

    pickle.dump(components, open('Components.pkl', 'wb'))

    print('Pickle Dump Created Successfully!!')

    pass

def main():
    """ Main method. You should not modify this. """

    # read dump files
    users = read_pickle_dump('Channels.pkl')
    H = read_pickle_dump('Graph.pkl')
    
    # find clusters
    components = find_clusters(H,5)
    
    # create dump for components 
    create_pickle_dump(components)

    # find componenet sizes
    find_compInfo(components)  
   
    # draw clusters
    draw_network(H, users, 'clusters.png')

    print('Found Clusters Successfully!!')

if __name__ == '__main__':
    main()

