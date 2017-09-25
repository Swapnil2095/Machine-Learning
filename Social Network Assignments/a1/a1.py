# coding: utf-8

# # CS579: Assignment 1
#
# In this assignment, we'll implement community detection and link prediction algorithms using Facebook "like" data.
#
# The file `edges.txt.gz` indicates like relationships between facebook users. This was collected using snowball sampling: beginning with the user "Bill Gates", I crawled all the people he "likes", then, for each newly discovered user, I crawled all the people they liked.
#
# We'll cluster the resulting graph into communities, as well as recommend friends for Bill Gates.
#
# Complete the **15** methods below that are indicated by `TODO`. I've provided some sample output to help guide your implementation.


# You should not use any imports not listed here:
from collections import Counter, defaultdict, deque
import copy
import math
import networkx as nx
import urllib.request


## Community Detection

def example_graph():
    """
    Create the example graph from class. Used for testing.
    Do not modify.
    """
    g = nx.Graph()
    g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])
    return g

def bfs(graph, root, max_depth):
    """
    Perform breadth-first search to compute the shortest paths from a root node to all
    other nodes in the graph. To reduce running time, the max_depth parameter ends
    the search after the specified depth.
    E.g., if max_depth=2, only paths of length 2 or less will be considered.
    This means that nodes greather than max_depth distance from the root will not
    appear in the result.

    You may use these two classes to help with this implementation:
      https://docs.python.org/3.5/library/collections.html#collections.defaultdict
      https://docs.python.org/3.5/library/collections.html#collections.deque

    Params:
      graph.......A networkx Graph
      root........The root node in the search graph (a string). We are computing
                  shortest paths from this node to all others.
      max_depth...An integer representing the maximum depth to search.

    Returns:
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree

    In the doctests below, we first try with max_depth=5, then max_depth=2.

    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> sorted(node2distances.items())
    [('A', 3), ('B', 2), ('C', 3), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('A', 1), ('B', 1), ('C', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('A', ['B']), ('B', ['D']), ('C', ['B']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
<<<<<<< HEAD
  
=======
>>>>>>> template/master
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 2)
    >>> sorted(node2distances.items())
    [('B', 2), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('B', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('B', ['D']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
<<<<<<< HEAD
  
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 1)
    >>> sorted(node2distances.items())
    [('D', 1), ('E', 0), ('F', 1)]
    >>> sorted(node2num_paths.items())
    [('D', 1), ('E', 1), ('F', 1)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('D', ['E']), ('F', ['E'])]


    """
    ###TODO
    
    #print('%d-------->In bfs'%(max_depth))
    
    #print(graph.edges())
    node2distances = dict() 
    node2num_paths = dict()
    node2parents = dict() 
    depth_dict = dict() # keeping all depth and nodes present at that depth 
                        #-> helping to increment the depth variable

    d = deque()
    
    depth_dict.setdefault(0,[]).append(root) # 0th depth -->root 
 
    node2distances[root]=0
    node2num_paths[root]=1
    depth = 0 # started with zero
    
    if(max_depth == 1) :
    
       nbr_list = graph.neighbors(root)
       
       for nbr in nbr_list :                             
           node2distances[nbr] = 1 # node2distances[root]  + 1             
           node2parents.setdefault(nbr,[]).append(root) 
           node2num_paths[nbr] = 1 # node2num_paths[root]

    else:  
      d.append(root) # 1st element in dequeue is root
      while (len(d) >= 1 and depth < max_depth) :
 
          current = d.popleft()                                      
          nbr_list = graph.neighbors(current)
          
          #print('--->1Current=%s ---> list_lst=%s'%(current,nbr_list))                               
          for nbr in nbr_list : 
              #print('1.Inside list nbr',nbr)
              
              # filling values to node2distances
              if nbr not in node2distances.keys():
                 #print('2.Inside list nbr',nbr)                 
                 node2distances[nbr] = node2distances[current] + 1  
                 d.append(nbr)  # adding to dequeue
                 depth_dict.setdefault(depth+1,[]).append(nbr)  # additional dictionary
              
              # filling values to node2parents     
              if nbr not in node2parents.keys() :
                 node2parents.setdefault(nbr,[]).append(current) 
                    
              else :
                    if(node2distances[nbr] > node2distances[current]) :
                       node2parents[nbr].append(current)
             
              # filling values to node2num_paths
              if nbr not in node2num_paths.keys() :
                  node2num_paths[nbr] = node2num_paths[current]
                  
              else :
                  if(node2distances[nbr] > node2distances[current]) :    
                     node2num_paths[nbr] += node2num_paths[current]
                     
                     
          #print('depth_dict',depth_dict)
          #print('node2distances',node2distances)
          #print('node2parents',node2parents) 
          #print('node2num_paths',node2num_paths)  
   
          if depth in depth_dict.keys():      
                 if(depth_dict[depth][-1] == current) : # if all nodes are traversed for particular depth then increment depth by 1                                              
                      depth = depth + 1
                      #print('2.depth',depth)
                            
            
    if root in node2parents: 
        del node2parents[root]   
     
    node2distances[root]=0  # Extra care --not needed 
    node2num_paths[root]=1
 
    #print('%d-------->Out bfs'%(max_depth))                   
    #print('node2parents',node2parents) 
    #print('node2distances',node2distances) 
    #print('node2num_paths',node2num_paths) 
                   
    return(node2distances,node2num_paths,node2parents)
    
    pass 
    

=======
    """
    ###TODO
    pass
>>>>>>> template/master


def complexity_of_bfs(V, E, K):
    """
    If V is the number of vertices in a graph, E is the number of
    edges, and K is the max_depth of our approximate breadth-first
    search algorithm, then what is the *worst-case* run-time of
    this algorithm? As usual in complexity analysis, you can ignore
    any constant factors. E.g., if you think the answer is 2V * E + 3log(K),
    you would return V * E + math.log(K)
<<<<<<< HEAD

    >>> v = complexity_of_bfs(13, 23, 7)
    >>> type(v) == int or type(v) == float
    True
    

     
    """
    ###TODO
    
    #value1 = V * E + math.log(K)
    #print('1.value=.2f',value1)
   
    #value2 = (V + E) * math.log(K)
    #print('2.value=.2f',value2)
    
    return((V + E) * math.log(K))  # What I get after my complexity calcultion of my bfs algo
    
    
=======
    >>> v = complexity_of_bfs(13, 23, 7)
    >>> type(v) == int or type(v) == float
    True
    """
    ###TODO
>>>>>>> template/master
    pass


def bottom_up(root, node2distances, node2num_paths, node2parents):
    """
    Compute the final step of the Girvan-Newman algorithm.
    See p 352 From your text:
    https://github.com/iit-cs579/main/blob/master/read/lru-10.pdf
        The third and final step is to calculate for each edge e the sum
        over all nodes Y of the fraction of shortest paths from the root
        X to Y that go through e. This calculation involves computing this
        sum for both nodes and edges, from the bottom. Each node other
        than the root is given a credit of 1, representing the shortest
        path to that node. This credit may be divided among nodes and
        edges above, since there could be several different shortest paths
        to the node. The rules for the calculation are as follows: ...

    Params:
      root.............The root node in the search graph (a string). We are computing
                       shortest paths from this node to all others.
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    Returns:
      A dict mapping edges to credit value. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).

      Any edges excluded from the results in bfs should also be exluded here.

<<<<<<< HEAD

=======
>>>>>>> template/master
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> result = bottom_up('E', node2distances, node2num_paths, node2parents)
    >>> sorted(result.items())
    [(('A', 'B'), 1.0), (('B', 'C'), 1.0), (('B', 'D'), 3.0), (('D', 'E'), 4.5), (('D', 'G'), 0.5), (('E', 'F'), 1.5), (('F', 'G'), 0.5)]
<<<<<<< HEAD
     
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 2)
    >>> result = bottom_up('E', node2distances, node2num_paths, node2parents)
    >>> sorted(result.items())
    [(('B', 'D'), 1.0), (('D', 'E'), 2.5), (('D', 'G'), 0.5), (('E', 'F'), 1.5), (('F', 'G'), 0.5)]
  
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 1)
    >>> result = bottom_up('E', node2distances, node2num_paths, node2parents)
    >>> sorted(result.items())
    [(('D', 'E'), 1.0), (('E', 'F'), 1.0)]
  

 
    """
    ###TODO
    #print('---------->In bootom_up')
    #print('node2parents',node2parents) 
    #print('node2distances',node2distances) 
    #print('node2num_paths',node2num_paths)
    node2distances1 = {} # keeping sorted(node2distances) i.e. node2distances1 = node2distances              
    node2distances1 = sorted(node2distances.items(), key=lambda x:(-x[1],x[0]) , reverse=False)
    #print('type of node2distances',type(node2distances1)) 
    #print('node2distances1',node2distances1)        

    credit = {}       # node credit
    credit_dict = {}  # path credit
    children_list ={} # all children of particular node
    

    for node,distance in node2distances1 :
            #print('------>node=%s distance=%d'%(node,distance))   
            # finding children list for each node     
            for node1 in node2parents.keys(): 
                for parent in node2parents[node1] :
                    if (node == parent) :
                        children_list.setdefault(node,[]).append(node1)            
                 
            #print('node=%s--->children_list=%s'%(node,children_list))     
            if node not in children_list.keys(): # leaf node
               credit[node] = 1
               #print('----->node=%s ---> No child ---> credit = %s'%(node,credit))
          
            else: # not leaf node
                 #print('2.node=%s ---> Has children=%s'%(node,children_list[node])) 
                 #print('cedit list for children - ',credit) 
                 children_credits = 0                              
                                    
                 for child in children_list[node]  : 
                      #print('2.credit of child %s = %d'%(child,credit[child]))                            
                      path_credit = (credit[child])/(node2num_paths[child])
                      #print('Path credit=%.2f'%(path_credit))                                                                                                    
                      t = [node,child]  
                      t.sort()    #keeping sorted order                                               
                      credit_dict[t[0],t[1]] = path_credit
                      
                      children_credits += path_credit    # adding all child's credits
 
                 #print('children_credits',children_credits)               
                 credit[node] = children_credits + 1  # additional 1 for itself
       
 

    
    #print('2.credit_dict',credit_dict) 
    #print('2.credit',credit)          
    #print('---------->Out bootom_up')      
    return(credit_dict)                
    pass



=======
    """
    ###TODO
    pass


>>>>>>> template/master
def approximate_betweenness(graph, max_depth):
    """
    Compute the approximate betweenness of each edge, using max_depth to reduce
    computation time in breadth-first search.

    You should call the bfs and bottom_up functions defined above for each node
    in the graph, and sum together the results. Be sure to divide by 2 at the
    end to get the final betweenness.

    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.

    Returns:
      A dict mapping edges to betweenness. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).

    >>> sorted(approximate_betweenness(example_graph(), 2).items())
    [(('A', 'B'), 2.0), (('A', 'C'), 1.0), (('B', 'C'), 2.0), (('B', 'D'), 6.0), (('D', 'E'), 2.5), (('D', 'F'), 2.0), (('D', 'G'), 2.5), (('E', 'F'), 1.5), (('F', 'G'), 1.5)]
<<<<<<< HEAD


    >>> sorted(approximate_betweenness(example_graph(), 1).items())
    [(('A', 'B'), 1.0), (('A', 'C'), 1.0), (('B', 'C'), 1.0), (('B', 'D'), 1.0), (('D', 'E'), 1.0), (('D', 'F'), 1.0), (('D', 'G'), 1.0), (('E', 'F'), 1.0), (('F', 'G'), 1.0)]
     
     """
    ###TODO
    #print('%d-----------> In approximate_betweenness'%(max_depth))   
  
    credit ={}      # final credit after summation from each node
    credit_dict ={} # credit for each node

    
    for node in graph.nodes() :
       
       #print('For Node--->',node)
       #print('--------------->credit_dict=',len(credit_dict))
       node2distances, node2num_paths, node2parents = bfs(graph, node, max_depth)

       credit_dict = bottom_up(node, node2distances, node2num_paths, node2parents)
      
       #print('--------------->credit_dict=',len(credit_dict))
       #initailly credit is empty

       for edge in credit_dict.keys() :
           #print('edge=',edge)                                                                   
           if edge not in credit.keys() :
                credit[edge] = credit_dict[edge] #just store as it is if newly find  
           else :                
                credit[edge] =  credit[edge] + credit_dict[edge] #add if credit s caleculated in previous nodes  
       
       credit_dict.clear()   
           
    for edge in credit.keys() :
         credit[edge] = credit[edge]/2  #****** final IMP step 
    
    #print('--------------->credit',credit)
    #print('%d-----------> Out approximate_betweenness'%(max_depth))        
    return(credit)
    
    pass

=======
    """
    ###TODO
    pass


>>>>>>> template/master
def is_approximation_always_right():
    """
    Look at the doctests for approximate betweenness. In this example, the
    edge with the highest betweenness was ('B', 'D') for both cases (when
    max_depth=5 and max_depth=2).

    Consider an arbitrary graph G. For all max_depth > 1, will it always be
    the case that the edge with the highest betweenness will be the same
    using either approximate_betweenness verses the exact computation?
    Answer this question below.

    In this function, you just need to return either the string 'yes' or 'no'
    (no need to do any actual computations here).
    >>> s = is_approximation_always_right()
    >>> type(s)
    <class 'str'>
    """
    ###TODO
<<<<<<< HEAD
    
    answer = 'no'   # if max_depth parameter set then it gives different approximate_betweenness for some cases
                    # like 2 nodes lot greater than the max_depth apart from each other 
    return (answer)
    
=======
>>>>>>> template/master
    pass


def partition_girvan_newman(graph, max_depth):
    """
    Use your approximate_betweenness implementation to partition a graph.
    Unlike in class, here you will not implement this recursively. Instead,
    just remove edges until more than one component is created, then return
    those components.
    That is, compute the approximate betweenness of all edges, and remove
    them until multiple comonents are created.

    You only need to compute the betweenness once.
    If there are ties in edge betweenness, break by edge name (e.g.,
    (('A', 'B'), 1.0) comes before (('B', 'C'), 1.0)).

    Note: the original graph variable should not be modified. Instead,
    make a copy of the original graph prior to removing edges.
    See the Graph.copy method https://networkx.github.io/documentation/development/reference/generated/networkx.Graph.copy.html
    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.

    Returns:
      A list of networkx Graph objects, one per partition.

    >>> components = partition_girvan_newman(example_graph(), 5)
    >>> components = sorted(components, key=lambda x: sorted(x.nodes())[0])
    >>> sorted(components[0].nodes())
    ['A', 'B', 'C']
    >>> sorted(components[1].nodes())
    ['D', 'E', 'F', 'G']
<<<<<<< HEAD
    
    >>> components = partition_girvan_newman(example_graph(), 3)
    >>> components = sorted(components, key=lambda x: sorted(x.nodes())[0])
    >>> sorted(components[0].nodes())
    ['A', 'B', 'C']
    >>> sorted(components[1].nodes())
    ['D', 'E', 'F', 'G']
    
    >>> components = partition_girvan_newman(example_graph(), 1)
    >>> components = sorted(components, key=lambda x: sorted(x.nodes())[0])
    >>> sorted(components[0].nodes())
    ['A']
    >>> sorted(components[1].nodes())
    ['B', 'C', 'D', 'E', 'F', 'G']

     
      
    """
    ###TODO
    
    #print('%d----------->In partition_girvan_newman'%(max_depth))

    G=graph.copy()
    #print('1.Number of Edges-',len(G.edges()))
    
    if (G.order() == 1) :
        return [G]

     # Each component is a separate community. We cluster each of these.
    components = [c for c in nx.connected_component_subgraphs(G)]
     
     # sorted credits
    eb = sorted((approximate_betweenness(G,max_depth)).items(), key=lambda x: (-x[1],x[0][0],x[0][1]))
     
    i = 0  # for next high credit if graph not divided into components not more than 1
    while len(components)==1 and (i < len(eb)): #just want 2 components 
    
        #print('1.Length of component-',len(components))       
        
        # taking one edge at a time to remove              
        edge_to_remove = eb[i][0]
        #print('Highest approximate_betweenness-',edge_to_remove)
               
        G.remove_edge(*edge_to_remove)        
        i = i + 1     #eb.pop(edge_to_remove, None)
        
        # checking 2 components created after removal of edge or not
        components = [c for c in nx.connected_component_subgraphs(G)]
        
        #print('2.Length of component-',len(components))
        #print('2.Number of Edges-',len(G.edges()))

    result = [c for c in components]
    #print('components=',result)

    #print('%d----------->Out partition_girvan_newman'%(max_depth))
    return (result)

    
    pass
    
=======
    """
    ###TODO
    pass

>>>>>>> template/master
def get_subgraph(graph, min_degree):
    """Return a subgraph containing nodes whose degree is
    greater than or equal to min_degree.
    We'll use this in the main method to prune the original graph.

    Params:
      graph........a networkx graph
      min_degree...degree threshold
    Returns:
      a networkx graph, filtered as defined above.

    >>> subgraph = get_subgraph(example_graph(), 3)
    >>> sorted(subgraph.nodes())
    ['B', 'D', 'F']
    >>> len(subgraph.edges())
    2
<<<<<<< HEAD

    >>> subgraph = get_subgraph(example_graph(), 4)
    >>> sorted(subgraph.nodes())
    ['D']
    >>> len(subgraph.edges())
    0
    
    
    """
    ###TODO
    #print('-------------> In get_subgraph')
    #print('Number of edges=',len(graph.edges()))

    deg = graph.degree()
    
    to_remove = [n for n in deg if (deg[n] < min_degree) ]
    
    graph.remove_nodes_from(to_remove)  


    #print('Number of edges=',len(graph.edges()))

    #print('-------------> Out get_subgraph')
    return(graph)
    
    pass

=======
    """
    ###TODO
    pass


>>>>>>> template/master
""""
Compute the normalized cut for each discovered cluster.
I've broken this down into the three next methods.
"""

def volume(nodes, graph):
    """
    Compute the volume for a list of nodes, which
    is the number of edges in `graph` with at least one end in
    nodes.
    Params:
      nodes...a list of strings for the nodes to compute the volume of.
      graph...a networkx graph

    >>> volume(['A', 'B', 'C'], example_graph())
    4
<<<<<<< HEAD
    
    >>> volume(['B'], example_graph())
    3
    
    >>> volume(['D','E', 'F', 'G'], example_graph())
    6

    >>> volume(['B', 'D', 'G'], example_graph())
    7
    
    >>> volume(['F', 'G'], example_graph())
    4

    
    """
    ###TODO
       
    #print('---------> In volume')
   
    G = graph.copy()
    initial_num_of_edges = graph.number_of_edges()
    
    G.remove_nodes_from(nodes)   #***********
   
    num_of_edges = G.number_of_edges()
  
    link_counter = initial_num_of_edges - num_of_edges
    
    #print('link_counter=',link_counter)
    #print('initial_num_of_edges=',initial_num_of_edges)
    #print('num_of_edges=',num_of_edges)
    
    #print('---------> Out volume')
    return(link_counter)
      
    pass

=======
    """
    ###TODO
    pass


>>>>>>> template/master
def cut(S, T, graph):
    """
    Compute the cut-set of the cut (S,T), which is
    the set of edges that have one endpoint in S and
    the other in T.
    Params:
      S.......set of nodes in first subset
      T.......set of nodes in second subset
      graph...networkx graph
    Returns:
      An int representing the cut-set.

    >>> cut(['A', 'B', 'C'], ['D', 'E', 'F', 'G'], example_graph())
    1
<<<<<<< HEAD
    
    >>> cut(['F', 'G'], ['A', 'B', 'C', 'D', 'E'], example_graph())
    3
        
    >>> cut(['E', 'F'], ['A', 'B', 'C', 'D', 'G'], example_graph())
    3
    
    >>> cut(['E'], ['A', 'B', 'C', 'D', 'F', 'G'], example_graph())
    2
 
 
    """
    ###TODO
    
    #print('------------->In cut')      
   
    G1 = graph.copy()
    G2 = graph.copy()
    initial_num_of_edges = graph.number_of_edges()
    #print('initial_num_of_edges=',initial_num_of_edges)    
   
    G1.remove_nodes_from(S)  
    G1_edges = G1.number_of_edges()
    #print('S=',len(G1.edges()))
    
    G2.remove_nodes_from(T)
    G2_edges = G2.number_of_edges()  
    #print('T=',len(G2.edges())) 
         
    link_counter = initial_num_of_edges - (G1_edges + G2_edges)
    #print('cut_Size=',link_counter) 
    
    
    #print('------------->Out cut')   
    return(link_counter)
    
    pass

=======
    """
    ###TODO
    pass


>>>>>>> template/master
def norm_cut(S, T, graph):
    """
    The normalized cut value for the cut S/T. (See lec06.)
    Params:
      S.......set of nodes in first subset
      T.......set of nodes in second subset
      graph...networkx graph
    Returns:
      An float representing the normalized cut value

<<<<<<< HEAD
    >>> norm_cut(['A', 'B', 'C'], ['D', 'E', 'F', 'G'], example_graph())
    0.41666666666666663

    >>> norm_cut(['F', 'G'], ['A', 'B', 'C', 'D', 'E'], example_graph())
    1.125

    """
    ###TODO
    
    #print('------------->In norm_cut')   
    volume1 = volume(S,graph) # 1st set
    volume2 = volume(T,graph)   # 2nd set 
    cut_size = cut(S, T, graph) # cut size
    
   
    if(volume1 == 0):
        part1 = 0.0
    else :
        part1 = 1. * (cut_size/volume1)
    
    if(volume2 == 0):
        part2 = 0.
    else : 
        part2 = 1. * (cut_size/volume2)
 
    norm_value = part1 + part2
    

    #print('------------->Out norm_cut')
    #print('Norm_value = ',norm_value)
    return(norm_value)
=======
    """
    ###TODO
>>>>>>> template/master
    pass


def score_max_depths(graph, max_depths):
    """
    In order to assess the quality of the approximate partitioning method
    we've developed, we will run it with different values for max_depth
    and see how it affects the norm_cut score of the resulting partitions.
    Recall that smaller norm_cut scores correspond to better partitions.

    Params:
      graph........a networkx Graph
      max_depths...a list of ints for the max_depth values to be passed
                   to calls to partition_girvan_newman

    Returns:
      A list of (int, float) tuples representing the max_depth and the
      norm_cut value obtained by the partitions returned by
      partition_girvan_newman. See Log.txt for an example.
<<<<<<< HEAD
 

      
    """
    ###TODO
    #print('------------->In score_max_depths')
    
    list1 = [] # keeping results of norm_value
    

    for depth in max_depths:
        #print('For MAX_depth =',depth)
        # 1st call approximate partitioning method
        components = partition_girvan_newman(graph, depth)
    
        # 2nd call norm_cut  
     
        norm_value = norm_cut(components[0].nodes(), components[1].nodes(), graph)
        
        #storing norm_values to list
        list1.append([depth,norm_value])
           
        # see whether smaller norm_cut scores correspond to better partitions 
  
    #print('------------->Out score_max_depths')
    #print('List of Norm-value = ',list1)
    return(list1)
    
=======
    """
    ###TODO
>>>>>>> template/master
    pass


## Link prediction

# Next, we'll consider the link prediction problem. In particular,
# we will remove 5 of the accounts that Bill Gates likes and
# compute our accuracy at recovering those links.

def make_training_graph(graph, test_node, n):
    """
    To make a training graph, we need to remove n edges from the graph.
    As in lecture, we'll assume there is a test_node for which we will
    remove some edges. Remove the edges to the first n neighbors of
    test_node, where the neighbors are sorted alphabetically.
    E.g., if 'A' has neighbors 'B' and 'C', and n=1, then the edge
    ('A', 'B') will be removed.

    Be sure to *copy* the input graph prior to removing edges.

    Params:
      graph.......a networkx Graph
      test_node...a string representing one node in the graph whose
                  edges will be removed.
      n...........the number of edges to remove.

    Returns:
      A *new* networkx Graph with n edges removed.

    In this doctest, we remove edges for two friends of D:
<<<<<<< HEAD

   In this doctest, we remove edges for two friends of D:
=======
>>>>>>> template/master
    >>> g = example_graph()
    >>> sorted(g.neighbors('D'))
    ['B', 'E', 'F', 'G']
    >>> train_graph = make_training_graph(g, 'D', 2)
    >>> sorted(train_graph.neighbors('D'))
    ['F', 'G']
<<<<<<< HEAD
    
    >>> g = example_graph()
    >>> sorted(g.neighbors('B'))
    ['A', 'C', 'D']
    >>> train_graph = make_training_graph(g, 'B', 2)
    >>> sorted(train_graph.neighbors('B'))
    ['D']
 
    >>> g = example_graph()
    >>> sorted(g.neighbors('F'))
    ['D', 'E', 'G']
    >>> train_graph = make_training_graph(g, 'F', 2)
    >>> sorted(train_graph.neighbors('F'))
    ['G']

    >>> g = example_graph()
    >>> sorted(g.neighbors('B'))
    ['A', 'C', 'D']
    >>> train_graph = make_training_graph(g, 'B', 1)
    >>> sorted(train_graph.neighbors('B'))
    ['C', 'D']
    
    >>> g = example_graph()
    >>> sorted(g.neighbors('G'))
    ['D', 'F']
    >>> train_graph = make_training_graph(g, 'G', 1)
    >>> sorted(train_graph.neighbors('G'))
    ['F']
  
    
    """
    ###TODO
    #print('------------->In make_training_graph')
    
    G = graph.copy() 
    num_of_cut = 0
    neighbours= sorted(G.neighbors(test_node))[:n] # first n sorted neighbors
  
   
    while (num_of_cut < n) :
       
       nbr = neighbours[num_of_cut] # taking neighbor one by one
       G.remove_edge(test_node,nbr)
       num_of_cut = num_of_cut + 1


    #print('-------------> Out make_training_graph')
    return(G)
=======
    """
    ###TODO
>>>>>>> template/master
    pass



def jaccard(graph, node, k):
    """
    Compute the k highest scoring edges to add to this node based on
    the Jaccard similarity measure.
    Note that we don't return scores for edges that already appear in the graph.

    Params:
      graph....a networkx graph
      node.....a node in the graph (a string) to recommend links for.
      k........the number of links to recommend.

    Returns:
      A list of tuples in descending order of score representing the
      recommended new edges. Ties are broken by
      alphabetical order of the terminal node in the edge.

    In this example below, we remove edges (D, B) and (D, E) from the
    example graph. The top two edges to add according to Jaccard are
    (D, E), with score 0.5, and (D, A), with score 0. (Note that all the
    other remaining edges have score 0, but 'A' is first alphabetically.)

    >>> g = example_graph()
    >>> train_graph = make_training_graph(g, 'D', 2)
    >>> jaccard(train_graph, 'D', 2)
    [(('D', 'E'), 0.5), (('D', 'A'), 0.0)]
<<<<<<< HEAD
    
    >>> g = example_graph()
    >>> train_graph = make_training_graph(g, 'B', 2)
    >>> jaccard(train_graph, 'B', 3)
    [(('B', 'E'), 0.5), (('B', 'G'), 0.5), (('B', 'F'), 0.3333333333333333)]

    >>> g = example_graph()
    >>> train_graph = make_training_graph(g, 'B', 1)
    >>> jaccard(train_graph, 'B', 3)
    [(('B', 'A'), 0.5), (('B', 'E'), 0.3333333333333333), (('B', 'G'), 0.3333333333333333)]

    
    """
    ###TODO
    
    
    neighbors = set(graph.neighbors(node)) #first set
    

    scores = [] # keeping all scores
    
    FinalScores = [] # keeping only k highest scores
    
    #calcualting scores
    for node1 in graph.nodes():
      if (node1 != node)  :
        neighbors2 = set(graph.neighbors(node1)) #2nd set
      
        value = (1. * (len(neighbors & neighbors2) / len(neighbors | neighbors2)))
             
        scores.append(((node,node1),value))
      
    
    # removing the edges present in the graphs from scores list
    for edge in graph.edges() :      
        node3 = edge[0]
        node4 = edge[1]
        for edge_tpl in scores :
           node1 = edge_tpl[0][0]
           node2 = edge_tpl[0][1]
           if ((node1 == node3 and node2 == node4 ) or (node1 == node4 and node2 == node3 ))  :                                            
               scores.remove(edge_tpl)
    
            
    scores.sort(key=lambda x: (-x[1],x[0][0],x[0][1]))

    FinalScores = scores[:k] # after sorting just taking first k number of links to recommend.
                      
    return (FinalScores)

    
    pass

=======
    """
    ###TODO
    pass


>>>>>>> template/master
# One limitation of Jaccard is that it only has non-zero values for nodes two hops away.
#
# Implement a new link prediction function that computes the similarity between two nodes $x$ and $y$  as follows:
#
# $$
# s(x,y) = \beta^i n_{x,y,i}
# $$
#
# where
# - $\beta \in [0,1]$ is a user-provided parameter
# - $i$ is the length of the shortest path from $x$ to $y$
# - $n_{x,y,i}$ is the number of shortest paths between $x$ and $y$ with length $i$


def path_score(graph, root, k, beta):
    """
    Compute a new link prediction scoring function based on the shortest
    paths between two nodes, as defined above.

    Note that we don't return scores for edges that already appear in the graph.

    This algorithm should have the same time complexity as bfs above.

    Params:
      graph....a networkx graph
      root.....a node in the graph (a string) to recommend links for.
      k........the number of links to recommend.
      beta.....the beta parameter in the equation above.

    Returns:
      A list of tuples in descending order of score. Ties are broken by
      alphabetical order of the terminal node in the edge.

    In this example below, we remove edge (D, F) from the
    example graph. The top two edges to add according to path_score are
    (D, F), with score 0.5, and (D, A), with score .25. (Note that (D, C)
    is tied with a score of .25, but (D, A) is first alphabetically.)

    >>> g = example_graph()
    >>> train_graph = g.copy()
    >>> train_graph.remove_edge(*('D', 'F'))
    >>> path_score(train_graph, 'D', k=4, beta=.5)
    [(('D', 'F'), 0.5), (('D', 'A'), 0.25), (('D', 'C'), 0.25)]
<<<<<<< HEAD

    """
    ###TODO
    #print(graph.edges())
    #print(root)
    #print(k)
    #print(beta)
    
    S_list = []    # keeping all scores S(x,y)
    node1 = root  #******
    
    # used same bfs function just ignored unwanted part here
    # so same complexity as bfs as required
    def search_paths(Graph, start ,end):
      
      number_of_paths  = 0  # storing number of shortest paths
      short_length = 0      # shortest path length
      node2distances = dict() 
      node2num_paths = dict()

    
      d = deque()
      node2distances[start]=0
      node2num_paths[start]=1
      
      d.append(start)
      
      while (len(d) >= 1) :
 
          current = d.popleft()                                      
          nbr_list = graph.neighbors(current)
                                             
          for nbr in nbr_list :
              
              if nbr not in node2distances.keys():                        
                 node2distances[nbr] = node2distances[current] + 1  
                 d.append(nbr)                   
                                
              if nbr not in node2num_paths.keys() :
                  node2num_paths[nbr] = node2num_paths[current]
              else :
                  if(node2distances[nbr] > node2distances[current]) :    
                     node2num_paths[nbr] += node2num_paths[current]
                     
             
                               
      short_length =  node2distances[end]
      number_of_paths = node2num_paths[end]                            
      #print('node2parents',node2parents) 
      #print('node2distances',node2distances) 
      #print('node2num_paths',node2num_paths) 
                   
      return(short_length,number_of_paths)  
                 
      
    for node2 in graph.nodes() :           
         if (node1,node2) not in graph.edges()  and (node1 != node2) and (node2,node1) not in graph.edges() :
                
                 #print('(node1,node2)=(%s,%s)'%(node1,node2))
                                  
                 short_length,number_of_paths = search_paths(graph,node1,node2)  # calling bfs function to get short path length and number of short paths                
                 
                 #print('short_length ---->=',short_length)

                 #print('number_of_paths---->=', number_of_paths)
                 
                 value1 = pow(beta,short_length)#(beta ** short_length) 
                 value2 = number_of_paths
                 
                 value = value1 * value2
                   
                 S_list.append(((node1,node2),value))
                 
                 #print('Value1=',value1)
                 #print('Value2=',value2)
                 #print('Score Value=',value)
        
    S_list.sort(key=lambda x: (-x[1],x[0][0],x[0][1]))

    FinalScores = S_list[:k]   # after sorting just taking first k number of links to recommend.  
    
    return(FinalScores)
    pass

=======
    """
    ###TODO
    pass


>>>>>>> template/master
def evaluate(predicted_edges, graph):
    """
    Return the fraction of the predicted edges that exist in the graph.

    Args:
      predicted_edges...a list of edges (tuples) that are predicted to
                        exist in this graph
      graph.............a networkx Graph

    Returns:
      The fraction of edges in predicted_edges that exist in the graph.

    In this doctest, the edge ('D', 'E') appears in the example_graph,
    but ('D', 'A') does not, so 1/2 = 0.5

    >>> evaluate([('D', 'E'), ('D', 'A')], example_graph())
    0.5
<<<<<<< HEAD
    
    >>> evaluate([('D', 'E'), ('D', 'A'), ('F', 'G')], example_graph())
    0.6666666666666666
    
    >>> evaluate([('D', 'E')], example_graph())
    1.0

    >>> evaluate([('D', 'E'), ('D', 'A'), ('F', 'G'), ('D', 'F')], example_graph())
    0.75
  
    
    """
    ###TODO
  
    size = len(predicted_edges) # size of predicted_edges
      
    presence = 0 # to calculate number of edges present in graph
        
    for edge1,edge2 in predicted_edges :           
        if graph.has_edge(edge1,edge2) :
                  presence = presence + 1
                 
                  
    fraction = 1. * (presence/size)

    return(fraction)
=======
    """
    ###TODO
>>>>>>> template/master
    pass


"""
Next, we'll download a real dataset to see how our algorithm performs.
"""
def download_data():
    """
    Download the data. Done for you.
    """
    urllib.request.urlretrieve('http://cs.iit.edu/~culotta/cs579/a1/edges.txt.gz', 'edges.txt.gz')


def read_graph():
    """ Read 'edges.txt.gz' into a networkx **undirected** graph.
    Done for you.
    Returns:
      A networkx undirected graph.
    """
    return nx.read_edgelist('edges.txt.gz', delimiter='\t')


def main():
    """
    FYI: This takes ~10-15 seconds to run on my laptop.
    """
    download_data()
    graph = read_graph()
    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
    subgraph = get_subgraph(graph, 2)
    print('subgraph has %d nodes and %d edges' %
          (subgraph.order(), subgraph.number_of_edges()))
    print('norm_cut scores by max_depth:')
    print(score_max_depths(subgraph, range(1,5)))
    clusters = partition_girvan_newman(subgraph, 3)
    print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
          (clusters[0].order(), clusters[1].order()))
    print('cluster 2 nodes:')
    print(clusters[1].nodes())

    test_node = 'Bill Gates'
    train_graph = make_training_graph(subgraph, test_node, 5)
    print('train_graph has %d nodes and %d edges' %
          (train_graph.order(), train_graph.number_of_edges()))


    jaccard_scores = jaccard(train_graph, test_node, 5)
    print('\ntop jaccard scores for Bill Gates:')
    print(jaccard_scores)
    print('jaccard accuracy=%g' %
          evaluate([x[0] for x in jaccard_scores], subgraph))

    path_scores = path_score(train_graph, test_node, k=5, beta=.1)
    print('\ntop path scores for Bill Gates for beta=.1:')
    print(path_scores)
    print('path accuracy for beta .1=%g' %
          evaluate([x[0] for x in path_scores], subgraph))


if __name__ == '__main__':
    main()
