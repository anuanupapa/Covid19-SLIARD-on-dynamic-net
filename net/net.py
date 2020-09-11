import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
#tti=time.time()


#--------------------
#Function for pointer problems during array coping
#--------------------
def arr_copy(arr):
  duplicated_arr = np.zeros(np.shape(arr))
  if len(np.shape(arr))==2:
    for i in range(np.shape(arr)[0]):
      for j in range(np.shape(arr)[1]):
        duplicated_arr[i][j]=arr[i][j]
  else:
    for k in range(len(arr)):
      duplicated_arr[k] = arr[k]
  return(duplicated_arr)
#--------------------


#--------------------
#Probability distribution for random number generation in adjacency matrix
#--------------------
def prob_func(x,c,ld): # ld is strength of lockdown
  return(np.exp(-1*(c**(ld+1))*x)) #c will capture the strength of lockdown. 
#--------------------
#plot function
#T=np.linspace(0,1,100)
#plt.plot(T,prob_func(T,0.4))
#plt.show()
#--------------------


#--------------------
# Initial network construction
#--------------------
def construct_net(nComm_arr, ld=0, threshhold=0.5): # LockDStrength is the lockdown strength.
  # nComm_arr is the number of lower level community
  # contained in the level of index position
  nComm_arr=np.asarray(nComm_arr)
  n=1
  for i in nComm_arr:
    n=n*i
  #print(n)
    
  adj_mat=np.zeros((n,n))
  counter=0
  for level_density in nComm_arr: # level density stores the size of each level in terms of lower level community
    counter=counter+1 #counter takes care of the level number at which we the loop is currently in.
    level_totNode=1
    for i in range(counter): #level3 [2,3,4] 
      level_totNode=level_totNode*nComm_arr[i] #level_totNode stores the total number of nodes in one unit of the level
    for level_counter in range(int(n/level_totNode)): #number of these levels present in the community
      Ind=level_counter*level_totNode
      for row in np.arange(1,level_totNode+1,1):
        for col in np.arange(row+1,level_totNode+1,1):
          if adj_mat[Ind+row-1,Ind+col-1]==0:
            adj_mat[Ind+row-1,Ind+col-1]=prob_func(np.random.random(), (counter-1)+1, ld) #1 is bias for closest nodes
            #adj_mat[Ind+row-1,Ind+col-1]=1#counter #Uncomment this and print adj_mat to see the structure.
          else:
            pass

  #print(adj_mat)
  adj_matT=np.transpose(adj_mat)
  adj_mat_beauty = adj_mat + adj_matT
  adj_matFinale=np.copy(np.multiply(adj_mat_beauty>threshhold,1))

  h=np.sum(adj_matFinale, axis=1)
  plt.hist(h, bins=15)
  plt.show()

  G=nx.from_numpy_matrix(adj_matFinale)
  nx.draw(G, node_size=30, width=0.2, pos=nx.fruchterman_reingold_layout(G))
  plt.savefig('net8763_strucLD25.png')
  plt.show()

  #return(adj_matFinale)
#--------------------

if __name__ == "__main__":
  construct_net([8,7,6,3],ld=10)
