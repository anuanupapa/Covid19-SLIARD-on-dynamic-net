import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from net import *
import networkx as nx


tti = time.time()

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

#Note: For making a succesful lockdown scenario, the contacts between individual
#      communitites keep increasing with level number. 

LDnormal=1
levs=[3,4,5,6,2,2]
AM=construct_net(levs, ld=LDnormal)
S_ini = (3*4*5*6*2*2)-1#int(input('S initial : '))
L_ini = 1 #int(input('L initial : '))
I_ini = 0
A_ini = 0
R_ini = 0
D_ini = 0
n=S_ini+L_ini
print(n)

SarrPrimer1=np.ones((S_ini,1))
SarrPrimer2=np.zeros((L_ini,1))
Sarr=np.concatenate((SarrPrimer1,SarrPrimer2))
np.random.shuffle(Sarr)
LarrPrimer1=np.zeros((S_ini,1))
LarrPrimer2=np.ones((L_ini,1))
Larr=np.concatenate((LarrPrimer1,LarrPrimer2))
np.random.shuffle(Larr)
Iarr=np.zeros((L_ini+S_ini,1))
Aarr=np.zeros((S_ini+L_ini,1))
Rarr=np.zeros((S_ini+L_ini,1))
Darr=np.zeros((S_ini+L_ini,1))

#The infected and the assympotomatics have to stay in the state for a while
#to spread the infection.
betaI=.06 #.25#.15
betaA=.1#.1#.06
thetaI=.04 #0.08#.05
thetaA=.08 #0.16#.1
alphaI=.015 # 0.05#.05
alphaA=.015 #0.1#.1
gamma=.002 #0.007


TotS = [np.sum(Sarr)]
TotL = [np.sum(Larr)]
TotI = [0]
TotA = [0]
TotR = [0]
TotD = [0]

#defining the integral params
tstart=0
tend=200.0
dt = 0.1
t=np.arange(tstart, tend, dt)
N = int((tend - tstart)/dt)
print('TotIter:',N)

def func_dot(arr,AdjMat):
  arr[3]=np.where(arr[3]>1e-5,arr[3],0) #Makes the nodes with values of A less than 1e-5 to 0.
  dsdt=-1*betaA*np.multiply(arr[0],np.dot(AdjMat,arr[3]))-1*betaI*np.multiply(arr[0],np.dot(AdjMat,arr[2]))
  dldt=betaA*np.multiply(arr[0],np.dot(AdjMat,arr[3])) + betaI*np.multiply(arr[0],np.dot(AdjMat,arr[2])) - thetaI*arr[1] - thetaA*arr[1]
  didt=thetaI*arr[1] - alphaI*arr[2] - gamma*arr[2]
  dadt=thetaA*arr[1] - alphaA*arr[3]
  drdt=alphaI*arr[2] + alphaA*arr[3]
  dddt=np.copy(gamma*arr[2])
  der_arr = np.array([dsdt,dldt,didt,dadt,drdt,dddt])
  return(der_arr)

def rk4(der_func, arr, AdMat):
  np.asarray(arr)
  k1 = np.array(der_func(arr,AdMat))*dt
  k2_arg = np.add(arr, k1/2)
  k2 = der_func(k2_arg,AdMat)*dt
  k3_arg = np.add(arr, k2/2)
  k3 = der_func(k3_arg,AdMat)*dt
  k4_arg = np.add(arr, k3)
  k4 = der_func(k4_arg,AdMat)*dt
  delta_arr_1 = np.add(k1, 2*k2)
  delta_arr_2 = np.add(2*k3, k4)
  delta_val_arr = np.add(delta_arr_1, delta_arr_2)*(1/6)
  arr_np1 = np.add(delta_val_arr, arr)
  return(arr_np1)


arr_n = [Sarr, Larr, Iarr, Aarr, Rarr, Darr]
array_0 = rk4(func_dot, arr_n, AM)
time_arr=[0]
#loop for evolution
NetChangeCounter=0
CounterLD=0
nFail=0
for i in range(N):
  array_np1 = rk4(func_dot, arr_n, AM)
  TotS.append(np.sum(array_np1[0]))
  TotL.append(np.sum(array_np1[1]))
  TotI.append(np.sum(array_np1[2]))
  TotA.append(np.sum(array_np1[3]))
  TotR.append(np.sum(array_np1[4]))
  TotD.append(np.sum(array_np1[5]))
  arr_n = array_np1
  time_arr.append(i/10.)
  if i%250==0:
    AM=construct_net(levs,ld=0)
    
  if i%(N/100)==0:
    print('step:',i)
    print(TotS[i],TotL[i],TotI[i],TotA[i],TotR[i],TotD[i])
    
  
#print(arr_n)

print(time.time()-tti)
plt.plot(time_arr, TotS, label='S')
plt.plot(time_arr, TotL, label='L')
plt.plot(time_arr, TotI, label='I')
plt.plot(time_arr, TotA, label='A')
plt.plot(time_arr, TotR, label='R')
plt.plot(time_arr, TotD, label='D')
plt.legend()
plt.ylabel('Number of idivduals')
plt.xlabel('Time')
plt.savefig('finPlots/noLD.png')
plt.show()
#plt.show(time_arr, S+I+L+A+R+D)
#plt.show()
print(TotS[-1]+TotL[-1]+TotI[-1]+TotA[-1]+TotR[-1]+TotD[-1])
print(TotD[-1]/TotR[-1])
