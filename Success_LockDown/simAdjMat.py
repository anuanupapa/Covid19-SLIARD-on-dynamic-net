import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from net import *
import networkx as nx
import numba as nb

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
ldnormal=1
levs=[3,4,5,4,2] # Keep this level structure, Other level structures can be explored but we dont have time; You need to have enough levels for LD
AM=construct_net(levs, ld=ldnormal)
S_ini = (3*4*5*2*4)-1#int(input('S initial : '))
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

#Don't change these values. 
betaI=.06 #.06
betaA=.1#.1
thetaI=.04 #0.04
thetaA=.08 #0.08
alphaI=.015 # 0.015
alphaA=.015 #0.015
gamma=.002 #0.002


TotS = [np.sum(Sarr)]
TotL = [np.sum(Larr)]
TotI = [0]
TotA = [0]
TotR = [0]
TotD = [0]
TotCases=[1]
DailyCases=[0]
TotDeath=[0]
DailyDeath=[0]

#defining the integral params
tstart=0
tend=1000.0
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
  #np.asarray(arr)
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


arr_n = np.array([Sarr, Larr, Iarr, Aarr, Rarr, Darr])
array_0 = rk4(func_dot, arr_n, AM)
time_arr=[0]

#Counters for network reconfigurations
InfThresh=10
LDCounterOnce=0
iterLD=0
#Start iteration
for i in range(N):
  prevI=arr_n[2]
  prevA=arr_n[3]
  prevD=arr_n[5]

  array_np1 = rk4(func_dot, arr_n, AM)
  TotS.append(np.sum(array_np1[0]))
  TotL.append(np.sum(array_np1[1]))
  TotI.append(np.sum(array_np1[2]))
  TotA.append(np.sum(array_np1[3]))
  TotR.append(np.sum(array_np1[4]))
  TotD.append(np.sum(array_np1[5]))

  nextD=array_np1[5]
  nextI=array_np1[2]
  nextA=array_np1[3]
  Idiff=np.add(nextI,np.multiply(-1,prevI))
  Adiff=np.add(nextA,np.multiply(-1,prevA))
  Ddiff=np.add(nextD,np.multiply(-1,prevD))
  AIincr=np.add(np.where(Idiff>0,Idiff,0),np.where(Adiff>0,Adiff,0))
  Dincr=np.where(Ddiff>0,Ddiff,0)
  TotCases.append(np.sum(AIincr)+TotCases[i])
  DailyCases.append(np.sum(AIincr))
  DailyDeath.append(np.sum(Dincr))
  TotDeath.append(np.sum(Dincr)+TotDeath[i])
  arr_n = array_np1
  time_arr.append(i/10.)

  if TotI[i]>=InfThresh and iterLD%5000==0:
    AM=construct_net(levs,100)
    LDCounterOnce=1
    print('LockDown !')
    #print(TotS[i],TotL[i],TotI[i],TotA[i],TotR[i],TotD[i])
  
  elif TotI[i]<InfThresh and i%250==0 and LDCounterOnce==0:
    AM=construct_net(levs,ld=ldnormal)
    iterLD=0
    print('Normal')
    #print(TotS[i],TotL[i],TotI[i],TotA[i],TotR[i],TotD[i])

  if LDCounterOnce==1:
    iterLD=iterLD+1
    
  #if (i%(N/200))==0: 
    #print('step:',i)
    #print(TotS[i],TotL[i],TotI[i],TotA[i],TotR[i],TotD[i])

  #if TotS[i]+TotR[i]+TotD[i]>n-10 and i>2000:
  #  break
  
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
plt.xlabel('Time(unit)')
#plt.savefig('fin/All'+'.png')
plt.show()

plt.plot(time_arr, TotI, 'g', label='Infected')
plt.xlabel('Time(unit)')
plt.ylabel('Number of indivduals')
plt.legend()
#plt.tick_params(labelleft=False)
#plt.savefig('fin/Infected.png')
plt.show()

plt.plot(time_arr, TotA, 'r', label='Assymptomatic')
plt.xlabel('Time(unit)')
plt.ylabel('Number of indivduals')
plt.legend()
#plt.tick_params(labelleft=False)
#plt.savefig('fin/Ass.png')
plt.show()

plt.plot(time_arr, TotD, 'k', label='Dead')
plt.xlabel('Time(unit)')
plt.ylabel('Number of indivduals')
plt.legend()
#plt.tick_params(labelleft=False)
#plt.savefig('fin/Dead.png')
plt.show()

plt.plot(time_arr, TotS, 'b', label='Susceptible')
plt.xlabel('Time(unit)')
plt.ylabel('Number of indivduals')
plt.legend()
#plt.tick_params(labelleft=False)
#plt.savefig('fin/susc.png')
plt.show()

plt.plot(time_arr, TotR, 'c', label='Recovered')
plt.xlabel('Time(unit)')
plt.ylabel('Number of indivduals')
plt.legend()
#plt.tick_params(labelleft=False)
#plt.savefig('fin/rec.png')
plt.show()

plt.plot(time_arr, np.add(TotI,TotA), 'r', label='Active Cases')
plt.xlabel('Time(unit)')
plt.ylabel('Number of indivduals')
plt.legend()
#plt.tick_params(labelleft=False)
#plt.savefig('fin/active.png')
plt.show()

plt.plot(time_arr, TotCases, 'b', label='Total Cases')
plt.xlabel('Time(unit)')
plt.ylabel('Number of indivduals')
plt.legend()
#plt.tick_params(labelleft=False)
#plt.savefig('fin/TotalCases.png')
plt.show()

plt.plot(time_arr, DailyCases, 'b', label='Daily Cases')
plt.xlabel('Time(unit)')
plt.ylabel('Number of indivduals')
plt.legend()
#plt.tick_params(labelleft=False)
#plt.savefig('fin/DailyCases.png')
plt.show()

plt.plot(time_arr, TotDeath, 'k', label='Total Deaths')
plt.xlabel('Time(unit)')
plt.ylabel('Number of indivduals')
plt.legend()
#plt.tick_params(labelleft=False)
#plt.savefig('fin/TotDeath.png')
plt.show()

plt.plot(time_arr, DailyDeath, 'k', label='Daily Deaths')
plt.xlabel('Time(unit)')
plt.ylabel('Number of indivduals')
plt.legend()
#plt.tick_params(labelleft=False)
#plt.savefig('fin/DailyDeath.png')
plt.show()


#plt.show(time_arr, S+I+L+A+R+D)
#plt.show()
print(TotS[-1]+TotL[-1]+TotI[-1]+TotA[-1]+TotR[-1]+TotD[-1])
print(TotD[-1]/TotR[-1])
