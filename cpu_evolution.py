#import libraries
import numbers # module for numerical types
import numpy as np # module for numerical arrays and calculations
import scipy # scientific computing library
import time # module for time-related functions
from scipy import sparse # module for sparse matrices
from scipy.sparse import csr_matrix, find # Compressed Sparse Row matrix format and find nonzero elements
from scipy.sparse.linalg import eigsh, eigs # module for eigenvalue and eigenvector calculations for sparse matrices
from scipy.sparse.linalg import lobpcg # module for Locally Optimal Block Preconditioned Conjugate Gradient Method
import sys # system-specific parameters and functions
import h5py # module for reading and writing HDF5 files
import os.path # module for common pathname manipulations
import os # module for interacting with the operating system
        
class spin_chain_evolution:
    def __init__(self, N):
        #initialize spin operators
        self.Sx = (1/np.sqrt(2)) * sparse.csr_matrix(np.array([[0, 1, 0],[1, 0, 1], [0,1,0]]))
        self.Sy = (1j/np.sqrt(2)) * sparse.csr_matrix(np.array([[0, -1, 0], [1, 0, -1], [0, 1, 0]]))
        self.Sz = sparse.csr_matrix(np.array([[1,0,0], [0, 0, 0], [0, 0, -1]]))
        self.Id = scipy.sparse.identity(3)
        self.N = N # number of spins in the chain
     # define tensor product for Sx operator at site m
    def tensorsx(self,m):
        List = [self.Id if n!=m else self.Sx for n in range(self.N)]
        T = List[0]
        for i in range(len(List)-1):
            T = sparse.kron(T,List[i+1],format = "csr")
        return T
    # define tensor product for Sy operator at site m
    def tensorsy(self,m):
        List = [self.Id if n!=m else self.Sy for n in range(self.N)]
        T = List[0]
        for i in range(len(List)-1):
            T = sparse.kron(T,List[i+1],format = "csr")
        return T
    # define tensor product for Sz operator at site m
    def tensorsz(self,m):
        List = [self.Id if n!=m else self.Sz for n in range(self.N)]
        T = List[0]
        for i in range(len(List)-1):
            T = sparse.kron(T,List[i+1],format = "csr")
        return T
     # define tensor product for Sz operators for all sites
    def tensorszprod(self):
        A = scipy.sparse.identity(3**self.N)
        for i in range(self.N):
            A = A @ self.tensorsz(i)
        return A
    # define tensor product for Sx operators for all sites
    def tensorsxprod(self):
        A = scipy.sparse.identity(3**self.N)
        for i in range(self.N):
            A = A @ self.tensorsx(i)
        return A
    # define tensor product for Sy operators for all sites
    def tensorsyprod(self):
        A = scipy.sparse.identity(3**self.N)
        for i in range(self.N):
            A = A @ self.tensorsy(i)
        return A
    #Define the model will be simulate
    def Bond_Hamiltonian(self, delta, Delta):
        H = 0 
        for n in range(self.N):
            if n<self.N-1:
                H += (1- delta*(-1)**n) * ((self.tensorsx(n) @ self.tensorsx(n+1))+ (self.tensorsy(n) @ self.tensorsy(n+1))+ Delta * (self.tensorsz(n) @ self.tensorsz(n+1)))
            else:#restrictions N+1=1
                H += (1- delta*(-1)**n) * ((self.tensorsx(n) @ self.tensorsx(0))+ (self.tensorsy(n) @ self.tensorsy(0))+ Delta * (self.tensorsz(n) @ self.tensorsz(0)))
        return H.real
    def Bilinear_Hamiltonian(self, theta):
        H = 0 
        for n in range(self.N):
            if n<self.N-1:
                H += np.cos(theta) * (self.tensorsx(n) @ self.tensorsx(n+1) + self.tensorsy(n) @ self.tensorsy(n+1) + self.tensorsz(n) @ self.tensorsz(n+1)) + np.sin(theta) * (self.tensorsx(n) @ self.tensorsx(n+1) + self.tensorsy(n) @ self.tensorsy(n+1) + self.tensorsz(n) @ self.tensorsz(n+1)) @ (self.tensorsx(n) @ self.tensorsx(n+1) + self.tensorsy(n) @ self.tensorsy(n+1) + self.tensorsz(n) @ self.tensorsz(n+1))
            else:#restrictions N+1=1
                H += np.cos(theta) * (self.tensorsx(n) @ self.tensorsx(0) + self.tensorsy(n) @ self.tensorsy(0) + self.tensorsz(n) @ self.tensorsz(0)) + np.sin(theta) * (self.tensorsx(n) @ self.tensorsx(0) + self.tensorsy(n) @ self.tensorsy(0) + self.tensorsz(n) @ self.tensorsz(0)) @ (self.tensorsx(n) @ self.tensorsx(0) + self.tensorsy(n) @ self.tensorsy(0) + self.tensorsz(n) @ self.tensorsz(0))
        return H.real
    def XXZ_Hamiltonian(self,Jz,D):
        H = 0
        for n in range(self.N):
            if n<self.N-1:
                H += self.tensorsx(n) @ self.tensorsx(n+1) + self.tensorsy(n) @ self.tensorsy(n+1) + Jz * (self.tensorsz(n) @ self.tensorsz(n+1)) + D * (self.tensorsz(n) @ self.tensorsz(n))
            else:#restrictions N+1=1
                H += self.tensorsx(n) @ self.tensorsx(0) + self.tensorsy(n) @ self.tensorsy(0) + Jz * (self.tensorsz(n) @ self.tensorsz(0)) + D * (self.tensorsz(n) @ self.tensorsz(n))
        return H.real
    #get the eigenvalues and eigenvectors to calculate the correlations
    def eingenvalues_eingevector(self,H):
        #eigsxSGS, eivsxSGS = lobpcg(H,np.random.rand(H.shape[0],30), largest = False)
        eigsxSGS, eivsxSGS = eigsh(H, k =20, which = 'SA')
        #eigsxSGS, eivsxSGS = np.linalg.eigh(H.todense())
        #eigsxSGS, eivsxSGS = np.linalg.eig(H.todense())
        avasxSGS = np.round(eigsxSGS,13)# toma até a oitava casa
        minEVsxSGS = min(avasxSGS)  #pega os autovalores menores
        pesmin = np.where(minEVsxSGS == avasxSGS)[0]
        return pesmin, sparse.csr_matrix(eivsxSGS)
    #Get the two bodies correlation
    def Valor_Esperado(self, graundseig,ketstate, A):
        vm = 0
        for pe in graundseig:
            p1vm = A @ ketstate[:,pe].reshape(-1,1)
            p2vm = ((ketstate[:,pe]).reshape(1,-1))@ p1vm
            vm += (p2vm.toarray())[0][0]
        return (vm/graundseig.shape[0]).real   
    # Modeling the evolution of the system
    def Evolution(self,G, H):
        j = G[0]
        print(G)
        if os.path.isfile('escrita_provisoria'+str(j)+'.hdf5'):
            print("arquivo já existe")
        else:
            VE = []
            if H=='XXZ':
                A = G[1][0]
                B = G[1][1]
                eigedata = self.eingenvalues_eingevector(self.XXZ_Hamiltonian(A,B))
                VE.append(A)
                VE.append(B)
            elif H=='Bond':
                A = G[1][0]
                B = G[1][1]
                eigedata = self.eingenvalues_eingevector(self.Bond_Hamiltonian(A,B))
                VE.append(A)
                VE.append(B)
            elif H=='Bilinear':
                A = G[1]
                eigedata = self.eingenvalues_eingevector(self.Bilinear_Hamiltonian(A))
                VE.append(A/np.pi)
            graundseig = eigedata[0]
            ketstate = eigedata[1]
            VEsx = [self.Valor_Esperado(graundseig, ketstate, self.tensorsx(0) @ self.tensorsx(n))
                   for n in range(1,int(self.N/2+2))]
            VEsy = [self.Valor_Esperado(graundseig, ketstate, self.tensorsy(0) @ self.tensorsy(n))
                   for n in range(1,int(self.N/2+2))]
            VEsz = [self.Valor_Esperado(graundseig, ketstate, self.tensorsz(0) @ self.tensorsz(n))
                   for n in range(1,int(self.N/2+2))]
            CTCx = self.Valor_Esperado(graundseig, ketstate , self.tensorsxprod())
            CTCy = self.Valor_Esperado(graundseig, ketstate , self.tensorsyprod())
            CTCz = self.Valor_Esperado(graundseig, ketstate , self.tensorszprod())
            VE.extend(VEsx)
            VE.extend(VEsy)
            VE.extend(VEsz)
            VE.append(CTCx)
            VE.append(CTCy)
            VE.append(CTCz)
            f = h5py.File('escrita_provisoria'+str(j)+'.hdf5','w')
            f.create_dataset("correlations_"+ str(j),data = np.array(VE), dtype ='f',compression="gzip")
            f.close()
            
    # Write the observables
    def write(self, L, H):
        if H=='Bilinear':
            featu = 3*(self.N/2+1)+ 4 #numero de correlações
        else:
            featu = 3*(self.N/2+1)+ 5 #numero de correlações
        g = h5py.File(str(self.N)+'spinsCF_'+H+'.hdf5', mode = 'w')
        corr = g.create_dataset('correlations',(len(L),featu),chunks=True, dtype ='f',compression="gzip")
        for i in L:
            j = i[0]
            f = h5py.File('escrita_provisoria'+str(j)+'.hdf5', mode = 'r')
            C = f.get("correlations_"+ str(j))
            corr[j,:] = C[:]
            f.close()
        g.close()
        for i in L:
            j = i[0]
            os.remove("escrita_provisoria"+str(j)+".hdf5")
#input the dimensionality of the system and the model
spin_number = int(input("Spin Number: "))  
model = input("Model: ")
if model == 'XXZ':
    Jz_D_list = [(x, y) for x in np.arange(-4, 4, 0.1) for y in np.arange(-4, 4, 0.1) ]
    data_list = list(enumerate(Jz_D_list))
elif model == "Bond":
    delta_D_list = [(x, y) for x in np.arange(0, 1, 0.0125) for y in np.arange(-1.5, 2.5, 0.05) ]
    data_list = list(enumerate(delta_D_list))
else:
    theta_list = [ x for x in np.arange(0,2*np.pi,0.00314159) ]
    data_list = list(enumerate(theta_list))
    
initial_time = time.time()

SCE = spin_chain_evolution(spin_number)


for G in data_list:
    print(G)
    t = time.time()
    j = G[0]
    theta = G[1]
    SCE.Evolution(G, model)
    print(time.time()-t)
    print('#########')
calculation_time = time.time()-initial_time
print(calculation_time)
SCE.write(data_list, model)
