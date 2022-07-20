import numpy as np
import matplotlib.pyplot as plt

#Define material properties
E = 3e7
A = 0.05

#Define node locations
node = []
node.append([0,0])
node.append([8,6])
node.append([12,0])

#numpy array for nodes
node = np.array(node).astype(float)

#Define Elements
element = []
element.append([0,1])
element.append([1,2])

#numpy array for elements
element = np.array(element)

#Define applied force
p = np.zeros_like(node) #set all values in array to zero
# p[node,axis] for axis 0 is x, 1 is y
p[1,0] = 50 #change this value in the array

#define support displacements set as many zeroes as DOF which need to be constrained
u=[0 ,0, 0, 0]

#Set degree of freedom for nodes 1 for free 0 for fixed
ADOF = np.ones_like(node).astype(int) #set all values in array to 1 (free)
ADOF[0,:] = 0
ADOF[2,:] = 0

def structural_solver():
    NodeNum = len(node) #number of nodes
    ElemNum = len(element) #number of elements
    DOF = 2 #number of degrees of freedom, 2 for 2d system, 3 for 3d system
    NDOF = DOF*NodeNum #total number of DOF in system
    
    #find relative coordinates to the previous node
    d = node[element[:,1],:] - node[element[:,0],:]
    #find legth of element
    l = np.sqrt((d**2).sum(axis=1))
    #find angle of element
    theta = d.T/l
    #create array for stiffness matrix
    a = np.concatenate((-theta.T, theta.T), axis=1)
    K = np.zeros([NDOF,NDOF])
    for k in range(ElemNum):
        aux = 2*element[k,:]
        index = np.r_[aux[0]:aux[0]+2,aux[1]:aux[1]+2]
        
        ES = np.dot(a[k][np.newaxis].T*E*A,a[k][np.newaxis])/l[k]
        K[np.ix_(index,index)] = K[np.ix_(index,index)] +ES
        
    freeDOF = ADOF.flatten().nonzero()[0]
    supportDOF = (ADOF.flatten() == 0).nonzero()[0]
    Kff = K[np.ix_(freeDOF, freeDOF)]
    Kfr = K[np.ix_(freeDOF, supportDOF)]
    Krf = Kfr.T
    Krr = K[np.ix_(supportDOF, supportDOF)]
    pf = p.flatten()[freeDOF]
    Uf = np.linalg.solve(Kff,pf)
    U = ADOF.astype(float).flatten()
    U[freeDOF] = Uf
    Ur =  U[supportDOF]
    U = U.reshape(NodeNum,DOF)
    u = np.concatenate((U[element[:,0]],U[element[:,1]]),axis=1)
    N = E*A/l[:]*(a[:]*u[:]).sum(axis=1)
    R = (Krf[:]*Uf).sum(axis=1) + (Krr[:]*Ur).sum(axis=1)
    R = R.reshape(2,DOF)
    return np.array(N), np.array(R), U
def plot(node,c,lt,lw,lg):
    for i in range(len(element)):
       xi , xf = node[element[i,0],0],node[element[i,1],0]
       yi , yf = node[element[i,0],1],node[element[i,1],1]
       line, = plt.plot([xi,xf],[yi,yf],color=c, linestyle=lt, linewidth=lw)
    line.set_label(lg)
    plt.legend(prop={'size': 8})
                                    
                                           
#Results
N, R, U = structural_solver()
print('Axial Forces')
print(N)
print('Reaction Forces')
print(R)
print('Deformation at Nodes')
print(U)

plot(node,'gray','--',1,'Undeformed')
scale = 1
Dnodes = U*scale + node
plot(Dnodes,'red','-',1,'Deformed')

plt.show()
