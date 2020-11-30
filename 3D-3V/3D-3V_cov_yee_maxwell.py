#
# author:         L. Pezzini
# e-mail :        luca.pezzini@edu.unito.it
# date:           24.11.2020
# copyright:      2020 KU Leuven (c)
# MIT license
#

#
# Maxwell Solver 3D-3V
#       • General coordinates
#       • Energy conserving
#       • Mimetic operator
#

import numpy as np
import matplotlib.pyplot as plt
import time
import math
import sys

'''
Choose the tipe of visualization:
flag_plt_et  -> each time step 
flag_plt_end -> final time step
'''
#flag_plt = True
flag_data    = False
flag_plt_et  = True
flag_plt_end = False

# Time steps
Nt = 101
dt = 0.001
# Nodes
nx1 = 76
nx2 = 76
nx3 = 3 # two layers are used for the BC

# Computational domain 
x1min, x1max = 0, 1    
x2min, x2max = 0, 1    
x3min, x3max = 0, .25  
Lx1 = int(abs(x1max - x1min))
Lx2 = int(abs(x2max - x2min)) 
Lx3 = abs(x2max - x2min)
dx1 = Lx1/(nx1 - 1)
dx2 = Lx2/(nx2 - 1)
dx3 = Lx3/(nx3 - 1)
dx = dx1
dy = dx2
dz = dx3

# Geometry: Tensor and Jacobian
''''
 To avoid an 8-point stencil for the metrics 
 is introduced one more index to define the metrics location:
 - s = 0: gij is colocated with the E field
 - s = 1: gij is colocated with the B field
'''
J = np.ones([nx1, nx2, nx3], dtype=float) 
g11 = np.ones([nx1, nx2, nx3, 2], dtype=float)
g12 = np.zeros([nx1, nx2, nx3, 2], dtype=float)
g13 = np.zeros([nx1, nx2, nx3, 2], dtype=float)
g21 = np.zeros([nx1, nx2, nx3, 2], dtype=float)
g22 = np.ones([nx1, nx2, nx3, 2], dtype=float)
g23 = np.zeros([nx1, nx2, nx3, 2], dtype=float)
g31 = np.zeros([nx1, nx2, nx3, 2], dtype=float)
g32 = np.zeros([nx1, nx2, nx3, 2], dtype=float)
g33 = np.ones([nx1, nx2, nx3, 2], dtype=float)
# Grid matrix
x1 = np.linspace(x1min, Lx1 - dx1, nx1, dtype=float)
x2 = np.linspace(x2min, Lx2 - dx2, nx2, dtype=float)
x3 = np.linspace(x3min, Lx3 - dx3, nx3, dtype=float)
x1v, x2v, x3v = np.meshgrid(x1, x2, x3, indexing='ij')
x1g, x2g = np.mgrid[0:Lx1:(nx1*1j), 0:Lx2:(nx2*1j)]

# Field matrix
Ex1 = np.zeros([nx1, nx2, nx3], dtype=float)
Ex2 = np.zeros([nx1, nx2, nx3], dtype=float)
Ex3 = np.zeros([nx1, nx2, nx3], dtype=float)
Bx1 = np.zeros([nx1, nx2, nx3], dtype=float)
Bx2 = np.zeros([nx1, nx2, nx3], dtype=float)
Bx3 = np.zeros([nx1, nx2, nx3], dtype=float) 
# Field matrix (old)
Ex1_old = np.zeros([nx1, nx2, nx3], dtype=float)
Ex2_old = np.zeros([nx1, nx2, nx3], dtype=float)
Ex3_old = np.zeros([nx1, nx2, nx3], dtype=float)
Bx1_old = np.zeros([nx1, nx2, nx3], dtype=float)
Bx2_old = np.zeros([nx1, nx2, nx3], dtype=float)
Bx3_old = np.zeros([nx1, nx2, nx3], dtype=float)
# Curl of fields
curl_Ex1 = np.zeros([nx1, nx2, nx3], dtype=float)
curl_Ex2 = np.zeros([nx1, nx2, nx3], dtype=float)
curl_Ex3 = np.zeros([nx1, nx2, nx3], dtype=float)
curl_Bx1 = np.zeros([nx1, nx2, nx3], dtype=float)
curl_Bx2 = np.zeros([nx1, nx2, nx3], dtype=float)
curl_Bx3 = np.zeros([nx1, nx2, nx3], dtype=float)

#Perturbation
Bx3[int((nx1-1)/2), int((nx2-1)/2), :] = 1.
# Total energy
U = np.zeros(Nt, dtype=float) 
# Divergence of E
divE = np.zeros(Nt, dtype=float) 

# Grid begin & end point
# Note we don't start from 0 cause 0 and nx1-1 are the same node
ib = 1
ie = nx1 - 1
jb = 1
je = nx2 - 1
kb = 1
ke = nx3 - 1

def myplot(values, name):
    '''
    To plot the Map of a vector fied over a grid.
    '''
    plt.figure(name)
    plt.imshow(values.T, origin='lower', extent=[0, Lx1, 0, Lx2], aspect='equal', vmin=-0.01, vmax=0.01)#,cmap='plasma')
    plt.colorbar()

def myplot2(values, name):
    '''
    To plot the behavior of a scalar fied in time.
    '''
    plt.figure(name)
    plt.plot(values)

'''
#------------------------- Cartesian -------------------------#
def shift(mat, x, y, z):
    # Matrix xyz-position shifter with periodic boundary conditions.
    result = np.roll(mat, -x, 0)
    result = np.roll(result, -y, 1)
    result = np.roll(result, -z, 2)
    return result


def ddx(A, s):
    # Derivative in x.
    return (shift(A, 1-s, 0, 0) - shift(A, 0-s, 0, 0))/dx1


def ddy(A, s):
    # Derivative in y.
    return (shift(A, 0, 1-s, 0) - shift(A, 0, 0-s, 0))/dx2


def ddz(A, s):
    # Derivative in z.
    return (shift(A, 0, 0, 1-s) - shift(A, 0, 0, 0-s))/dx3


def curl(Ax, Ay, Az, s):
    # Curl operator.
    curl_x = ddy(Az, s) - ddz(Ay, s)
    curl_y = ddz(Ax, s) - ddx(Az, s)
    curl_z = ddx(Ay, s) - ddy(Ax, s)
    return curl_x, curl_y, curl_z
#------------------------- Cartesian -------------------------#
'''

def avg1(A):
    '''
    To compute the average over i index.
    '''
    res = np.zeros_like(A)
    #print(res[ib:ie, jb:je, kb:ke].shape)
    #print(A.shape)
    #print(A[ib+1:ie+1, jb:je, kb:ke].shape)
    #print(A[ib:ie, jb:je, kb:ke].shape)
    #print(ib, ie, jb, je, kb, ke)
    res[ib:ie, jb:je, kb:ke] = (A[ib+1:ie+1, jb:je, kb:ke] + A[ib:ie, jb:je, kb:ke])/2
    return res

def avg2(A):
    '''
    To compute the average over j index.
    '''
    res = np.zeros_like(A)
    res[ib:ie, jb:je, kb:ke] = (A[ib:ie, jb+1:je+1, kb:ke] + A[ib:ie, jb:je, kb:ke])/2
    return res

def avg3(A):
    '''
    To compute the average over k index.
    '''
    res = np.zeros_like(A)
    res[ib:ie, jb:je, kb:ke] = (A[ib:ie, jb:je, kb+1:ke+1] + A[ib:ie, jb:je, kb:ke])/2
    return res

def derx1(A, s):
    '''
    To compute the derivative along the direction x1.
    s = 0 -> Forward derivative
    s = 1 -> Beckward derivative
    '''
    res = np.zeros_like(A)
    res[ib:ie, jb:je, kb:ke] = (A[ib+1-s:ie+1-s, jb:je, kb:ke] - A[ib-s:ie-s, jb:je, kb:ke])/dx1
    return res

def derx2(A, s):
    '''
    To compute the derivative along the direction x2.
    s = 0 -> Forward derivative
    s = 1 -> Beckward derivative
    '''
    res = np.zeros_like(A)
    res[ib:ie, jb:je, kb:ke] = (A[ib:ie, jb+1-s:je+1-s, kb:ke] - A[ib:ie, jb-s:je-s, kb:ke])/dx2
    return res

def derx3(A, s):
    '''
    To compute the derivative along the direction x3.
    s = 0 -> Forward derivative
    s = 1 -> Beckward derivative
    '''
    res = np.zeros_like(A)
    res[ib:ie, jb:je, kb:ke] = (A[ib:ie, jb:je, kb+1-s:ke+1-s] - A[ib:ie, jb:je, kb-s:ke-s])/dx3
    return res

def curl(A1, A2, A3, s):
    '''
    To compute the Curl in covariant coordinate.
    '''
    curlx1 = np.zeros([nx1, nx2, nx3], dtype=float)
    curlx2 = np.zeros([nx1, nx2, nx3], dtype=float)
    curlx3 = np.zeros([nx1, nx2, nx3], dtype=float)
    
    curlx1 = ((derx2(avg1(avg3(g31[:, :, :, s] * A1)), s) + derx2(avg2(avg3(g32[:, :, :, s] * A2)), s) + derx2(g33[:, :, :, s] * A3, s))\
           -  (derx3(avg1(avg2(g21[:, :, :, s] * A1)), s) + derx3(g22[:, :, :, s] * A2, s) + derx3(avg3(avg2(g23[:, :, :, s] * A3)), s)))/J
    curlx2 = ((derx3(g11[:, :, :, s] * A1, s) + derx3(avg2(avg1(g12[:, :, :, s] * A2)), s) + derx3(avg3(avg1(g13[:, :, :, s] * A3)), s))\
           -  (derx1(avg1(avg3(g31[:, :, :, s] * A1)), s) + derx1(avg2(avg3(g32[:, :, :, s] * A2)), s) + derx1(g33[:, :, :, s] * A3, s)))/J
    curlx3 = ((derx1(avg1(avg2(g21[:, :, :, s] * A1)), s) + derx1(g22[:, :, :, s] * A2, s) + derx1(avg3(avg2(g23[:, :, :, s] * A3)), s))\
           -  (derx2(g11[:, :, :, s] * A1, s) + derx2(avg2(avg1(g12[:, :, :, s] * A2)), s) + derx2(avg3(avg1(g13[:, :, :, s] * A3)), s)))/J
    return curlx1, curlx2, curlx3

def dx1dx(A, s):
    '''
    To compute the change of coordinate x1 - x.
    '''
    res = np.zeros_like(A)
    res[ib:ie, jb:je, kb:ke] = (A[ib+1-s:ie+1-s, jb:je, kb:ke] - A[ib-s:ie-s, jb:je, kb:ke])/dx
    return res

def dx2dy(A, s):
    '''
    To compute the change of coordinate x2 - y.
    '''
    res = np.zeros_like(A)
    res[ib:ie, jb:je, kb:ke] = (A[ib:ie, jb+1-s:je+1-s, kb:ke] - A[ib:ie, jb-s:je-s, kb:ke])/dy
    return res

def dx3dz(A, s):
    '''
    To compute the change of coordinate x3 - z.
    '''
    res = np.zeros_like(A)
    res[ib:ie, jb:je, kb:ke] = (A[ib:ie, jb:je, kb+1-s:ke+1-s] - A[ib:ie, jb:je, kb-s:ke-s])/dz
    return res

def div(A1, A2, A3):
    ''' 
    To compute the Divergence in covariant coordinate.
    ''' 
    res = np.zeros([nx1, nx2, nx3], dtype=float)
    res = (derx1(dx1dx(x1v[:, :, :], 0)*J[:, :, :]*A1[:, :, :],0)\
        +  derx2(dx2dy(x2v[:, :, :], 0)*J[:, :, :]*A2[:, :, :],0)\
        +  derx3(dx3dz(x3v[:, :, :], 0)*J[:, :, :]*A3[:, :, :],0))/J
    return res

def periodicBC(A):
    '''
    To impose periodic BC. The border of the grid domain is used 
    to impose periodicity in each direction, while the center of the box 
    contains the sensitive informations.
    '''
    A_w = np.zeros([nx1, nx2, nx3], dtype=float)
    A_s = np.zeros([nx1, nx2, nx3], dtype=float)
    A_b = np.zeros([nx1, nx2, nx3], dtype=float)
    # swop var.
    A_w[0, :, :] = A[0, :, :]
    A_s[:, 0, :] = A[:, 0, :]
    A_b[:, :, 0] = A[:, :, 0]
    # Reflective BC for A field
    A[0, :, :]  = A[-1, :, :]   # west = est
    A[-1, :, :] = A_w[0, :, :]  # est = west
    A[:, 0, :]  = A[:, -1, :]   # south = north
    A[:, -1, :] = A_s[:, 0, :]  # north = south
    A[:, :, 0]  = A[:, :, -1]   # bottom = top
    A[:, :, -1] = A_b[:, :, 0]  # top = bottom  

for t in range(Nt):
    start = time.time()
    plt.clf()
    print('')
    print('TIME STEP:', t)

    Bx1_old[:, :, :] = Bx1[:, :, :]
    Bx2_old[:, :, :] = Bx2[:, :, :]
    Bx3_old[:, :, :] = Bx3[:, :, :] 

    curl_Bx1, curl_Bx2, curl_Bx3 = curl(Bx1, Bx2, Bx3, 0)

    Ex1 += dt*curl_Bx1
    Ex2 += dt*curl_Bx2
    Ex3 += dt*curl_Bx3

    periodicBC(Ex1)
    periodicBC(Ex2)
    periodicBC(Ex3)

    curl_Ex1, curl_Ex2, curl_Ex3 = curl(Ex1, Ex2, Ex3, 1)

    Bx1 -= dt*curl_Ex1
    Bx2 -= dt*curl_Ex2
    Bx3 -= dt*curl_Ex3

    periodicBC(Bx1)
    periodicBC(Bx2)
    periodicBC(Bx3)

    U[t] = 0.5 * np.sum(Ex1[ib:ie, jb:je, kb:ke]**2 + Ex2[ib:ie, jb:je, kb:ke]**2 + Ex3[ib:ie, jb:je, kb:ke]**2\
                      + Bx1[ib:ie, jb:je, kb:ke]*Bx1_old[ib:ie, jb:je, kb:ke] + Bx2[ib:ie, jb:je, kb:ke]*Bx2_old[ib:ie, jb:je, kb:ke]\
                      + Bx3[ib:ie, jb:je, kb:ke]*Bx3_old[ib:ie, jb:je, kb:ke])


    divE[t] = np.sum(div(Ex1, Ex2, Ex3))

    print('E field         :', np.sum(Ex1), np.sum(Ex2), np.sum(Ex3))
    print('B field         :', np.sum(Bx1), np.sum(Bx2), np.sum(Bx3))
    print('Field Energy    :', U[t])
    print('div(E)          :', divE[t])

    if flag_data == True:
        f = open("output_cov_maxwell_yee3D.dat", "a")
        #print(t, np.sum(Ex1), np.sum(Ex2), np.sum(Ex3), np.sum(Bx1), np.sum(Bx2), np.sum(Bx3), U[t], divE[t], file=f)
        print(t, np.sum(Ex1), np.sum(Ex2), np.sum(Ex3), np.sum(Bx1), np.sum(Bx2), np.sum(Bx3), file=f)
        f.close()

    stop = time.time()
    print('TIME LEFT [min] :', (stop - start)*(Nt - 1 - t)/60.)

    if flag_plt_et == True:
        #plt.figure(figsize =(10, 8))
        '''
        plt.subplot(2, 2, 1)
        plt.pcolor(x1v, x2v, Bx3[:,:,1])
        plt.title('B_z Field Map')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.pcolor(x1v, x2v, Ex1[:,:,1])
        plt.title('B_z Field Map')
        plt.xlabel('x')
        plt.ylabel('y')
        '''
        plt.subplot(2, 2, 1)
        plt.pcolor(x1g, x2g, Bx3[:, :, 1])
        plt.title('B_z Map')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.pcolor(x1g, x2g, Ex1[:, :, 1])
        plt.title('E_x Map')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.subplot(2, 2, 3)
        plt.plot(divE)
        plt.title('Divergence Free')
        plt.xlabel('time')
        plt.ylabel('divE[t]')
        plt.subplot(2, 2, 4)
        plt.plot(U)
        plt.title('Total Energy')
        plt.xlabel('time')
        plt.ylabel('U')

        plt.pause(0.0001)
        plt.clf()

if flag_plt_end == True:
    myplot(Ex1[:, :, 1], 'Ex1')
    myplot(Ex2[:, :, 1], 'Ex2')
    myplot(Bx3[:, :, 1], 'Bx3')
    myplot2(divE, 'divE vs t')
    myplot2(U, 'U_field vs t')
    plt.show()
