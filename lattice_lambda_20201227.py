import numpy as np  
import matplotlib.pyplot as plt
from math import *
import time
import sys

##A=np.array([[1,2],[3,2]])
##En,Vec=np.linalg.eig(A)
##B=np.array([[1+1j,2+1j],[3,4+5j]])
##print(En)
##print(Vec)
##print(Vec[:,[0]])
##print(Vec[[0]])
##print(Vec[0,0])
#print(np.conjugate(B))
#print(np.transpose(np.conjugate(B)))
#sys.exit(0)

start = time.time()
#@jit(nopython=True)
def intg(num,n,kx_min,kx_max,ky_min,ky_max):

    H = np.zeros((4,4),dtype=np.complex)
    v_yop = np.zeros((4,4),dtype=np.complex)
    M=np.zeros((4,4),dtype=np.complex)
    
    Kx = np.linspace(kx_min,kx_max,n+1)
    Ky = np.linspace(ky_min,ky_max,n+1)

    #y component of the velocity operator
    v_yop[0,0] = lambda_0
    v_yop[0,1] = 1j * v_y
    v_yop[0,2] = alpha_y
    v_yop[1,0] = -1j * v_y
    v_yop[1,1] = lambda_0
    v_yop[1,3] = alpha_y
    v_yop[2,0] = alpha_y
    v_yop[2,2] = -lambda_0
    v_yop[2,3] = 1j * v_y
    v_yop[3,1] = alpha_y
    v_yop[3,2] = -1j * v_y
    v_yop[3,3] = -lambda_0
    #the net division
    res_Sx=0
    res_Sy=0
    res_Sz=0
    for m1 in range(0,n):
        #Gaussian integral
        for i_x in range(0,num):
            k_x = ((1 - x_gauss[i_x]) * Kx[m1] + (1 + x_gauss[i_x]) * Kx[m1 + 1]) / 2
            for i_y in range(0,num):
                k_y = ((1 - x_gauss[i_y]) * Ky[m1] + (1 + x_gauss[i_y]) * Ky[m1+1]) / 2
                
                if i_lattice==1:
                    H=system_lattice(k_x,k_y)
                else:
                    H=system_continuous(k_x,k_y)
                Vc,Va=np.linalg.eig(H)
                
                #cumsum for four bands
                f_x=0
                f_y=0
                f_z=0
                for I in range(0,4):
                    G_R = 1/(E_F - Vc[I] + 1j * eta)
                    G_A =1/(E_F - Vc[I] - 1j * eta)
                    for ii in range(0,4):
                        for jj in range(0,4):
                            M[ii,jj]=Va[ii,I]*np.conjugate(Va[jj,I])
                    #f_z = f_z+np.trace(cos(k_y)*v_yop.dot(M).dot(s_z/2).dot(M))*G_R*G_A
                    #f_z = f_z+np.trace(v_yop.dot(M).dot(s_z/2).dot(M))*G_R*G_A
                    factor=np.sqrt(3.14159)/eta/eta*np.exp(-(E_F - Vc[I])**2/eta/eta)
                    if i_lattice==1:
                        factor=factor*cos(k_y)
                    f_x = f_x+np.trace(v_yop.dot(M).dot(s_x/2).dot(M))*factor
                    f_y = f_y+np.trace(v_yop.dot(M).dot(s_y/2).dot(M))*factor
                    f_z = f_z+np.trace(v_yop.dot(M).dot(s_z/2).dot(M))*factor
                res_Sx = res_Sx + f_x * w[i_x] * w[i_y]
                res_Sy = res_Sy + f_y * w[i_x] * w[i_y]
                res_Sz = res_Sz + f_z * w[i_x] * w[i_y]
        #S_z[i_cal] = S_z[i_cal] * (K[m1+1]-K[m1])**2/4
    res_Sx = res_Sx * (kx_max-kx_min)*(ky_max-ky_min)/(n**2*4)
    res_Sy = res_Sy * (kx_max-kx_min)*(ky_max-ky_min)/(n**2*4)
    res_Sz = res_Sz * (kx_max-kx_min)*(ky_max-ky_min)/(n**2*4)
    return res_Sx,res_Sy,res_Sz


def system_continuous(k_x,k_y):
    H = np.zeros((4,4),dtype=np.complex)
    H_0 = np.zeros((4,4),dtype=np.complex)
    H_Z = np.zeros((4,4),dtype=np.complex)
    H_R = np.zeros((4,4),dtype=np.complex)

    H_0[0,0] = m
    H_0[0,1] = v_x *k_x + 1j * v_y * k_y
    H_0[1,0] = v_x *k_x- 1j * v_y * k_y
    H_0[1,1] = -m
    H_0[2,2] = m
    H_0[2,3] = -H_0[1,0]
    H_0[3,2] = -H_0[0,1]
    H_0[3,3] = -m

    H_Z[0,0] = lambda_0 * k_y
    H_Z[0,1] = 1j * delta_z
    H_Z[1,0] = -H_Z[0,1]
    H_Z[1,1] = H_Z[0,0]
    H_Z[2,2] = -H_Z[0,0]
    H_Z[2,3] = H_Z[1,0]
    H_Z[3,2] = H_Z[0,1]
    H_Z[3,3] = -H_Z[0,0]

    H_R[0,2] = -1j * (alpha_x * k_x + 1j * alpha_y * k_y)
    H_R[0,3] = 1j * delta_x
    H_R[1,2] = H_R[0,3]
    H_R[1,3] = H_R[0,2]
    H_R[2,0] = 1j * (alpha_x * k_x - 1j * alpha_y * k_y)
    H_R[2,1] = -H_R[0,3]
    H_R[3,0] = -H_R[0,3]
    H_R[3,1] = H_R[2,0]

    H = H_0 + H_Z + H_R
    return H

def system_lattice(k_x,k_y):
    H = np.zeros((4,4),dtype=np.complex)
    H_0 = np.zeros((4,4),dtype=np.complex)
    H_Z = np.zeros((4,4),dtype=np.complex)
    H_R = np.zeros((4,4),dtype=np.complex)

    H_0[0,0] = m
    H_0[0,1] = v_x *sin( k_x) + 1j * v_y * sin(k_y)
    H_0[1,0] = v_x * sin( k_x)- 1j * v_y * sin(k_y)
    H_0[1,1] = -m
    H_0[2,2] = m
    H_0[2,3] = -H_0[1,0]
    H_0[3,2] = -H_0[0,1]
    H_0[3,3] = -m

    H_Z[0,0] = lambda_0 * sin(k_y)
    H_Z[0,1] = 1j * delta_z
    H_Z[1,0] = -H_Z[0,1]
    H_Z[1,1] = H_Z[0,0]
    H_Z[2,2] = -H_Z[0,0]
    H_Z[2,3] = H_Z[1,0]
    H_Z[3,2] = H_Z[0,1]
    H_Z[3,3] = -H_Z[0,0]

    H_R[0,2] = -1j * (alpha_x * sin( k_x) + 1j * alpha_y * sin(k_y))
    H_R[0,3] = 1j * delta_x
    H_R[1,2] = H_R[0,3]
    H_R[1,3] = H_R[0,2]
    H_R[2,0] = 1j * (alpha_x * sin( k_x) - 1j * alpha_y * sin(k_y))
    H_R[2,1] = -H_R[0,3]
    H_R[3,0] = -H_R[0,3]
    H_R[3,1] = H_R[2,0]

    H = H_0 + H_Z + H_R
    return H


# test the regime of the integration
def curve_function_test(ii,kx_min,kx_max,ky_min,ky_max):

    n=80
    k_x=np.linspace(kx_min,kx_max,n)
    k_y=np.linspace(ky_min,ky_max,n)
    kx, ky=np.meshgrid(k_x, k_y)
    fun=np.zeros((n,n),dtype=np.complex)

    H = np.zeros((4,4),dtype=np.complex)
    M=np.zeros((4,4),dtype=np.complex)

    #y component of the velocity operator
    v_yop = np.zeros((4,4),dtype=np.complex)
    v_yop[0,0] = lambda_0
    v_yop[0,1] = 1j * v_y
    v_yop[0,2] = alpha_y
    v_yop[1,0] = -1j * v_y
    v_yop[1,1] = lambda_0
    v_yop[1,3] = alpha_y
    v_yop[2,0] = alpha_y
    v_yop[2,2] = -lambda_0
    v_yop[2,3] = 1j * v_y
    v_yop[3,1] = alpha_y
    v_yop[3,2] = -1j * v_y
    v_yop[3,3] = -lambda_0

    for i_kx in range(0,n):
        kxx=kx[0,i_kx]
        for i_ky in range(0,n):
            kyy=ky[i_ky,0]
            if i_lattice==1:
                H=system_lattice(kxx,kyy)
            else:
                H=system_continuous(kxx,kyy)
            Vc,Va=np.linalg.eig(H)
            for I in range(0,4):
                G_R = 1/(E_F - Vc[I] + 1j * eta)
                G_A =1/(E_F - Vc[I] - 1j * eta)
                factor=np.sqrt(3.14159)/eta/eta*np.exp(-(E_F - Vc[I])**2/eta/eta)
                if i_lattice==1:
                    factor=factor*cos(k_y)
                for ii in range(0,4):
                    for jj in range(0,4):
                        M[ii,jj]=Va[ii,I]*np.conjugate(Va[jj,I])
                #fun[i_ky,i_kx] = fun[i_ky,i_kx]+np.trace(cos(kyy)*v_yop.dot(M).dot(s_z/2).dot(M))*G_R*G_A
                #fun[i_ky,i_kx] = fun[i_ky,i_kx]+np.trace(v_yop.dot(M).dot(s_z/2).dot(M))*G_R*G_A
                if ii==1:
                    fun[i_ky,i_kx] = fun[i_ky,i_kx]+np.trace(v_yop.dot(M).dot(s_x/2).dot(M))*factor
                elif ii==2:
                    fun[i_ky,i_kx] = fun[i_ky,i_kx]+np.trace(v_yop.dot(M).dot(s_y/2).dot(M))*factor
                elif ii==3:
                    fun[i_ky,i_kx] = fun[i_ky,i_kx]+np.trace(v_yop.dot(M).dot(s_z/2).dot(M))*factor
        
    plt.figure()
    plt.axes(projection='3d').plot_surface(kx,ky,fun.real,cmap='rainbow')
    #plt.contourf(kx,ky,fun,100,cmap='Spectral')
    #plt.colorbar()
    plt.xlabel('$k_x$ (1/nm)')
    plt.ylabel('$k_y$ (1/nm)')
    plt.show()
    exit()

    return


#spin operator
s_x = np.zeros((4,4),dtype=np.complex)
s_y = np.zeros((4,4),dtype=np.complex)
s_z = np.zeros((4,4),dtype=np.complex)
s_x[0,2] = 1
s_x[1,3] = 1
s_x[2,0] = 1
s_x[3,1] = 1
s_y[0,2] = -1j
s_y[1,3] = -1j
s_y[2,0] = 1j
s_y[3,1] = 1j
s_z[0,0] = 1
s_z[1,1] = 1
s_z[2,2] = -1
s_z[3,3] = -1

c_10 = 1.0 #unit eV
c_20 = 0 #unit eV
c_30 = -0.4 #unit eV
v_1x = 0.171 #eVnm
v_1y = 0.048 #eVnm
v_3x = 0.048 #eVnm
v_3y = -0.048 #eVnm
epsilon_1 = c_10
epsilon_2 = c_20
epsilon_3 = c_30
r = 0.0

epsilon_c = epsilon_1
epsilon_v = epsilon_2 / (1 + r) - r * epsilon_3 / (1 + r)
#m=(epsilon_c-epsilon_v)/2
m=0.0025

k_x0 = 0
k_y0 = 3.85 #nm**-1
r_0 = ((v_3x * k_x0) ** 2 + (v_3y * k_y0) ** 2) / (epsilon_3 ** 2)
v_x = np.sqrt(1 / (1 + r_0)) * v_1x #eVnm
v_y = np.sqrt(1 / (1 + r_0)) * v_1y #eVnm
#print(v_x,v_y)
#v_x=0.1
#v_y=0.1
eta = 0.005 #eV
E_F = 0.05 #eV

#coupling constants
#lambda_0=0.00001 #eVnm
alpha_x = 0.010 #eVnm
alpha_y = 0.06 #eVnm
delta_x = 0.01 #eV
delta_z = 0.0 #eV

i_lattice=0
kx_min=-0.5
kx_max=0.5
ky_min=-0.8
ky_max=0.8
ii=2
#curve_function_test(ii,-0.5,0.5,-0.8,0.8) #delta_z=0.0
#curve_function_test(ii,-0.5,0.5,-0.8,0.8) #delta_z=0.01
lambda_0=0.01
curve_function_test(ii,kx_min,kx_max,ky_min,ky_max)

num=5
n=100
x_gauss,w = np.polynomial.legendre.leggauss(num)
num_cal=10
lambda_array = np.linspace(0,0.01,num_cal)

S_x = np.zeros(num_cal,dtype=np.complex)
S_y = np.zeros(num_cal,dtype=np.complex)
S_z = np.zeros(num_cal,dtype=np.complex)

#S-lambda
for i_cal in range(0,num_cal): 
    lambda_0 = lambda_array[i_cal] #Fermi energy: eV
    S_x[i_cal],S_y[i_cal],S_z[i_cal]=intg(num,n,kx_min,kx_max,ky_min,ky_max)
    print("%d Sx=(%.2e %.2e) Sy=(%.2e %.2e) Sz=(%.2e %.2e)"%(i_cal+1,S_x[i_cal].real,S_x[i_cal].imag,S_y[i_cal].real,S_y[i_cal].imag,S_z[i_cal].real,S_z[i_cal].imag))
    
end=time.time()
print("calculation time:%.2fs" % (end - start))
#print(lambda_array)
#print(S_z)

plt.plot(lambda_array,S_x.real,'+',label='$S_x$',markerfacecolor='none')
plt.plot(lambda_array,S_y.real,'o',label='$S_y$',markerfacecolor='none')
plt.plot(lambda_array,S_z.real,'s',label='$S_z$',markerfacecolor='none')
plt.xlabel('$\\lambda$ (eVnm)')
plt.ylabel('$S$ (arb. unit)')
plt.legend(loc='best')
plt.show()

np.savetxt('data/spin_lambda.dat', np.transpose([lambda_array,S_x.real,S_y.real,S_z.real]), fmt="%.6e")

