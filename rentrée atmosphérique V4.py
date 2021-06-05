# -*- coding: utf-8 -*-
"""
Created on Sun May 30 16:42:47 2021

@author: Emilien
"""
import matplotlib.pyplot as plt
import numpy as np
from math import exp
from pylab import *



def graphe_masse_volumique_altitude():
    a=-6
    M=0.029
    T0=293
    p0=10**5
    g=9.81
    R=8.314
    z=np.linspace(0, 60000, 60000)
    masse_volumique= (M*p0)/( R*(T0-a*z))  *(1-a / T0 *z)**((M*g) / (R*a))
    
    masse_volumique2= (p0*M)/(R*T0) *exp (-(g *M)*z/ (R*T0))
    
    print (masse_volumique2)
    
    plt.plot(z,masse_volumique)  
    plt.plot(z,masse_volumique2,"r")
    
    
    plt.title ("masse volumique")
    plt.xlabel("altitude m") 
    plt.ylabel("masse volumique (kg/m^3)")
    
    plt.show
    
    
    
def transformation_degré_radian(alpha):
    return np.pi *alpha/ 180   


#constantes physiques
R_t= 6400000
M_t=6*10**24
G=6.7*10**(-11)

#précision du calcul
pas = 0.01

#constante pour masse volumique
a=-6
M=0.029
T0=293
p0=10**5
g=9.81
R=8.314

#constante navette
m= 7000
r=10
mode_débogue=0

def rentree_atmospherique( altitude_initiale, vitesse_ux_initiale, vitesse_uy_initiale, Cx, Cz,):
    print("démarage ")
    tab_tab_initialisation= initialisation(altitude_initiale,vitesse_ux_initiale, vitesse_uy_initiale)
    tab_tab,nombre_tour = résolution_équa_dif(tab_tab_initialisation,Cx,Cz)
    distance_parcourue_temps_t=tab_tab[0]
    tab_altitude_au_temps_t= tab_tab[1]
    tab_vitesse_temps_ux=tab_tab[2]
    tab_vitesse_uy=tab_tab[3]
    affichage_graphe= graphe(distance_parcourue_temps_t, tab_altitude_au_temps_t)
    """affichage_graphe_bis= graphe (tab_vitesse_uy,tab_altitude_au_temps_t)"""
    
 
def norme1(x,y):
    log("norme1")
    return np.sqrt(x**2+y**2)

def log(message):
    if mode_débogue==1:
        print(message)
        


def norme1_au_carré(x,y):
    log("norme1")
    return np.sqrt(x**2+y**2)

def cartesian(r,thêta):
    return ([r*np.cos(thêta, r*np.sin(thêta))])
    
def initialisation (altitude_initiale,vitesse_ux_initiale, vitesse_uy_initiale):
    log("initialisation")
    return [[0],[altitude_initiale],[vitesse_ux_initiale], [vitesse_uy_initiale]]

def résolution_équa_dif(tab_tab_initialisation,Cx,Cz):
    log("résolution_équa_dif")

    tab_tab=tab_tab_initialisation
    #print (tab_tab)
    
    while tab_tab[1][len(tab_tab[0])-1] > 0:
        i= len(tab_tab[0])-1
        distance_parcourue_t= calcul_distance_t(tab_tab[0][i],tab_tab[2][i])
        altitude_au_temps_t= calcul_altitude_t(tab_tab[1][i],tab_tab[3][i])
        bêta = calcul_angle_attaque(tab_tab,i)
        vitesse_ux_temps_t= calcul_vitesse_ux_t(tab_tab[1][i], tab_tab[2][i],tab_tab[3][i], bêta,Cx,Cz)
        vitesse_uy_temps_t= calul_vitesse_uy_t(tab_tab[1][i],tab_tab[2][i],tab_tab[3][i], bêta,Cx,Cz)                                                                 
        
        tab_tab= mis_dans_tab(tab_tab,distance_parcourue_t,altitude_au_temps_t,vitesse_ux_temps_t,vitesse_uy_temps_t)
        
    #print ( tab_tab)
    return tab_tab,i
                                                                                                                                  
def calcul_distance_t(x, vx):
    log("calcul_distance_t")
    return x + pas*vx

def calcul_altitude_t(y, vy):
    log("calcul_altitude_t")
    return y + pas*vy

def calcul_angle_attaque(tab_tab,i):
    log ("calcul_angle_attaque")
    return np.arctan (tab_tab[3][i]/tab_tab[2][i])


def calcul_vitesse_ux_t( z, vx, vy, bêta, Cx,Cz):
    log("calcul_vitesse_ux_t")
    return vx + pas *(1/(2*m) * masse_volumique(z) *np.pi* r **2 * norme1_au_carré ( vx, vy)* (Cz*np.sin (bêta)- Cx* np.cos(bêta)))


def calul_vitesse_uy_t(z,vx,vy,bêta,Cx,Cz):
    log("calul_vitesse_uy_t")
    return vy+ pas*(-g +1/(2*m) * masse_volumique(z) *np.pi* r **2 * norme1_au_carré ( vx, vy)* ( Cz* np.cos(bêta)+Cx*np.sin (bêta)))


def mis_dans_tab(tab_tab,x,y,vx,vy):
    log("mis_dans_tab")
    tab=[x,y,vx,vy]
    for i in range (len (tab_tab)):
        tab_tab[i].append(tab[i])
    return tab_tab



def masse_volumique(z):
    log("masse_volumique")
    #return (M*p0)/( R*(T0-a*z )) *(1-a / T0 *z)**((M*g) / (R*a))
    return  (p0*M)/(R*T0) *exp (-(g *M)*z/ (R*T0))
    

def graphe(distance_parcourue_temps_t, altitude_au_temps_t):
    log("graphe")
    print (i)
    #print (altitude_au_temps_t)
    altitude_au_temps_t= np.array(altitude_au_temps_t)
    plt.plot(distance_parcourue_temps_t,altitude_au_temps_t)    
    plt.title ("position de l'objet ")
    plt.xlabel("distance parcourue (m)") 
    plt.ylabel("hauteur de l'objet (m)")
    
    plt.show
    
rentree_atmospherique (2*10**5, 6333,-100,0.5,10)
    
