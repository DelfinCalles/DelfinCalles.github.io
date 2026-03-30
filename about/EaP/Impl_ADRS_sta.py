import math
import numpy as np
import matplotlib.pyplot as plt

#u,t = -V*u,x + k*u,xx - lamda*u + f

#Paramètres issues de la physique
K = 0.1 #Coeff de difussion
L = 1.0 #Taille du domaine
Time = 20. #Temps d'integration
V = 1
lamda = 1

plt.figure(1)
plt.figure(2)

#Paramètres pour l'analyse numérique
NX = 5 #Nombre de points du maillage
NT = 10000 #Nombre max de pas de temps
ifre = 1000000 #Frequence pour plotter
eps = 0.001 #Ratio de convergence relative
niter_refinement = 10 #Nombre de differents calculs pour changer la taille du maillage

errorL2 = np.zeros(niter_refinement)
errorH1 = np.zeros(niter_refinement)
tab_h = np.zeros(niter_refinement)

for iter in range(niter_refinement):
    NX = NX + 5 #On augmente le maillage de façon régulière, sans imposer aucune condition
    dx = L/(NX-1) #Pas en espace
    dt = dx**2/(V*dx+K+dx**2) #Pas de temps, condition stabilité CFL
    print(dx,dt)
    
    #Initialisation
    x = np.linspace(0.0, 1.0, NX) #Creation du maillage
    T = np.zeros(NX) #Solution approchée
    F = np.zeros(NX) #Fonction flux
    rest = []
    RHS = np.zeros(NX) #Comodité, c'est le Right Hand Side de notre équation
    
    Tex = np.zeros(NX) #Solution exacte
    Texx = np.zeros(NX) #Dérivée solution exacte
    
    for j in range(0,NX): #Déclaration de la solution exacte
        xx = j/NX
        Tex[j] = np.sin(2*math.pi*xx)*np.exp(-20*(xx-0.5)**2)
    for j in range(1,NX-1):
        Texx[j] = (Tex[j+1]-Tex[j-1])/(2*dx) #Première dérivée exacte
        Txx = (Tex[j+1]-2*Tex[j]+Tex[j-1])/(dx**2) #Seconde dérivée exacte
        F[j] = V*Texx[j] - K*Txx + lamda*Tex[j] #Fonction flux
    
    dt = dx**2/(V*dx+2*K+abs(np.max(F))*dx**2) #Pas de temps, condition de stabilité CFL
    
    plt.figure(1)
    
    #Main loop en temps
    n = 0
    res = 1
    res0 = 1
    while (n<NT and res/res0>eps): #Conditions boucle, pas trop de pas de temps et un erreur assez petit
        n += 1
        #Discretisation de la ADRS équation
        res = 0
        for j in range(1,NX-1):
            xnu = K + 0.5*dx*abs(V) #Viscosité numérique
            Tx = (T[j+1]-T[j-1])/(2*dx) #Solution approchée
            Txx = (T[j-1]-2*T[j]+T[j+1])/(dx**2) #Dérivée en espace solution approchée
            RHS[j] = dt*(-V*Tx + xnu*Txx - lamda*T[j] + F[j])
            res += abs(RHS[j])
        
        for j in range(1,NX-1):
            T[j] += RHS[j] #Calcul de la solution pour chaque pas de temps
            RHS[j] = 0
        
        
        T[NX-1] = T[NX-2] #Conditions aux limites, Neumann homogène à droite
        
        if (n==1): #?????
            res0 = res
        
        rest.append(res)
        
        #Plot selon ifre choisi
        if (n%ifre == 0 or (res/res0)<=eps):
            print(n,res)
            plotlabel = "t = %1.2f" %(n * dt)
            plt.plot(x,T, label=plotlabel,color = plt.get_cmap('copper')(float(n)/NT))
        
        print(n,res)
        plt.figure(1) #Comparaison solution approchée et exacte
        plt.plot(x,T)
        plt.plot(x,Tex)
        
        plt.figure(2)
        plt.plot(np.log10(rest/rest[0])) #Graphique de la convergence du reste
        
        errL2 = np.dot(T-Tex, T-Tex)/NX #Calcul produit vectoriel en norme L2
        
        errH1 = 0
        H1_ex = 0
        for j in range(1,NX-1): #Calcul "produit vectoriel" en seminorme H1
            Txx_ex = (Tex[j+1]-2*Tex[j]+Tex[j-1])/(dx**2)
            H1_ex += Txx_ex**2/NX
            errH1 += (Texx[j]-(T[j+1]-T[j-1])/(2*dx))**2/NX 
        
        tab_h[iter] = 1/NX
        errorL2[iter] = np.sqrt(errL2)/np.sqrt(H1_ex) #Erreur L2??
        errorH1[iter] = np.sqrt(errH1)/np.sqrt(H1_ex) #Erreur H1???
        
        print("norm error = ", errorL2[iter], errorH1[iter])

f = plt.figure(3)
f.add_subplot(1,2,1) #Plot des fonctions des erreurs
plt.plot(errorL2, label='L2')
plt.plot(errorH1, label='H1')
plt.legend()
f.add_subplot(1,2,2) #Plot de la convergence en temps
plt.plot(np.log10(rest/rest[0]),label='Convergence in time')
plt.legend()

##Façon numérique d'obtenir une valeur pour l'ordre de convergence de la méthode
plt.figure(4)
from scipy.optimize import curve_fit

#Modèle linéaire
def func(x,a,b):
    return a*x + b

def func_exp(x,a,b):
    return b*x**a

#Ici on passe a travailler sur un modèle linéaire
xdata = np.log10(tab_h) 
ydata = np.log10(errorL2)

popt, pcov = curve_fit(func, xdata, ydata) #Popt garde l'ordre de la méthode et une approx de la constante utilisée pour majorer l'erreur, cov c'est la covariance

model = func(xdata, popt[0], popt[1])

print('Code is order: ', popt[0], 'Constant C = ', 10**popt[1])
print('Covariance: ', pcov)

#Modèle exponentiel
xdata = tab_h
ydata = errorL2
popt, pcov = curve_fit(func_exp, xdata, ydata)

model_exp = func_exp(xdata, popt[0], popt[1])

plt.plot(ydata, label="L2")
plt.plot(10**model, label='Model linéaire')
plt.plot(model_exp, label='Model Exp.')
plt.legend()

print('Code is order: ', popt[0], 'Constant C = ', 10**popt[1])
print('Covariance: ', pcov)