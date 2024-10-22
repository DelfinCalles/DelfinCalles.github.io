import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def plot_info_fin(X, Y, u, minnn, maxxx):

    minu = np.min(u)
    maxu = np.max(u)
    if maxu - minu <= 1.e-6:
        minu -= 1
        maxu += 1
        
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf=ax.plot_surface(X, Y, u.T, cmap='jet')
    ax.view_init(azim=-150,elev=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature')
    ax.set_title(r"$\mathbf{Configuration\;\; finale}$", fontsize=14)
    surf.set_clim(minnn, maxxx)
    ax.set_zlim(minnn, maxxx)
    plt.show()

    
    moy = np.mean(u[:, :])
    std_dev = np.std(u[:, :])

    label1="\033[1mLa température minimum est de {:.2f}°C".format(minu)
    label2="\033[1mLa température maximum est de {:.2f}°C".format(maxu)
    label3="\033[1mLa température moyenne est de {:.2f}°C".format(moy)
    label4="\033[1mL'écart type des températures est de {:.2f}°C".format(std_dev)

    max_length = max(len(label1), len(label2), len(label3), len(label4))
    label1 = label1.ljust(max_length)
    label2 = label2.ljust(max_length)
    label3 = label3.ljust(max_length)
    label4 = label4.ljust(max_length)

    print(label1.center(max_length * 2))
    print(label2.center(max_length * 2))
    print(label3.center(max_length * 2))
    print(label4.center(max_length * 2))
    print(" ")


def plot_info_init(X, Y, u, minnn, maxxx):

    minu = np.min(u)
    maxu = np.max(u)
    if maxu - minu <= 1.e-6:
        minu -= 1
        maxu += 1
        
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf=ax.plot_surface(X, Y, u.T, cmap='jet')
    ax.view_init(azim=-150,elev=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature')
    ax.set_title(r"$\mathbf{Configuration\;\; initiale}$", fontsize=14)
    surf.set_clim(minnn, maxxx)
    ax.set_zlim(minnn, maxxx)
    plt.show()