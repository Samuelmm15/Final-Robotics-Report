import colorsys as cs
import matplotlib.pyplot as plt
import sys
from math import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

# ******************************************************************************
# Declaración de funciones

def muestra_origenes(O, final=0):
    # Muestra los orígenes de coordenadas para cada articulación
    print('Origenes de coordenadas:')
    for i in range(len(O)):
        print('(O'+str(i)+')0\t= '+str([round(j, 3) for j in O[i]]))
    if final:
        print('E.Final = '+str([round(j, 3) for j in final]))


def muestra_robot(O, obj):
    # Muestra el robot graficamente
    plt.figure(1)
    plt.xlim(-L, L)
    plt.ylim(-L, L)
    T = [np.array(o).T.tolist() for o in O]
    for i in range(len(T)):
        plt.plot(T[i][0], T[i][1], '-o',
                 color=cs.hsv_to_rgb(i/float(len(T)), 1, 1))
    plt.plot(obj[0], obj[1], '*')
    plt.show()
    input("Continuar...")
    plt.close()

def matriz_T(d, th, a, al):
    # Calcula la matriz T (ángulos de entrada en grados)

    return [[cos(th), -sin(th)*cos(al),  sin(th)*sin(al), a*cos(th)], [sin(th),  cos(th)*cos(al), -sin(al)*cos(th), a*sin(th)], [0,          sin(al),          cos(al),         d], [0,                0,                0,         1]
            ]

def cin_dir(th, a):
    # Sea 'th' el vector de thetas
    # Sea 'a'  el vector de longitudes
    T = np.identity(4)
    o = [[0, 0]]
    for i in range(len(th)):
        T = np.dot(T, matriz_T(0, th[i], a[i], 0))
        tmp = np.dot(T, [0, 0, 0, 1])
        o.append([tmp[0], tmp[1]])
    return o

def read_input_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    ## Comprobación de errores en la estructura del fichero introducido
    if len(lines) != 5:
        sys.exit("El fichero de entrada debe de tener 5 líneas.")
    type_arm = [int(i) for i in lines[0].split()]
    th = [float(i) for i in lines[1].split()]
    a = [float(i) for i in lines[2].split()]
    upper_limit = [float(i) for i in lines[3].split()]
    lower_limit = [float(i) for i in lines[4].split()]
    return type_arm, th, a, upper_limit, lower_limit

# ******************************************************************************
# Cálculo de la cinemática inversa de forma iterativa por el método CCD
plt.ion()  # modo interactivo

# introducción del punto para la cinemática inversa
if sys.argv[1] == "-h" or sys.argv[1] == "--help":
    print("<< Bienvenido al programa de ayuda de cinemática inversa >>")
    print("Para ejecutar el programa, debe de seguir la siguiente estructura de ejecución:")
    print("python ccdp3.py << FileName >> << x >> << y >>")
    print("Donde FileName es el nombre del fichero de entrada")
    print("y x e y son las coordenadas del punto al que se quiere llegar")
    print("El fichero de entrada debe de tener la siguiente estructura (CADA VALOR SEPARADO POR ESPACIOS):")
    print("<< Tipo de articulación >> (0 para rotacional, 1 para prismática)")
    print("<< Theta >>")
    print("<< a, Longitud >>")
    print("<< Límite superior >>")
    print("<< Límite inferior >>")
    print("Ejemplo:")
    print("=====================================================")
    print("0 0 1 1")
    print("10. 20. 0. 0.")
    print("1. 1. 1. 1.")
    print("90 90 0 0")
    print("-90 -90 0 0")
    print("=====================================================")
    sys.exit()

if len(sys.argv) != 4:
    sys.exit("python " + sys.argv[0] + " FileName" + " x y")
objetivo = [float(i) for i in sys.argv[2:]]

# Lectura del fichero de entrada
file_name = sys.argv[1]
type_arm, th, a, Upper_limit, Lower_limit = read_input_file(file_name)
L = sum(a)
EPSILON = .01 # umbral de convergencia

O = list(range(len(th)+1))  # Reservamos estructura en memoria
O[0] = cin_dir(th, a)  # Calculamos la posicion inicial
print("- Posicion inicial:")
muestra_origenes(O[0])

dist = float("inf")
prev = 0.
iteracion = 1
while (dist > EPSILON and abs(prev-dist) > EPSILON/100.):
    prev = dist
    # Para cada combinación de articulaciones:
    for i in range(len(th)):
        # cálculo de la cinemática inversa:
        current = len(th)-i-1
        if (type_arm[current] == 1): # Si es prismática
            w = np.sum(th[:current+1]) # Suma acumulativa de los ángulos anteriores
            u = [np.cos(w), np.sin(w)] # Vector unitario u
            d = np.dot(u, np.subtract(objetivo, O[i][-1])) # Distancia a recorrer
            a[current] += d # Se suma la distancia a recorrer
            # Se realiza una normalización de la longitud
            if (a[current] > Upper_limit[current]):
                a[current] = Upper_limit[current]
            if (a[current] < Lower_limit[current]):
                a[current] = Lower_limit[current]
        else: # Si es rotacional
            v1 = np.subtract(objetivo, O[i][current]) # Vector objetivo
            v2 = np.subtract(O[i][-1], O[i][current]) # Vector actual
            c1 = atan2(v1[1], v1[0]) # Ángulo asociado al vector v1
            c2 = atan2(v2[1], v2[0]) # Ángulo asociado al vector v2
            th[current] += c1 - c2 # Ajuste de ángulo para acercarse al objetivo final
            while th[current] > pi: # Se realiza una normalización del ángulo
                th[current] -= 2*pi
            while th[current] < -pi: # Se realiza una normalización del ángulo
                th[current] += 2*pi
            # Se realiza una normalización del ángulo
            if (th[current] > Upper_limit[current]):
                th[current] = Upper_limit[current]
            if (th[current] < Lower_limit[current]):
                th[current] = Lower_limit[current]
        O[i+1] = cin_dir(th, a)

    dist = np.linalg.norm(np.subtract(objetivo, O[-1][-1]))
    print("\n- Iteracion " + str(iteracion) + ':')
    muestra_origenes(O[-1])
    muestra_robot(O, objetivo)
    print("Distancia al objetivo = " + str(round(dist, 5)))
    iteracion += 1
    O[0] = O[-1]

if dist <= EPSILON:
    print("\n" + str(iteracion) + " iteraciones para converger.")
else:
    print("\nNo hay convergencia tras " + str(iteracion) + " iteraciones.")
print("- Umbral de convergencia epsilon: " + str(EPSILON))
print("- Distancia al objetivo:          " + str(round(dist, 5)))
print("- Valores finales de las articulaciones:")
for i in range(len(th)):
    print("  theta" + str(i+1) + " = " + str(round(th[i], 3)))
for i in range(len(th)):
    print("  L" + str(i+1) + "     = " + str(round(a[i], 3)))
    
plt.show(block=True)
