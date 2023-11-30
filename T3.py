import matplotlib
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import heapq


#Funcion para reconstruir camino mas corto
def reconstruct_path(prev, fin):
    nodo = fin
    path = []
    while nodo is not None:
        path.append(nodo)
        nodo = prev[nodo]
    return path[::-1]

#Funcion para calcular distancia de Manhattan
def dist_Man(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

#Funcion para pasar de coordenadas i,j a indice de la matriz de adyacencia
def idd(i,j,c):
    return int(i * c + j)

def idd_i(index,c):
    return int(index/c)
def idd_j(index,c):
    return index- int(index / c)*c

def lab_list_Ad(list_Ad,f,c,s,s2,p): #f:Filas c:columnas s:semilla p_prob
    list_peso = [[] for col in range(f * c)] #Se crea lista de pesos
    random.seed(s)
    state1 = random.getstate()
    random.seed(s2)
    state2 = random.getstate()
    for i in range(f):
        for j in range(c):
            random.setstate(state1)
            ranu = random.uniform(0, 1)
            state1 = random.getstate()

            random.setstate(state2)
            peso = round(random.uniform(1, 12))
            state2 = random.getstate()

            if i > 0 and ranu < p:#Con probabilidad p se crea pasillo entre i,j y i-1.j
                list_Ad[idd(i,j,c)].append(idd(i - 1, j,c))
                list_peso[idd(i, j, c)].append(peso)
                list_Ad[idd(i-1,j,c)].append(idd(i, j,c))
                list_peso[idd(i - 1, j, c)].append(peso)

            random.setstate(state1)
            ranu = random.uniform(0, 1)
            state1 = random.getstate()

            random.setstate(state2)
            peso = round(random.uniform(1, 12))
            state2 = random.getstate()

            if j > 0 and ranu < p:#Con probabilidad p se crea pasillo entre i,j y i.j-1
                list_Ad[idd(i,j,c)].append(idd(i, j-1,c))
                list_peso[idd(i,j,c)].append(peso)
                list_Ad[idd(i,j-1,c)].append(idd(i, j,c))
                list_peso[idd(i, j - 1, c)].append(peso)
    return([list_Ad,list_peso])

## Crear e Inicializar matriz de mapa de calor con ceros
def mat_Heat(list_Ad,f,c):
    mat = np.full((f * 2 + 1, c * 2 + 1), -100) #Muros con valor -100
    ## Recorrer los índices originales de las habitaciones
    for i in range(f):
        for j in range(c):
            ## Poner habitación en mat (multiplicar por 2 para dejar espacio a los pasillos y sumar 1 para saltarse el borde)
            mat[i * 2 + 1][j * 2 + 1] = 0
            ## Poner pasillos hacia abajo (si la habitación es de la última fila de habitaciones la saltamos, no puede tener pasillo hacia abajo y preguntaríamos por un nodo cuyo índice no existe).
            if i < f - 1 and idd(i + 1, j, c) in list_Ad[idd(i, j, c)]:
                mat[i * 2 + 2][j * 2 + 1] = 0
            ##Poner pasillos hacia la derecha (si la habitación es de la última columna de habitaciones, la saltamos, no puede tener pasillo hacia la derecha y preguntaríamos por un nodo cuyo índice no existe).
            if j < c - 1 and idd(i, j + 1, c) in list_Ad[idd(i, j, c)]:
                mat[i * 2 + 1][j * 2 + 2] = 0

    return mat

def dij_mod_bid(LAd,list_peso,i_com,j_com,i_fin,j_fin,f):
    # Inicializar conjuntos frontera y distancias
    start = idd(i_com, j_com, c)
    end = idd(i_fin, j_fin, c)
    frontier_start = []
    frontier_end = []
    heapq.heappush(frontier_start, (0, start))
    heapq.heappush(frontier_end, (0, end))
    expanded_start = set()
    expanded_end = set()
    dist_start = [float("inf")] * (f * f)  # Distancia actual desde el nodo inicial
    dist_end = [float("inf")] * (f * f)  # Distancia actual desde el nodo final
    prev_start = [None] * (f * f)  # Nodo previo desde el nodo inicial
    prev_end = [None] * (f * f)  # Nodo previo desde el nodo final
    dist_start[start] = 0
    dist_end[end] = 0
    union=-1

    # Iterar hasta que alguno de los conjuntos de frontera esté vacío
    while frontier_start and frontier_end:
        # Obtener el nodo con la distancia mínima desde el nodo inicial
        cost, node = heapq.heappop(frontier_start)

        # Si el nodo ya ha sido expandido desde el nodo final, se ha encontrado un camino mínimo
        if node in expanded_end:
            union=node
            break

        # Si el nodo no ha sido expandido, añádelo al conjunto de nodos expandidos desde el nodo inicial y expande todos sus vecinos
        if node not in expanded_start:
            expanded_start.add(node)
            for vec in LAd[node]:
                # Calcular la distancia alternativa
                alt = dist_start[node] + list_peso[node][LAd[node].index(vec)]
                # Si la distancia alternativa es menor que la distancia actual, actualizar la distancia y agregar el nodo a la frontera
                if alt < dist_start[vec]:
                    dist_start[vec] = alt
                    prev_start[vec] = node
                    heapq.heappush(frontier_start, (alt, vec))

        # Obtener el nodo con la distancia mínima desde el nodo final
        cost, node = heapq.heappop(frontier_end)

        # Si el nodo ya ha sido expandido desde el nodo inicial, se ha encontrado un camino mínimo
        if node in expanded_start:
            union=node
            break

        # Si el nodo no ha sido expandido, añádelo al conjunto de nodos expandidos desde el nodo inicial y expande todos sus vecinos
        if node not in expanded_end:
            expanded_end.add(node)
            for vec in LAd[node]:
             # Calcular la distancia alternativa
             alt = dist_end[node] + list_peso[node][LAd[node].index(vec)]
             # Si la distancia alternativa es menor que la distancia actual, actualizar la distancia y agregar el nodo a la frontera
             if alt < dist_end[vec]:
                 dist_end[vec] = alt
                 prev_end[vec] = node
                 heapq.heappush(frontier_end, (alt, vec))
    # Crear lista para almacenar el camino
    path = []
    # Si se ha encontrado un camino mínimo
    if union != -1:
        path_start=reconstruct_path(prev_start,union)
        path_end = reconstruct_path(prev_end, union)
        path_end.pop()
        path_end=path_end[::-1]
        path = path_start + path_end
    return [dist_start,dist_end,union,path]

def A_star_bid(LAd,list_peso,i_com,j_com,i_fin,j_fin,f):
    # Inicializar conjuntos frontera y distancias
    start = idd(i_com, j_com, c)
    end = idd(i_fin, j_fin, c)
    frontier_start = []
    frontier_end = []
    heapq.heappush(frontier_start, (0, start))
    heapq.heappush(frontier_end, (0, end))
    expanded_start = set()
    expanded_end = set()
    dist_start = [float("inf")] * (f * f)  # Distancia actual desde el nodo inicial
    dist_end = [float("inf")] * (f * f)  # Distancia actual desde el nodo final
    prev_start = [None] * (f * f)  # Nodo previo desde el nodo inicial
    prev_end = [None] * (f * f)  # Nodo previo desde el nodo final
    dist_start[start] = 0
    dist_end[end] = 0
    union=-1

    # Iterar hasta que alguno de los conjuntos de frontera esté vacío
    while frontier_start and frontier_end:
        # Obtener el nodo con la distancia mínima desde el nodo inicial
        cost, node = heapq.heappop(frontier_start)

        # Si el nodo ya ha sido expandido desde el nodo final, se ha encontrado un camino mínimo
        if node in expanded_end:
            union=node
            break

        # Si el nodo no ha sido expandido, añádelo al conjunto de nodos expandidos desde el nodo inicial y expande todos sus vecinos
        if node not in expanded_start:
            expanded_start.add(node)
            for vec in LAd[node]:
                # Calcular la distancia alternativa
                alt = dist_start[node] + list_peso[node][LAd[node].index(vec)]
                # Si la distancia alternativa es menor que la distancia actual, actualizar la distancia y agregar el nodo a la frontera
                if alt < dist_start[vec]:
                    dist_start[vec] = alt
                    prev_start[vec] = node
                    # Calculamos la distancia de Manhattan entre el nodo vecino y el nodo final como una estimación del costo restante
                    manhattan = dist_Man(idd_i(vec, c), idd_j(vec, c), i_fin, j_fin)
                    heapq.heappush(frontier_start, (alt + manhattan, vec))

        # Obtener el nodo con la distancia mínima desde el nodo final
        cost, node = heapq.heappop(frontier_end)

        # Si el nodo ya ha sido expandido desde el nodo inicial, se ha encontrado un camino mínimo
        if node in expanded_start:
            union=node
            break

        # Si el nodo no ha sido expandido, añádelo al conjunto de nodos expandidos desde el nodo inicial y expande todos sus vecinos
        if node not in expanded_end:
            expanded_end.add(node)
            for vec in LAd[node]:
             # Calcular la distancia alternativa
             alt = dist_end[node] + list_peso[node][LAd[node].index(vec)]
             # Si la distancia alternativa es menor que la distancia actual, actualizar la distancia y agregar el nodo a la frontera
             if alt < dist_end[vec]:
                 dist_end[vec] = alt
                 prev_end[vec] = node
                 # Calculamos la distancia de Manhattan entre el nodo vecino y el nodo final como una estimación del costo restante
                 manhattan = dist_Man(idd_i(vec, c), idd_j(vec, c), i_com, j_com)
                 heapq.heappush(frontier_end, (alt + manhattan, vec))
    # Crear lista para almacenar el camino
    path = []
    # Si se ha encontrado un camino mínimo
    if union != -1:
        path_start = reconstruct_path(prev_start, union)
        path_end = reconstruct_path(prev_end, union)
        path_end.pop()
        path_end = path_end[::-1]
        path = path_start + path_end
    return [dist_start, dist_end, union, path]

def pinta_dij(z,dist,c):
    for i in range(c):
        for j in range(c):
            if dist[idd(i,j,c)]!=np.inf:
                z[i * 2 + 1][j * 2 + 1] = dist[idd(i,j,c)]
                if j!=0 :
                    if dist[idd(i, j-1, c)] !=np.inf and z[i * 2 + 1][j * 2 ] != -100:
                        z[i * 2 + 1][j * 2 ] = min(dist[idd(i, j, c)],dist[idd(i, j-1, c)])
                if i!=0 :
                    if dist[idd(i-1, j, c)] !=np.inf and z[i * 2][j * 2 +1] != -100:
                        z[i * 2][j * 2 +1] = min(dist[idd(i, j, c)],dist[idd(i-1, j, c)])







##########INTRODUCCION PARAMETROS#########

f=20    #filas
c=20    #columnas
p=0.75     #Probabilidad de pasillo abierto
s=4 #semilla
s2=3 #semilla pesos
s3=9 #semilla inicio fin
mod_dij="modif" #modo del algoritmo de Dijkstra: "modif", "a_star"
##########INTRODUCCION PARAMETROS#########


#Se crea lista
list_Ad=[[] for col in range(f*c)]

#Se rellena la estructura indicada en modo con tamaño f y c
#creando los muros  aleatorios segun el parametro p
list_Ad=lab_list_Ad(list_Ad,f,c,s,s2,p)
list_peso=list_Ad[1]
list_Ad=list_Ad[0]

#Se pintan la matrices

z=mat_Heat(list_Ad,f,c)
z_end=mat_Heat(list_Ad,f,c)


random.seed(s3)
#Se elige el nodo de comienzo
i_com = round(random.uniform(0, f-1))
j_com = round(random.uniform(0,c-1))


# Se elige el nodo de final
i_fin = round(random.uniform(0, f - 1))
j_fin = round(random.uniform(0, c - 1))

while i_fin==i_com and j_fin==j_com:
    i_fin = round(random.uniform(0, f - 1))
    j_fin = round(random.uniform(0, c - 1))




#Se aplica el algoritmo de Dijkstra con o sin la modificacion propuesta
result=[]
t_start = time.time()
if mod_dij=="modif":
    result =dij_mod_bid(list_Ad,list_peso,i_com,j_com,i_fin,j_fin,f)
elif mod_dij=="a_star":
    result = A_star_bid(list_Ad, list_peso, i_com, j_com, i_fin, j_fin, f)

t_end=time.time()
t_tot=(t_end-t_start)
print("Tiempo: ",(t_end-t_start)* 1000000, " microsegundos")

dist_start=result[0]
dist_end=result[1]
union=result[2]
path=result[3]

print("Camino: ", path)

pinta_dij(z,dist_start,c)
pinta_dij(z_end,dist_end,c)

#Se marcan los nodos de inicio y final y la union
z[i_com * 2 + 1][j_com * 2 + 1] = -5
z[i_fin * 2 + 1][j_fin * 2 + 1] = -15
if union>-1:
    z[idd_i(union,f)* 2 + 1][idd_j(union,f) * 2 + 1] = -25

z_end[i_com * 2 + 1][j_com * 2 + 1] = -5
z_end[i_fin * 2 + 1][j_fin * 2 + 1] = -15
if union>-1:
    z_end[idd_i(union,f)* 2 + 1][idd_j(union,f) * 2 + 1] = -25


#Representacion grafica
#Se separan muros (negro) de pasillos y nodos recorridos(rojo)
z1=np.ma.masked_outside(z, 1, 1000)
z1_end=np.ma.masked_outside(z_end, 1, 1000)
z2=np.ma.masked_outside(z,-1000,-31)
z_union=np.ma.masked_outside(z,-30,-21)
z3=np.ma.masked_outside(z, -20,-10)
z4=np.ma.masked_outside(z, -9,-1)
z4=z4+300

plt.imshow(z4,cmap="summer") #Verde para inicio
plt.imshow(z3,cmap="gist_rainbow")#Rojo para final
plt.imshow(z2,cmap="gray")
plt.imshow(z_union,cmap="cool") #Punto de unión
plt.imshow(z1,cmap="plasma",interpolation="nearest") #Desde nodo start
plt.imshow(z1_end,cmap="YlGn",interpolation="nearest") #Desde nodo end

plt.gca().invert_yaxis()

matplotlib.pyplot.xticks([])
matplotlib.pyplot.yticks([])

plt.show()  #Mostramos el mapa