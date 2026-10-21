"""
Robustez de redes ante fallas aleatorias y ataques dirigidos.

Dos piezas:
1. Simulación: se quitan nodos y se mide el tamaño de la componente gigante.
   Se usa el algoritmo de Newman-Ziff (2000): en vez de quitar nodos uno por uno
   y recalcular todo (lento, ~N^2), se AGREGAN en orden inverso y se van uniendo
   clusters con una estructura union-find (casi lineal en N).
2. Teoría: percolación en redes con distribución de grados arbitraria
   (Molloy-Reed 1995, Callaway et al. 2000, Cohen et al. 2000), calculada a
   partir de la distribución de grados real de cada red.
"""

import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# Simulación
# ---------------------------------------------------------------------------

def _encontrar(padre, i):
    """Raíz del cluster de i, comprimiendo el camino de paso."""
    while padre[i] != i:
        padre[i] = padre[padre[i]]
        i = padre[i]
    return i


def curva_componente_gigante(G, orden_remocion):
    """
    Tamaño relativo de la componente gigante S(f) al quitar nodos en el orden dado.

    Devuelve un arreglo de longitud N+1: S[j] es la fracción de los N nodos
    originales que queda en el cluster más grande después de quitar j nodos.

    Truco (Newman-Ziff): recorremos la remoción al revés. Arrancamos con la red
    vacía y vamos agregando nodos desde el último en ser quitado; cada vez que
    un nodo entra, lo unimos con sus vecinos que ya están presentes.
    """
    nodos = list(orden_remocion)
    N = len(nodos)
    indice = {n: i for i, n in enumerate(G.nodes())}
    vecinos = [[] for _ in range(N)]
    for u, v in G.edges():
        vecinos[indice[u]].append(indice[v])
        vecinos[indice[v]].append(indice[u])

    padre = np.arange(N)
    tamano = np.ones(N, dtype=np.int64)
    presente = np.zeros(N, dtype=bool)

    mayor = 0
    S_inverso = np.zeros(N + 1)          # S_inverso[m] = gigante con m nodos presentes
    for m, nodo in enumerate(reversed(nodos), start=1):
        i = indice[nodo]
        presente[i] = True
        for j in vecinos[i]:
            if presente[j]:
                ri, rj = _encontrar(padre, i), _encontrar(padre, j)
                if ri != rj:
                    if tamano[ri] < tamano[rj]:
                        ri, rj = rj, ri
                    padre[rj] = ri
                    tamano[ri] += tamano[rj]
        mayor = max(mayor, tamano[_encontrar(padre, i)])
        S_inverso[m] = mayor / N

    # j nodos quitados  <=>  N - j nodos presentes
    return S_inverso[::-1]


def orden_aleatorio(G, rng):
    """Fallas: todos los nodos tienen la misma probabilidad de caer."""
    nodos = np.array(list(G.nodes()))
    return nodos[rng.permutation(len(nodos))]


def orden_por_grado(G, rng):
    """
    Ataque: se quitan primero los nodos más conectados (grado inicial).
    Los empates se rompen al azar para no depender del orden de los índices.
    """
    nodos = np.array(list(G.nodes()))
    grados = np.array([G.degree(n) for n in nodos])
    desempate = rng.random(len(nodos))
    return nodos[np.lexsort((desempate, -grados))]


def simular(generador, estrategia, n_repeticiones, semilla=0):
    """
    Repite la remoción sobre varias redes independientes.
    Devuelve (fracciones_quitadas, matriz de curvas S con una fila por repetición).
    """
    rng = np.random.default_rng(semilla)
    curvas = []
    for r in range(n_repeticiones):
        G = generador(int(rng.integers(0, 2**31)))
        curvas.append(curva_componente_gigante(G, estrategia(G, rng)))
    curvas = np.array(curvas)
    f = np.arange(curvas.shape[1]) / (curvas.shape[1] - 1)
    return f, curvas


# ---------------------------------------------------------------------------
# Teoría (funciones generatrices)
# ---------------------------------------------------------------------------

def distribucion_grados(G):
    """p[k] = fracción de nodos con grado k."""
    grados = np.array([d for _, d in G.degree()])
    return np.bincount(grados) / len(grados)


def parametro_kappa(p):
    """kappa = <k^2> / <k>. La red tiene componente gigante si kappa > 2."""
    k = np.arange(len(p))
    return (k**2 @ p) / (k @ p)


def fraccion_critica_aleatoria(p):
    """
    Fracción de nodos que hay que quitar al azar para destruir la componente
    gigante (Cohen et al. 2000):  f_c = 1 - 1/(kappa - 1).
    """
    return 1.0 - 1.0 / (parametro_kappa(p) - 1.0)


def S_teorica_aleatoria(p, f, iteraciones=2000, tol=1e-12):
    """
    Tamaño esperado de la componente gigante tras quitar una fracción f al azar,
    para una red aleatoria con la misma distribución de grados p.

    Con q = 1 - f la fracción de nodos que sobreviven:
        u = 1 - q + q * G1(u)          (prob. de que una arista NO lleve al gigante)
        S = q * (1 - G0(u))
    con G0(x) = sum p_k x^k  y  G1(x) = G0'(x) / G0'(1).
    """
    k = np.arange(len(p))
    k_medio = k @ p
    q_k = (k[1:] * p[1:]) / k_medio            # distribución de grado "en exceso"

    def G0(x):
        return np.polyval(p[::-1], x)

    def G1(x):
        return np.polyval(q_k[::-1], x)

    S = np.zeros_like(f, dtype=float)
    for idx, fi in enumerate(f):
        q = 1.0 - fi
        u = 0.0                                 # arranque lejos del punto fijo trivial u=1
        for _ in range(iteraciones):
            u_nuevo = 1.0 - q + q * G1(u)
            if abs(u_nuevo - u) < tol:
                break
            u = u_nuevo
        S[idx] = q * (1.0 - G0(u))
    return np.clip(S, 0.0, 1.0)


def umbral_empirico(f, S_media, nivel=0.05):
    """Primera fracción quitada en la que el gigante cae por debajo de `nivel`."""
    por_debajo = np.nonzero(S_media < nivel)[0]
    return f[por_debajo[0]] if len(por_debajo) else np.nan


# ---------------------------------------------------------------------------
# Las tres familias de redes, todas con el mismo N y el mismo grado medio
# ---------------------------------------------------------------------------

def generadores(N=2000, k_medio=6):
    return {
        "Erdős-Rényi": lambda s: nx.erdos_renyi_graph(N, k_medio / (N - 1), seed=s),
        "Watts-Strogatz": lambda s: nx.watts_strogatz_graph(N, k_medio, 0.1, seed=s),
        "Barabási-Albert": lambda s: nx.barabasi_albert_graph(N, k_medio // 2, seed=s),
    }
