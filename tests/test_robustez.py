"""Pruebas: el algoritmo rápido debe dar lo mismo que la fuerza bruta, y la teoría
debe reproducir resultados conocidos."""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import robustez as rb  # noqa: E402


def gigante_fuerza_bruta(G, orden):
    """Quita nodos uno a uno y recalcula la componente gigante con NetworkX."""
    H = G.copy()
    N = G.number_of_nodes()
    S = [max(len(c) for c in nx.connected_components(H)) / N]
    for nodo in orden:
        H.remove_node(nodo)
        S.append(max((len(c) for c in nx.connected_components(H)), default=0) / N)
    return np.array(S)


@pytest.mark.parametrize("semilla", [1, 2, 3])
def test_newman_ziff_coincide_con_fuerza_bruta(semilla):
    rng = np.random.default_rng(semilla)
    G = nx.erdos_renyi_graph(120, 0.03, seed=semilla)
    for estrategia in (rb.orden_aleatorio, rb.orden_por_grado):
        orden = estrategia(G, rng)
        np.testing.assert_allclose(
            rb.curva_componente_gigante(G, orden), gigante_fuerza_bruta(G, orden)
        )


def test_curva_empieza_en_gigante_y_termina_en_cero():
    G = nx.barabasi_albert_graph(300, 2, seed=0)
    S = rb.curva_componente_gigante(G, rb.orden_aleatorio(G, np.random.default_rng(0)))
    assert S[0] == pytest.approx(1.0)      # BA siempre es conexa
    assert S[-1] == 0.0


def test_kappa_poisson():
    # Para una Poisson con media c, <k^2>/<k> = c + 1.
    from scipy.stats import poisson
    c = 6.0
    p = poisson.pmf(np.arange(80), c)
    assert rb.parametro_kappa(p) == pytest.approx(c + 1, rel=1e-6)
    assert rb.fraccion_critica_aleatoria(p) == pytest.approx(1 - 1 / c, rel=1e-6)


def test_teoria_poisson_sin_remocion():
    # Sin quitar nada, S resuelve S = 1 - exp(-c S). Para c = 6, S ≈ 0.99752.
    from scipy.stats import poisson
    p = poisson.pmf(np.arange(80), 6.0)
    S = rb.S_teorica_aleatoria(p, np.array([0.0]))[0]
    assert S == pytest.approx(1 - np.exp(-6 * S), abs=1e-9)
    assert S == pytest.approx(0.99752, abs=1e-4)


def test_teoria_se_anula_pasado_el_umbral():
    from scipy.stats import poisson
    p = poisson.pmf(np.arange(80), 6.0)
    fc = rb.fraccion_critica_aleatoria(p)
    S = rb.S_teorica_aleatoria(p, np.array([fc - 0.05, fc + 0.02]))
    assert S[0] > 0.01
    assert S[1] < 1e-6
