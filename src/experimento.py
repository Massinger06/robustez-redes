"""
Corre el experimento completo y guarda resultados y figuras.

    python src/experimento.py

Tres familias de redes con el MISMO número de nodos y el MISMO grado medio;
lo único que cambia es la regla con que se conectan. A cada una se le aplican
fallas aleatorias y ataques a los nodos más conectados.
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import robustez as rb

RAIZ = Path(__file__).resolve().parents[1]
FIG = RAIZ / "figures"
RES = RAIZ / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

N, K_MEDIO, REPETICIONES, SEMILLA = 2000, 6, 30, 42

# Paleta (categórica validada, ver README): azul = fallas, naranja = ataque.
AZUL, NARANJA, TINTA, TINTA_SUAVE, REJILLA = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": TINTA_SUAVE, "axes.labelcolor": TINTA,
    "xtick.color": TINTA_SUAVE, "ytick.color": TINTA_SUAVE, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def main():
    t0 = time.time()
    filas, curvas = [], {}
    for i, (nombre, gen) in enumerate(rb.generadores(N, K_MEDIO).items()):
        f, S_fal = rb.simular(gen, rb.orden_aleatorio, REPETICIONES, SEMILLA + 10 * i)
        _, S_ata = rb.simular(gen, rb.orden_por_grado, REPETICIONES, SEMILLA + 10 * i + 1)

        # Distribución de grados promediada sobre varias realizaciones para la teoría.
        rng = np.random.default_rng(SEMILLA + 100 + i)
        ps = [rb.distribucion_grados(gen(int(rng.integers(0, 2**31)))) for _ in range(REPETICIONES)]
        largo = max(len(p) for p in ps)
        p = np.mean([np.pad(q, (0, largo - len(q))) for q in ps], axis=0)

        f_teo = np.linspace(0, 1, 401)
        curvas[nombre] = dict(f=f, fal=S_fal, ata=S_ata, f_teo=f_teo,
                              S_teo=rb.S_teorica_aleatoria(p, f_teo))

        # Umbral por repetición -> media y desviación estándar.
        def umbrales(M):
            u = np.array([rb.umbral_empirico(f, c) for c in M])
            return u.mean(), u.std(ddof=1)

        fc_fal, fc_fal_sd = umbrales(S_fal)
        fc_ata, fc_ata_sd = umbrales(S_ata)
        filas.append({
            "red": nombre,
            "kappa": rb.parametro_kappa(p),
            "grado_max_medio": float(np.mean([len(q) - 1 for q in ps])),
            "fc_teoria_fallas": rb.fraccion_critica_aleatoria(p),
            "fc_sim_fallas": fc_fal, "fc_sim_fallas_sd": fc_fal_sd,
            "fc_sim_ataque": fc_ata, "fc_sim_ataque_sd": fc_ata_sd,
        })

    tabla = pd.DataFrame(filas)
    tabla.to_csv(RES / "umbrales.csv", index=False)
    graficar_curvas(curvas, tabla)
    graficar_umbrales(tabla)
    print(tabla.round(3).to_string(index=False))
    print(f"\nTiempo total: {time.time() - t0:.1f} s")


def graficar_curvas(curvas, tabla):
    fig, ejes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (nombre, c) in zip(ejes, curvas.items()):
        ax.grid(True, color=REJILLA, linewidth=0.8)
        ax.set_axisbelow(True)
        for M, color, etiqueta in ((c["fal"], AZUL, "Fallas aleatorias"),
                                    (c["ata"], NARANJA, "Ataque a los más conectados")):
            media, sd = M.mean(0), M.std(0, ddof=1)
            ax.fill_between(c["f"], media - 2 * sd, media + 2 * sd, color=color, alpha=0.18, lw=0)
            ax.plot(c["f"], media, color=color, lw=2, label=etiqueta)
        ax.plot(c["f_teo"], c["S_teo"], color=TINTA, lw=1.4, ls="--",
                label="Teoría (fallas, red aleatoria)")
        fc = tabla.set_index("red").loc[nombre, "fc_teoria_fallas"]
        ax.axvline(fc, color=TINTA_SUAVE, lw=1, ls=":")
        ax.text(fc, 1.02, f"$f_c$ = {fc:.2f}", ha="center", va="bottom", fontsize=9, color=TINTA_SUAVE)
        ax.set_title(nombre, fontsize=11, color=TINTA, pad=18)
        ax.set_xlabel("Fracción de nodos quitados, f")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.0)
    ejes[0].set_ylabel("Tamaño relativo de la componente gigante, S")
    ejes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle(f"Misma cantidad de nodos ({N}) y conexiones (grado medio {K_MEDIO}), "
                 "tres comportamientos distintos", fontsize=12, color=TINTA, y=1.03)
    fig.text(0.5, -0.03, f"Media de {REPETICIONES} redes independientes; banda = ±2 desviaciones estándar. "
             "Línea punteada vertical: umbral teórico de Molloy-Reed / Cohen.",
             ha="center", fontsize=8.5, color=TINTA_SUAVE)
    fig.tight_layout()
    fig.savefig(FIG / "curvas_robustez.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def graficar_umbrales(tabla):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.grid(True, axis="x", color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    y = np.arange(len(tabla))[::-1]
    alto = 0.34
    ax.barh(y + alto / 2 + 0.01, tabla["fc_sim_fallas"], alto, xerr=2 * tabla["fc_sim_fallas_sd"],
            color=AZUL, label="Fallas aleatorias", error_kw=dict(ecolor=TINTA_SUAVE, lw=1))
    ax.barh(y - alto / 2 - 0.01, tabla["fc_sim_ataque"], alto, xerr=2 * tabla["fc_sim_ataque_sd"],
            color=NARANJA, label="Ataque a los más conectados", error_kw=dict(ecolor=TINTA_SUAVE, lw=1))
    for yi, (_, fila) in zip(y, tabla.iterrows()):
        for clave, dy in (("fallas", alto / 2 + 0.01), ("ataque", -alto / 2 - 0.01)):
            x = fila[f"fc_sim_{clave}"] + 2 * fila[f"fc_sim_{clave}_sd"] + 0.015
            ax.text(x, yi + dy, f"{fila[f'fc_sim_{clave}']:.2f}", va="center", fontsize=9, color=TINTA)
    ax.set_yticks(y, tabla["red"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Fracción de nodos que hay que quitar para romper la red\n"
                  "(componente gigante < 5% de los nodos)")
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.set_title("¿Cuánto aguanta cada red?", fontsize=11, color=TINTA, loc="left", pad=26)
    fig.tight_layout()
    fig.savefig(FIG / "umbrales.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
