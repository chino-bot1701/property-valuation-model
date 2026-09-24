"""
Pipeline completo, sin credenciales y sin red.

    python scripts/run_demo.py

Genera el mercado, deflacta, entrena los tres modelos, valida año por año,
calibra los intervalos, mide la cobertura, corre el experimento de deflactar,
valúa el portafolio del grupo, escribe las gráficas y termina en aserciones.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import evaluacion, explicacion, portafolio as pf  # noqa: E402
from src.conformal import ancho_relativo  # noqa: E402
from src.deflactar import inflacion_acumulada  # noqa: E402
from src.generar_mercado import (ANIO_INICIAL, generar_operaciones,  # noqa: E402
                                 generar_portafolio)
from src.validacion_temporal import entrenar_final, validar  # noqa: E402

DATOS = RAIZ / "data" / "synthetic"
DOCS = RAIZ / "docs"
ALPHA = 0.10


def titulo(t: str) -> None:
    print(f"\n{t}\n" + "-" * len(t))


def main() -> int:
    DATOS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    titulo("1. Mercado sintético")
    ops = generar_operaciones()
    port = generar_portafolio()
    print(f"{len(ops):,} operaciones · {ops['anio'].min()}–{ops['anio'].max()} · "
          f"{ops['ciudad'].nunique()} ciudades · {ops['uso_suelo'].nunique()} usos")
    print(f"precio/m² nominal: mediana ${ops['precio_m2_nominal'].median():,.0f} "
          f"(de ${ops['precio_m2_nominal'].min():,.0f} a "
          f"${ops['precio_m2_nominal'].max():,.0f})")
    print(f"{len(port)} predios en la cartera del grupo")
    ops.to_csv(DATOS / "operaciones.csv", index=False)
    port.to_csv(DATOS / "portafolio.csv", index=False)

    titulo("2. Inflación del periodo")
    infl = inflacion_acumulada(ANIO_INICIAL)
    print(f"acumulada {ANIO_INICIAL}–2025: {infl:.1%}")
    print("Entrenar sin descontarla haría que el modelo prediga, sobre todo, esto.")

    titulo("3. Validación temporal (ventana expansiva)")
    r = validar(ops, alpha=ALPHA, deflactar=True)
    print(f"{r['anio'].nunique()} años evaluados, "
          f"{len(r):,} operaciones valuadas fuera de muestra")
    print("cada año se valúa con un modelo entrenado solo con los años previos")

    titulo("4. Los tres modelos")
    tabla = evaluacion.comparar(r)
    print(tabla.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    titulo("5. Intervalos de predicción")
    mi = evaluacion.metricas_intervalo(r, alpha=ALPHA)
    print(f"cobertura prometida   {mi['cobertura_objetivo']:.0%}")
    print(f"cobertura observada   {mi['cobertura_observada']:.1%}")
    print(f"ancho mediano         ±{mi['ancho_mediano'] / 2:.0%} del valor")
    print(f"ancho del 10% peor    ±{mi['ancho_p90'] / 2:.0%} del valor")

    r.to_csv(DATOS / "validacion.csv", index=False)
    anual = evaluacion.por_anio(r, alpha=ALPHA)
    print()
    print(anual.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    titulo("6. Dónde falla")
    print(evaluacion.por_segmento(r, "ciudad")
          .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print()
    print(evaluacion.por_segmento(r, "uso_suelo")
          .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    titulo("7. El experimento: qué pasa si NO se deflacta")
    comp = explicacion.comparar_deflactado(ops)
    for etiqueta in ("nominal", "deflactado"):
        sub = comp[comp["entrenamiento"] == etiqueta].head(4)
        lugar_anio = int(sub_lugar(comp, etiqueta))
        print(f"\nentrenado en pesos {etiqueta} — "
              f'"año de la operación" queda en el lugar {lugar_anio} de 20')
        for _, f in sub.iterrows():
            print(f"   {f['lugar']:>2}. {f['variable']:<34} {f['peso_relativo']:.2f}")
    comp.to_csv(DOCS / "importancias.csv", index=False)

    titulo("8. El portafolio del grupo")
    modelo, intervalo, _ = entrenar_final(ops, alpha=ALPHA)
    analisis = pf.analizar(port, modelo, intervalo)
    res = pf.resumen(analisis)
    print(f"invertido (pesos constantes)  ${res['invertido_real'] / 1e6:,.0f} M")
    print(f"valor hoy                     ${res['valor_hoy'] / 1e6:,.0f} M")
    print(f"plusvalía real de la cartera  {res['plusvalia_real_cartera']:+.1%}")
    print(f"por encima del rango          {res['por_encima_del_rango']} predios")
    print(f"por debajo del rango          {res['por_debajo_del_rango']} predios")
    print(f"dentro de mercado             {res['dentro_de_mercado']} predios")

    inyectadas = int(analisis["_fuera_de_precio"].sum())
    detectadas = int(analisis.loc[analisis["_fuera_de_precio"] == 1, "fuera_de_rango"].sum())
    falsas = int(analisis.loc[analisis["_fuera_de_precio"] == 0, "fuera_de_rango"].sum())
    print(f"\ncontra la etiqueta oculta: {detectadas} de {inyectadas} compras fuera "
          f"de precio señaladas, con {falsas} señalamientos que no lo eran")
    print(f"plusvalía real quitando los predios marcados: "
          f"{res['plusvalia_sin_marcados']:+.1%} "
          f"(contra {res['plusvalia_real_cartera']:+.1%} con todo)")

    print("\nEl umbral es una decisión de política, no del modelo:")
    niveles = {f"{int((1 - a) * 100)}%": entrenar_final(ops, alpha=a)[1]
               for a in (0.10, 0.20, 0.30)}
    print(pf.tamiz(port, modelo, niveles)
          .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    print("\nLos cinco con mayor brecha contra el mercado de su año:")
    cols = ["predio_id", "ciudad", "uso_suelo", "superficie_m2",
            "pagado_m2_real", "justo_m2", "brecha_pct", "veredicto"]
    print(analisis.head(5)[cols].to_string(index=False,
                                           float_format=lambda v: f"{v:,.2f}"))
    analisis.to_csv(DATOS / "portafolio_valuado.csv", index=False)

    titulo("9. Gráficas")
    _grafica_modelos(tabla, DOCS / "modelos.png")
    _grafica_intervalos(r, DOCS / "intervalos.png")
    _grafica_importancias(comp, DOCS / "deflactar.png")
    _grafica_dependencia(modelo, ops, DOCS / "dependencia_distancia.png")
    for n in ("modelos", "intervalos", "deflactar", "dependencia_distancia"):
        print(f"-> {DOCS / (n + '.png')}")

    titulo("10. Aserciones")
    return _verificar(tabla, mi, anual, comp, res, detectadas, falsas)


def _fila_anio(comp: pd.DataFrame, etiqueta: str) -> pd.Series:
    return comp[(comp["entrenamiento"] == etiqueta)
                & (comp["columna"] == "anios_desde_inicio")].iloc[0]


def sub_lugar(comp: pd.DataFrame, etiqueta: str) -> int:
    return int(_fila_anio(comp, etiqueta)["lugar"])


def sub_peso(comp: pd.DataFrame, etiqueta: str) -> float:
    return float(_fila_anio(comp, etiqueta)["peso_relativo"])


# --------------------------------------------------------------------------
# Gráficas
# --------------------------------------------------------------------------

def _grafica_modelos(tabla: pd.DataFrame, destino: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    t = tabla.iloc[::-1]
    ax.barh(t["modelo"], t["MdAPE"] * 100, color=["#2b6cb0", "#4a90d9", "#a0aec0"])
    for y, v in enumerate(t["MdAPE"] * 100):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("error porcentual mediano (MdAPE) — menos es mejor")
    ax.set_title("El modelo contra la regla que el negocio ya usa")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(destino, dpi=140); plt.close(fig)


def _grafica_intervalos(r: pd.DataFrame, destino: Path) -> None:
    """Ancho del intervalo contra superficie: la prueba visual de que se adapta."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))

    anchos = ancho_relativo(r["lo"].to_numpy(), r["hi"].to_numpy())
    a1.scatter(r["superficie_m2"], anchos * 100, s=6, alpha=0.25, color="#2b6cb0")
    a1.set_xscale("log")
    a1.set_xlabel("superficie del predio (m², escala log)")
    a1.set_ylabel("ancho del intervalo (% del valor)")
    a1.set_title("El rango se abre donde hay menos comparables")
    a1.spines[["top", "right"]].set_visible(False)

    porc = r.groupby("anio").apply(
        lambda b: float(np.mean((b["y"] >= b["lo"]) & (b["y"] <= b["hi"]))),
        include_groups=False)
    a2.bar(porc.index.astype(str), porc.to_numpy() * 100, color="#2b6cb0")
    a2.axhline(90, color="#c05621", ls="--", lw=1.2, label="prometido: 90%")
    a2.set_ylim(0, 105)
    a2.set_ylabel("cobertura observada (%)")
    a2.set_title("Lo prometido contra lo cumplido, año por año")
    a2.legend(frameon=False)
    a2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(); fig.savefig(destino, dpi=140); plt.close(fig)


def _grafica_importancias(comp: pd.DataFrame, destino: Path) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
    for ax, etiqueta, titulo_ in zip(
            ejes, ["nominal", "deflactado"],
            ["Entrenado en pesos corrientes", "Entrenado en pesos constantes"]):
        sub = comp[comp["entrenamiento"] == etiqueta].head(7).iloc[::-1]
        colores = ["#c05621" if c == "anios_desde_inicio" else "#2b6cb0"
                   for c in sub["columna"]]
        ax.barh(sub["variable"], sub["peso_relativo"], color=colores)
        ax.set_title(titulo_, fontsize=11)
        ax.set_xlabel("importancia relativa")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("El año de la operación (naranja) deja de mandar al descontar inflación",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(destino, dpi=140); plt.close(fig)


def _grafica_dependencia(modelo, ops: pd.DataFrame, destino: Path) -> None:
    valores = np.log(np.linspace(0.6, 25, 30))
    dp = explicacion.dependencia_parcial(modelo, ops.sample(600, random_state=1),
                                         "log_dist_centro", valores)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.exp(dp["log_dist_centro"]), dp["precio_m2"],
            color="#2b6cb0", lw=2)
    ax.set_xlabel("distancia al centro urbano (km)")
    ax.set_ylabel("precio estimado por m² (pesos constantes)")
    ax.set_title("Cómo usa el modelo la distancia al centro")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(destino, dpi=140); plt.close(fig)


# --------------------------------------------------------------------------

def _verificar(tabla, mi, anual, comp, res, detectadas, falsas) -> int:
    md = tabla.set_index("modelo")["MdAPE"]
    lugar_nominal = sub_lugar(comp, "nominal")
    lugar_deflactado = sub_lugar(comp, "deflactado")
    peso_nominal = sub_peso(comp, "nominal")
    peso_deflactado = sub_peso(comp, "deflactado")

    pruebas = [
        ("el modelo le gana a la mediana por ciudad y uso",
         md["ridge"] < md["mediana por ciudad y uso"]),
        # No es un typo ni una decepción: en estos datos el lineal gana, y por
        # eso es el que se despliega. La discusión está en el README.
        ("la regresión lineal le gana al gradient boosting",
         md["ridge"] < md["gradient boosting"]),
        ("el error mediano del modelo está por debajo de 15%",
         md["ridge"] < 0.15),
        ("la cobertura observada alcanza lo prometido (±3 puntos)",
         mi["cobertura_observada"] >= mi["cobertura_objetivo"] - 0.03),
        ("ningún año se queda por debajo de 80% de cobertura",
         bool((anual["cobertura"] >= 0.80).all())),
        ("el intervalo es útil: ancho mediano por debajo de ±40%",
         mi["ancho_mediano"] / 2 < 0.40),
        ("sin deflactar, el año de la operación entra al top 6 de variables",
         lugar_nominal <= 6),
        ("deflactar lo saca del top 10",
         lugar_deflactado > 10),
        ("sin deflactar, el año pesa al menos 4 veces más",
         peso_nominal >= 4 * peso_deflactado),
        # Con un intervalo del 90% —ancho a propósito— se atrapan las compras
        # más caras, no todas. El tamiz de la sección 8 muestra el intercambio.
        ("señala al menos 5 de las 12 compras fuera de precio de la cartera",
         detectadas >= 5),
        ("la mitad de lo que señala es real",
         detectadas >= falsas),
    ]
    fallos = 0
    for desc, ok in pruebas:
        print(f"  [{'OK ' if ok else 'FALLA'}] {desc}")
        fallos += 0 if ok else 1
    print(f"\n{len(pruebas) - fallos}/{len(pruebas)} aserciones pasan")
    return 0 if fallos == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
