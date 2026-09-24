"""
El modelo aplicado a lo que el grupo ya compró.

Un modelo de valuación que solo predice precios es un ejercicio. Lo que lo
vuelve útil es apuntarlo a la cartera propia y contestar dos preguntas que
nadie puede responder con una hoja de cálculo:

**¿Se pagó de más?** Se valúa cada predio con el mercado del año en que se
compró —no con el de hoy, que sería juzgar una decisión de 2019 con
información de 2025— y se compara contra lo que efectivamente se pagó, en pesos
constantes. Un predio fuera del intervalo del 90% no es "una mala compra": es
un caso que amerita que alguien revise el expediente. La diferencia importa.

**¿Cuánto vale hoy?** El mismo predio, con las mismas características, valuado
con el mercado actual. La diferencia contra lo pagado, ya descontada la
inflación, es plusvalía real — no el espejismo de que todo sube porque los
pesos valen menos.

La lista que sale ordenada por brecha es la lista corta de inversión: dónde
volver a comprar, y qué revisar antes de volver a hacerlo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .deflactar import a_pesos_constantes
from .generar_mercado import ANIO_FINAL, ANIO_INICIAL


def valuar(portafolio: pd.DataFrame, modelo, intervalo,
           anio_valuacion: int | None = None) -> pd.DataFrame:
    """Valúa cada predio con el mercado de un año dado.

    Con `anio_valuacion=None` se usa el año de adquisición de cada predio, que
    es la comparación justa: qué decía el mercado cuando se firmó.
    """
    d = portafolio.copy()
    if anio_valuacion is not None:
        d["anios_desde_inicio"] = anio_valuacion - ANIO_INICIAL
        d["anio"] = anio_valuacion

    pred = modelo.predecir(d)
    lo, hi = intervalo.intervalo(modelo, d)
    return pd.DataFrame({
        "valor_m2": np.exp(pred),
        "valor_m2_lo": np.exp(lo),
        "valor_m2_hi": np.exp(hi),
    }, index=d.index)


def analizar(portafolio: pd.DataFrame, modelo, intervalo) -> pd.DataFrame:
    """Tabla completa: qué se pagó, qué decía el mercado, y qué vale hoy."""
    d = portafolio.copy()

    # Lo pagado, llevado a pesos de hoy para que todo sea comparable.
    d["pagado_m2_real"] = a_pesos_constantes(d, "precio_m2_pagado").to_numpy()
    d["pagado_total_real"] = d["pagado_m2_real"] * d["superficie_m2"]

    # (1) El mercado del año de la compra.
    en_su_momento = valuar(d, modelo, intervalo, anio_valuacion=None)
    d["justo_m2"] = en_su_momento["valor_m2"]
    d["justo_m2_lo"] = en_su_momento["valor_m2_lo"]
    d["justo_m2_hi"] = en_su_momento["valor_m2_hi"]
    d["brecha_pct"] = d["pagado_m2_real"] / d["justo_m2"] - 1.0

    # Fuera del intervalo = amerita revisión, no "mala compra". El intervalo ya
    # absorbe la incertidumbre normal del mercado; salirse de él significa que
    # el precio no se explica con las características del predio.
    d["fuera_de_rango"] = ((d["pagado_m2_real"] < d["justo_m2_lo"])
                           | (d["pagado_m2_real"] > d["justo_m2_hi"]))
    d["veredicto"] = np.select(
        [d["pagado_m2_real"] > d["justo_m2_hi"],
         d["pagado_m2_real"] < d["justo_m2_lo"]],
        ["revisar: por encima del rango", "compra por debajo del rango"],
        default="dentro de mercado")

    # (2) El mercado de hoy, sobre el mismo predio.
    hoy = valuar(d, modelo, intervalo, anio_valuacion=ANIO_FINAL)
    d["valor_hoy_m2"] = hoy["valor_m2"]
    d["valor_hoy_total"] = d["valor_hoy_m2"] * d["superficie_m2"]
    # Plusvalía REAL: los dos números están en pesos constantes, así que lo que
    # queda es apreciación por encima de la inflación, no la inflación misma.
    d["plusvalia_real_pct"] = d["valor_hoy_m2"] / d["pagado_m2_real"] - 1.0

    return d.sort_values("brecha_pct", ascending=False).reset_index(drop=True)


def tamiz(portafolio: pd.DataFrame, modelo, intervalos: dict) -> pd.DataFrame:
    """El mismo análisis con distintos niveles de confianza.

    El umbral no es una propiedad del modelo, es una **decisión de política**.
    Un intervalo del 90% es un tamiz estricto: cuando marca, casi siempre vale
    la pena abrir el expediente, pero deja pasar compras moderadamente caras.
    Uno del 70% marca más y se equivoca más.

    Cuál conviene depende de qué cuesta cada error. Revisar de más cuesta horas
    de un analista; dejar pasar una compra cara cuesta la compra. Esta tabla
    pone los dos números enfrente para que la decisión se tome con datos y no
    por costumbre.
    """
    filas = []
    for etiqueta, intervalo in intervalos.items():
        a = analizar(portafolio, modelo, intervalo)
        marcados = a["fuera_de_rango"]
        reales = a["_fuera_de_precio"] == 1
        filas.append({
            "confianza": etiqueta,
            "marcados": int(marcados.sum()),
            "aciertos": int((marcados & reales).sum()),
            "falsos": int((marcados & ~reales).sum()),
            "recall": float((marcados & reales).sum() / reales.sum()),
            "precision": float((marcados & reales).sum() / max(marcados.sum(), 1)),
        })
    return pd.DataFrame(filas)


def resumen(analisis: pd.DataFrame) -> dict:
    """Los números que iría a ver un director de inversiones."""
    return {
        "predios": len(analisis),
        "invertido_real": float(analisis["pagado_total_real"].sum()),
        "valor_hoy": float(analisis["valor_hoy_total"].sum()),
        "plusvalia_real_cartera": float(
            analisis["valor_hoy_total"].sum() / analisis["pagado_total_real"].sum() - 1),
        "por_encima_del_rango": int(
            (analisis["veredicto"] == "revisar: por encima del rango").sum()),
        "por_debajo_del_rango": int(
            (analisis["veredicto"] == "compra por debajo del rango").sum()),
        "dentro_de_mercado": int((analisis["veredicto"] == "dentro de mercado").sum()),
        "brecha_mediana": float(analisis["brecha_pct"].median()),
        # La plusvalía sin las compras que el modelo manda a revisar. La
        # diferencia contra la cifra de arriba dice cuánto le costó a la
        # cartera un puñado de operaciones.
        "plusvalia_sin_marcados": float(
            analisis.loc[~analisis["fuera_de_rango"], "valor_hoy_total"].sum()
            / analisis.loc[~analisis["fuera_de_rango"], "pagado_total_real"].sum() - 1),
    }
