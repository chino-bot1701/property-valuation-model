"""
Validación con ventana expansiva, por año.

Un `train_test_split` aleatorio sobre precios entrena con operaciones de 2025 y
evalúa con operaciones de 2019. El modelo llega al año de prueba sabiendo ya
hacia dónde se movió el mercado, y el error sale artificialmente bajo. Después
se pone en producción, se le pide valuar un predio de un año que todavía no
existe, y no se parece al reporte.

Lo que se hace aquí es lo único que reproduce esa situación:

    entrena con los años ..T-1   →   valúa el año T   →   avanza T

Hay un detalle propio de los intervalos. La calibración conforme necesita datos
que el modelo no haya visto, y se usa **el último año del entrenamiento**, no
una muestra aleatoria de todo el historial. La razón: el conjunto de
calibración debería parecerse lo más posible a lo que viene, y 2024 se parece a
2025 mucho más que 2019.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .conformal import IntervaloConforme
from .features import objetivo
from .modelo import LineaBaseMediana, ModeloBoosting, ModeloLineal

# Antes de este año no hay historial suficiente para entrenar y calibrar.
PRIMER_ANIO_PRUEBA = 2021


def validar(d: pd.DataFrame, alpha: float = 0.10,
            deflactar: bool = True,
            primer_anio: int = PRIMER_ANIO_PRUEBA) -> pd.DataFrame:
    """Corre la ventana expansiva y devuelve las predicciones de cada año.

    `deflactar=False` existe para poder medir, y no solo afirmar, qué le pasa
    al modelo cuando se entrena sobre precios nominales.
    """
    d = d.copy()
    d["y"] = objetivo(d, deflactar=deflactar)
    anios = sorted(d["anio"].unique())

    bloques = []
    for anio in anios:
        if anio < primer_anio:
            continue

        historico = d[d["anio"] < anio]
        prueba = d[d["anio"] == anio]
        if len(historico) < 400 or prueba.empty:
            continue

        anio_calibracion = historico["anio"].max()
        entrena = historico[historico["anio"] < anio_calibracion]
        calibra = historico[historico["anio"] == anio_calibracion]
        if len(entrena) < 200 or len(calibra) < 100:
            continue

        base = LineaBaseMediana().entrenar(historico, historico["y"])
        lineal = ModeloLineal().entrenar(historico, historico["y"])
        boosting = ModeloBoosting().entrenar(historico, historico["y"])
        # El intervalo envuelve al modelo que de verdad se despliega. Cuál es
        # ese lo decidió la comparación, no el gusto: ver el README.
        punto = ModeloLineal().entrenar(entrena, entrena["y"])
        intervalo = IntervaloConforme(alpha=alpha).entrenar(
            punto, entrena, entrena["y"], calibra, calibra["y"])

        lo, hi = intervalo.intervalo(punto, prueba)
        bloque = prueba.copy()
        bloque["pred_base"] = base.predecir(prueba)
        bloque["pred_lineal"] = lineal.predecir(prueba)
        bloque["pred_boosting"] = boosting.predecir(prueba)
        bloque["lo"] = lo
        bloque["hi"] = hi
        bloque["entrenado_hasta"] = anio - 1
        bloque["n_entrenamiento"] = len(historico)
        bloques.append(bloque)

    if not bloques:
        raise RuntimeError("Ningún año pudo evaluarse; revisa el rango de fechas.")
    return pd.concat(bloques, ignore_index=True)


def entrenar_final(d: pd.DataFrame, alpha: float = 0.10):
    """Ajusta con todo el historial, que es lo que se desplegaría.

    Devuelve el boosting, el intervalo calibrado y la línea base, listos para
    valuar predios nuevos. La calibración vuelve a usar el último año.
    """
    d = d.copy()
    d["y"] = objetivo(d)
    ultimo = d["anio"].max()
    entrena = d[d["anio"] < ultimo]
    calibra = d[d["anio"] == ultimo]

    modelo = ModeloLineal().entrenar(d, d["y"])
    punto_calibracion = ModeloLineal().entrenar(entrena, entrena["y"])
    intervalo = IntervaloConforme(alpha=alpha).entrenar(
        punto_calibracion, entrena, entrena["y"], calibra, calibra["y"])
    base = LineaBaseMediana().entrenar(d, d["y"])
    return modelo, intervalo, base
