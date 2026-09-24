"""
Las métricas que usa la industria de valuación, no las de un curso.

El R² y el RMSE son malas métricas aquí, por una razón concreta: los predios
van de $450 a $35,000 el m², así que el RMSE lo domina un puñado de predios
caros y el R² sale alto solo por acertarle al orden de magnitud. Ninguno de los
dos responde la pregunta del negocio, que es *¿de cuánto se equivoca, en
porcentaje?*

Lo que sí se usa en valuación automatizada:

| Métrica | Qué responde |
|---|---|
| **MdAPE** | de cuánto se equivoca el caso típico, en % |
| **PPE10** | qué proporción de valuaciones cae dentro de ±10% del precio real |
| **PPE20** | lo mismo con ±20%, la tolerancia habitual de un avalúo |
| **Cobertura** | de los intervalos del 90%, cuántos contuvieron el precio real |
| **Ancho** | qué tan ancho es ese intervalo, en % del valor |

Las dos últimas van juntas y hay que leerlas juntas. Un intervalo de −90% a
+400% cubre todo y no sirve de nada; la gracia es cubrir lo prometido siendo
lo más angosto posible.

Se usan medianas y no promedios porque un puñado de operaciones atípicas
—remates, ventas entre partes relacionadas— existen en el padrón y no vienen
marcadas. El promedio se las come; la mediana describe el caso típico, que es
lo que se va a valuar el 97% de las veces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .conformal import ancho_relativo, cobertura

MODELOS = {
    "mediana por ciudad y uso": "pred_base",
    "ridge": "pred_lineal",
    "gradient boosting": "pred_boosting",
}


def error_relativo(y_log: np.ndarray, pred_log: np.ndarray) -> np.ndarray:
    """Error porcentual absoluto, calculado en pesos y no en logaritmos."""
    real = np.exp(np.asarray(y_log, dtype=float))
    pred = np.exp(np.asarray(pred_log, dtype=float))
    return np.abs(pred - real) / real


def metricas_modelo(y_log: np.ndarray, pred_log: np.ndarray) -> dict:
    ape = error_relativo(y_log, pred_log)
    sesgo = np.exp(np.asarray(pred_log) - np.asarray(y_log)) - 1
    return {
        "MdAPE": float(np.median(ape)),
        "PPE10": float(np.mean(ape <= 0.10)),
        "PPE20": float(np.mean(ape <= 0.20)),
        # Sesgo mediano: si es positivo el modelo sobrevalúa de forma
        # sistemática, que en una cartera de compras es peor que errar parejo.
        "sesgo_mediano": float(np.median(sesgo)),
    }


def comparar(r: pd.DataFrame) -> pd.DataFrame:
    """Tabla de los tres modelos sobre las mismas operaciones."""
    filas = []
    for nombre, columna in MODELOS.items():
        m = metricas_modelo(r["y"].to_numpy(), r[columna].to_numpy())
        filas.append({"modelo": nombre, **m})
    tabla = pd.DataFrame(filas)
    # Cuánto mejora cada modelo sobre la regla que el negocio ya usa.
    base = tabla.loc[tabla["modelo"] == "mediana por ciudad y uso", "MdAPE"].iloc[0]
    tabla["mejora_vs_base"] = 1 - tabla["MdAPE"] / base
    return tabla


def metricas_intervalo(r: pd.DataFrame, alpha: float = 0.10) -> dict:
    lo, hi = r["lo"].to_numpy(), r["hi"].to_numpy()
    anchos = ancho_relativo(lo, hi)
    return {
        "cobertura_objetivo": 1 - alpha,
        "cobertura_observada": cobertura(r["y"].to_numpy(), lo, hi),
        "ancho_mediano": float(np.median(anchos)),
        "ancho_p90": float(np.quantile(anchos, 0.90)),
    }


def por_anio(r: pd.DataFrame, alpha: float = 0.10) -> pd.DataFrame:
    """Desglose año por año. Un promedio global puede esconder un año malo."""
    filas = []
    for anio, bloque in r.groupby("anio"):
        m = metricas_modelo(bloque["y"].to_numpy(), bloque["pred_boosting"].to_numpy())
        filas.append({
            "anio": int(anio),
            "n": len(bloque),
            "entrenado_hasta": int(bloque["entrenado_hasta"].iloc[0]),
            "MdAPE": m["MdAPE"],
            "PPE10": m["PPE10"],
            "cobertura": cobertura(bloque["y"].to_numpy(),
                                   bloque["lo"].to_numpy(), bloque["hi"].to_numpy()),
        })
    return pd.DataFrame(filas)


def por_segmento(r: pd.DataFrame, columna: str = "ciudad") -> pd.DataFrame:
    """Dónde falla. Un MdAPE global de 12% puede ser 8% en cinco ciudades y
    30% en la sexta, y esa sexta es la que hay que dejar de valuar a ciegas."""
    filas = []
    for valor, bloque in r.groupby(columna):
        m = metricas_modelo(bloque["y"].to_numpy(), bloque["pred_boosting"].to_numpy())
        filas.append({columna: valor, "n": len(bloque),
                      "MdAPE": m["MdAPE"], "PPE20": m["PPE20"]})
    return pd.DataFrame(filas).sort_values("MdAPE")


def efecto_de_deflactar(r_real: pd.DataFrame, r_nominal: pd.DataFrame) -> pd.DataFrame:
    """La comparación medida entre entrenar con pesos constantes y sin ellos.

    Ojo con cómo se lee: los dos modelos predicen cosas distintas, así que el
    MdAPE no es comparable entre ellos punto por punto. Lo que sí se compara, y
    es lo que importa, es **de dónde saca cada uno su poder predictivo** — eso
    está en la importancia por permutación, en `explicacion.py`.

    Aquí solo se deja el dato duro: cuánto del movimiento del precio nominal es
    inflación pura.
    """
    return pd.DataFrame([
        {"entrenamiento": "pesos constantes (deflactado)",
         **metricas_modelo(r_real["y"].to_numpy(), r_real["pred_boosting"].to_numpy())},
        {"entrenamiento": "pesos corrientes (nominal)",
         **metricas_modelo(r_nominal["y"].to_numpy(),
                           r_nominal["pred_boosting"].to_numpy())},
    ])
