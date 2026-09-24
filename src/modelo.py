"""
Tres modelos, en orden de ambición, y una comparación honesta entre ellos.

El orden no es decorativo. Un modelo sin línea base no se puede juzgar: un R²
de 0.85 puede ser excelente o puede ser peor que tomar la mediana de la
colonia. Así que primero se construye la regla que el negocio ya usa —la
mediana por ciudad y uso, que es literalmente lo que hace un valuador con una
tabla— y todo lo demás tiene que ganarle.

    1. Mediana por ciudad y uso   la regla de mesa, sin modelo
    2. Ridge sobre la matriz      lineal, interpretable, coeficientes que se leen
    3. Gradient boosting          captura interacciones que la lineal no ve

La lineal no está de relleno. Si el boosting apenas le gana, la respuesta
correcta es quedarse con la lineal: se explica ante un comité, se audita, y no
se rompe cuando llega un predio distinto a todo lo visto.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import COLUMNAS, matriz

SEMILLA = 20260924


class LineaBaseMediana:
    """La regla de mesa: la mediana del precio por m² de esa ciudad y ese uso.

    Es lo que ya se hace sin modelo, y por eso es la vara. Cae de vuelta a la
    mediana de la ciudad, y luego a la global, cuando una combinación no tiene
    comparables suficientes — que es el punto débil real de trabajar con tablas.
    """

    MIN_COMPARABLES = 8

    def __init__(self) -> None:
        self.por_grupo: pd.Series | None = None
        self.por_ciudad: pd.Series | None = None
        self.global_: float = 0.0

    def entrenar(self, d: pd.DataFrame, y: pd.Series) -> "LineaBaseMediana":
        tmp = d[["ciudad", "uso_suelo"]].copy()
        tmp["y"] = y.to_numpy()
        grupos = tmp.groupby(["ciudad", "uso_suelo"])["y"]
        self.por_grupo = grupos.median()[grupos.size() >= self.MIN_COMPARABLES]
        self.por_ciudad = tmp.groupby("ciudad")["y"].median()
        self.global_ = float(tmp["y"].median())
        return self

    def predecir(self, d: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([d["ciudad"], d["uso_suelo"]])
        p = pd.Series(self.por_grupo.reindex(idx).to_numpy(), index=d.index)
        p = p.fillna(pd.Series(self.por_ciudad.reindex(d["ciudad"]).to_numpy(),
                               index=d.index))
        return p.fillna(self.global_).to_numpy()


@dataclass
class ModeloLineal:
    """Ridge sobre la matriz de diseño.

    Se estandariza porque Ridge penaliza el tamaño de los coeficientes, y sin
    poner las variables en la misma escala la penalización castiga a las que
    vienen en unidades grandes por el simple hecho de venir en unidades
    grandes. `alpha` es pequeño: hay 20 columnas y 2,000 filas, no hace falta
    apretar mucho.
    """

    alpha: float = 1.0
    pipeline: Pipeline | None = field(default=None, repr=False)

    def entrenar(self, d: pd.DataFrame, y: pd.Series) -> "ModeloLineal":
        self.pipeline = Pipeline([
            ("escala", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha)),
        ]).fit(matriz(d), y)
        return self

    def predecir(self, d: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(matriz(d))

    @property
    def estimador(self):
        """El objeto de scikit-learn, para quien necesite validación cruzada."""
        return self.pipeline

    def coeficientes(self) -> pd.Series:
        """En escala logarítmica y estandarizada: cada coeficiente es el cambio
        porcentual aproximado del precio ante un cambio de una desviación
        estándar en esa variable. Se leen directo."""
        return (pd.Series(self.pipeline.named_steps["ridge"].coef_, index=COLUMNAS)
                .sort_values(key=abs, ascending=False))


@dataclass
class ModeloBoosting:
    """Gradient boosting por histogramas.

    Aquí sí importan las interacciones: la distancia al centro no pesa igual en
    Querétaro que en Culiacán, y el CUS no vale lo mismo en suelo industrial que
    en comercial. Un modelo lineal necesitaría que alguien escribiera esos
    términos a mano; este los encuentra.
    """

    semilla: int = SEMILLA
    modelo: HistGradientBoostingRegressor | None = field(default=None, repr=False)

    def entrenar(self, d: pd.DataFrame, y: pd.Series) -> "ModeloBoosting":
        self.modelo = HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.06,
            max_leaf_nodes=24,
            min_samples_leaf=25,
            l2_regularization=0.5,
            early_stopping=True,
            validation_fraction=0.12,
            random_state=self.semilla,
        ).fit(matriz(d), y)
        return self

    def predecir(self, d: pd.DataFrame) -> np.ndarray:
        return self.modelo.predict(matriz(d))

    @property
    def estimador(self):
        return self.modelo


def a_pesos(log_precio: np.ndarray) -> np.ndarray:
    """Devuelve el logaritmo a pesos por m².

    `exp` de una predicción en log da la **mediana** condicional, no la media.
    Para una valuación eso es lo correcto —se quiere el precio típico, no el
    promedio inflado por la cola— y es coherente con que las métricas de este
    repo sean medianas de error, no promedios.
    """
    return np.exp(log_precio)
