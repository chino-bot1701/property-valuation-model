"""
Intervalos de predicción. Una valuación no es un número.

Decirle a un comité de inversión que un predio "vale $8,400 el m²" es inútil y
un poco deshonesto. La pregunta que sigue es *¿qué tan seguro?*, y la
diferencia entre ±8% y ±40% cambia la decisión por completo.

El método es **predicción conforme normalizada**. Tres piezas:

1. **Un estimador del punto** — el modelo de valuación, cualquiera que sea.
2. **Un estimador de la incertidumbre local** `σ̂(x)`: un segundo modelo,
   entrenado sobre el tamaño del error del primero, que aprende *dónde* se
   equivoca más. Un predio industrial de 60,000 m² a 20 km del centro tiene
   muchos menos comparables que un local céntrico, y el intervalo tiene que
   reflejarlo. Sin esta pieza el intervalo sería una banda de ancho fijo:
   demasiado ancha donde el modelo sabe, demasiado angosta donde no.
3. **Una calibración conforme** sobre datos que ninguno de los dos vio, que
   ajusta la escala hasta que la cobertura prometida se cumpla de verdad.

    intervalo(x) = ŷ(x) ± q̂ · σ̂(x)

Bajo intercambiabilidad esto trae una garantía de cobertura en muestra finita,
sin suponer nada sobre la distribución de los errores.

**Y aquí va la parte honesta: con datos temporales la intercambiabilidad no se
cumple.** El mercado de 2025 no es una permutación del de 2019. La garantía
teórica no aplica tal cual, así que este repo no la cita y ya: mide la
cobertura empírica año por año sobre la validación temporal. Si sale por debajo
de lo prometido, ese es el resultado y se reporta.

*Sobre la alternativa:* el otro camino habitual es CQR —conformalizar dos
modelos cuantílicos— y es mejor cuando hay datos de sobra, porque estima la
forma completa de la incertidumbre en vez de solo su escala. Aquí se probó y
salió perdiendo: los cuantiles extremos necesitan más observaciones de las que
deja un mercado de suelo por ciudad y por año, quedaban mal estimados, y la
corrección conforme tenía que ensanchar el intervalo hasta volverlo inservible.
Normalizar la escala pide mucho menos a los datos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict

from .features import matriz

SEMILLA = 20260924

# Piso para σ̂: sin él, un predio donde el modelo de error predice casi cero
# tendría un intervalo de ancho nulo, que es una afirmación que ningún modelo
# puede sostener.
PISO_SIGMA = 0.02


@dataclass
class IntervaloConforme:
    """Intervalo de predicción adaptativo y calibrado.

    `alpha` es la tasa de error que se acepta: 0.10 pide un intervalo del 90%.
    """

    alpha: float = 0.10
    semilla: int = SEMILLA
    _sigma: HistGradientBoostingRegressor | None = field(default=None, repr=False)
    q: float = field(default=0.0)

    def entrenar(self, modelo_punto, d_entrena: pd.DataFrame, y_entrena: pd.Series,
                 d_calibra: pd.DataFrame, y_calibra: pd.Series) -> "IntervaloConforme":
        """Ajusta σ̂ con un conjunto y calibra la escala con otro.

        Los dos tienen que ser disjuntos. Calibrar sobre los mismos datos con
        los que se ajustó da una corrección optimista —el modelo ya vio esos
        puntos— y el intervalo sale más angosto de lo que debe.

        Qué tan grave sea depende del modelo, y está medido en
        `tests/test_pipeline.py`: con la regresión lineal el efecto es
        despreciable, porque apenas sobreajusta. Con el gradient boosting el
        intervalo se encoge un 40% y la cobertura cae de 86% a 63% mientras
        sigue anunciando 90%. Nada truena; simplemente el número deja de ser
        cierto.
        """
        X_e = matriz(d_entrena)
        y_e = np.asarray(y_entrena, dtype=float)

        # Los residuos con los que se entrena σ̂ salen de validación cruzada,
        # no del ajuste. Los residuos en muestra de un modelo ya entrenado son
        # sistemáticamente más chicos que los que cometerá con datos nuevos, y
        # σ̂ heredaría ese optimismo.
        pred_fuera = cross_val_predict(modelo_punto.estimador, X_e, y_e, cv=5)
        residuo = np.abs(y_e - pred_fuera)

        # Se modela el logaritmo del error: obliga a que σ̂ sea positiva y hace
        # que el objetivo sea el error *relativo*, que es como se comporta.
        self._sigma = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.07, max_leaf_nodes=12,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            random_state=self.semilla,
        ).fit(X_e, np.log(residuo + PISO_SIGMA))

        # Calibración: qué tan grande es el error, medido en unidades de σ̂.
        y_c = np.asarray(y_calibra, dtype=float)
        pred_c = modelo_punto.predecir(d_calibra)
        puntajes = np.abs(y_c - pred_c) / self.escala(d_calibra)

        # El ajuste (n+1)/n es lo que hace que la garantía valga en muestra
        # finita y no solo asintóticamente.
        n = len(puntajes)
        nivel = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q = float(np.quantile(puntajes, nivel, method="higher"))
        return self

    def escala(self, d: pd.DataFrame) -> np.ndarray:
        """σ̂(x): qué tan incierto es este predio en particular."""
        if self._sigma is None:
            raise RuntimeError("El intervalo no ha sido calibrado.")
        return np.maximum(np.exp(self._sigma.predict(matriz(d))), PISO_SIGMA)

    def intervalo(self, modelo_punto, d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (límite inferior, límite superior) en escala logarítmica."""
        centro = modelo_punto.predecir(d)
        margen = self.q * self.escala(d)
        return centro - margen, centro + margen


def cobertura(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Qué proporción de los casos reales cayó dentro del intervalo."""
    y = np.asarray(y, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))


def ancho_relativo(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Ancho del intervalo como proporción del punto central.

    Se calcula en pesos, no en logaritmos: un intervalo de ±0.3 en log suena
    inocuo y en realidad va de −26% a +35%. El número que se le enseña a
    alguien tiene que estar en la escala en la que decide.
    """
    centro = np.exp((lo + hi) / 2)
    return (np.exp(hi) - np.exp(lo)) / centro
