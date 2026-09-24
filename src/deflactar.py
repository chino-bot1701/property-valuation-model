"""
Pesos constantes. La decisión que cambia el modelo entero.

Un precio de escritura es nominal: dice cuántos pesos cambiaron de mano ese
día. Si se entrena directo sobre eso, el modelo aprende su relación más fuerte
—que lo reciente es más caro— y esa relación **no es valor, es inflación**.

El resultado se ve bien en las métricas y es inútil en la práctica: no puedes
comparar un predio comprado en 2018 con uno de 2024, que es exactamente lo que
un comité de inversión necesita hacer.

La corrección es de una línea, pero hay que hacerla antes de todo lo demás:

    precio_real = precio_nominal × (INPC_hoy / INPC_de_la_operación)

Lo que queda después es apreciación **real**: lo que el suelo ganó por encima
de la inflación, que sí es una señal de mercado. En `docs/` está la comparación
medida de qué le pasa al modelo cuando este paso se omite.

El índice aquí es sintético, con la forma del INPC mexicano del periodo. En un
despliegue real se lee la serie publicada por el INEGI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .generar_mercado import ANIO_FINAL, INPC


def indice() -> pd.Series:
    """La serie de índice de precios, base 2018 = 100."""
    return pd.Series(INPC, name="inpc").rename_axis("anio")


def factor(anio, anio_base: int = ANIO_FINAL) -> np.ndarray:
    """Multiplicador que lleva un precio de `anio` a pesos de `anio_base`."""
    serie = indice()
    faltantes = set(pd.unique(pd.Series(anio))) - set(serie.index)
    if faltantes:
        raise KeyError(f"No hay índice para los años {sorted(faltantes)}")
    return (serie.loc[anio_base] / pd.Series(anio).map(serie)).to_numpy()


def a_pesos_constantes(d: pd.DataFrame, columna: str,
                       anio_base: int = ANIO_FINAL) -> pd.Series:
    """Convierte una columna de precios nominales a pesos de `anio_base`."""
    return pd.Series(d[columna].to_numpy() * factor(d["anio"].to_numpy(), anio_base),
                     index=d.index, name=f"{columna}_real")


def inflacion_acumulada(anio_desde: int, anio_hasta: int = ANIO_FINAL) -> float:
    """Cuánto subieron los precios entre dos años, en proporción."""
    serie = indice()
    return float(serie.loc[anio_hasta] / serie.loc[anio_desde] - 1.0)
