"""
La matriz de diseño, y por qué se ve así.

Tres decisiones gobiernan este archivo.

**1. Se modela el precio por m², no el precio total.** El precio total lo
domina la superficie: un modelo entrenado sobre él acierta el 95% de la
varianza sin haber aprendido nada de valuación, solo a multiplicar por metros.
El precio unitario es lo que un valuador discute, y es donde está la señal.

**2. Casi todo entra en logaritmo — la variable objetivo también.** El mercado
de suelo es multiplicativo: una esquina no suma $800 por m², agrega un
porcentaje; el segundo kilómetro de distancia al centro no cuesta lo mismo que
el décimo. En escala logarítmica esos efectos son lineales y estables, y el
error que minimiza el modelo pasa a ser relativo en vez de absoluto — que es el
error que importa cuando los predios van de $450 a $35,000 el m².

**3. El año entra, pero solo después de deflactar.** Sobre precios nominales,
el año es la variable más predictiva del conjunto y el modelo se convierte en
una calculadora de inflación. Ya descontada la inflación, lo que el año captura
es apreciación real, que sí es información de mercado. Es la misma columna con
dos significados completamente distintos, y la diferencia la hace un paso
previo de dos líneas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .deflactar import a_pesos_constantes

CIUDADES = ["Aguascalientes", "Culiacan", "Leon", "Merida", "Puebla", "Queretaro"]
USOS = ["Comercial", "Habitacional", "Industrial", "Mixto", "Oficinas"]

NUMERICAS = [
    "log_superficie",
    "log_forma",
    "log_dist_centro",
    "log_dist_vialidad",
    "log_densidad",
    "nivel_socioeconomico",
    "esquina",
    "cus",
    "anios_desde_inicio",
]

COLUMNAS = NUMERICAS + [f"ciudad_{c}" for c in CIUDADES] + [f"uso_{u}" for u in USOS]

# Lo que jamás entra al modelo: nada que identifique al dueño ni a la
# operación. El valor de un predio no depende de quién lo compró.
PROHIBIDAS = ["empresa", "nombre", "predio_id", "operacion_id", "propietario"]


def matriz(d: pd.DataFrame) -> pd.DataFrame:
    """Construye la matriz de diseño a partir del padrón crudo."""
    X = pd.DataFrame(index=d.index)

    X["log_superficie"] = np.log(d["superficie_m2"])
    # Forma del lote: qué tan ancho es su frente para la superficie que tiene.
    # Un predio cuadrado y una tira angosta de los mismos metros no valen igual.
    X["log_forma"] = np.log(d["frente_m"] / np.sqrt(d["superficie_m2"]))
    X["log_dist_centro"] = np.log(d["dist_centro_km"])
    X["log_dist_vialidad"] = np.log1p(d["dist_vialidad_m"] / 100.0)
    X["log_densidad"] = np.log1p(d["densidad_comercios_1km"] / 50.0)
    X["nivel_socioeconomico"] = d["nivel_socioeconomico"].astype(float)
    X["esquina"] = d["esquina"].astype(float)
    X["cus"] = d["cus"].astype(float)
    X["anios_desde_inicio"] = d["anios_desde_inicio"].astype(float)

    # One-hot con las categorías fijadas de antemano: si un lote de datos no
    # trae alguna ciudad, la columna se crea igual en cero. Sin esto, la matriz
    # cambia de forma entre entrenamiento y predicción y el modelo truena en
    # producción por un caso que no estaba en el mes de entrenamiento.
    for c in CIUDADES:
        X[f"ciudad_{c}"] = (d["ciudad"] == c).astype(float)
    for u in USOS:
        X[f"uso_{u}"] = (d["uso_suelo"] == u).astype(float)

    verificar(X)
    return X[COLUMNAS]


def objetivo(d: pd.DataFrame, columna: str = "precio_m2_nominal",
             deflactar: bool = True) -> pd.Series:
    """El log del precio por m². Con o sin deflactar, para poder comparar."""
    precio = a_pesos_constantes(d, columna) if deflactar else d[columna]
    return pd.Series(np.log(np.asarray(precio, dtype=float)),
                     index=d.index, name="log_precio_m2")


def verificar(X: pd.DataFrame) -> None:
    """Falla antes de entrenar si la matriz trae algo que no debe."""
    intrusas = [c for c in X.columns if c.lower() in PROHIBIDAS]
    if intrusas:
        raise ValueError(f"La matriz contiene columnas del propietario: {intrusas}")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        malas = [c for c in X.columns if not np.isfinite(X[c].to_numpy()).all()]
        raise ValueError(f"Valores no finitos en {malas}")
