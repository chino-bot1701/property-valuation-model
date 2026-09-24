"""
De dónde saca el modelo lo que sabe.

Dos herramientas, y las dos responden preguntas que en una entrevista o ante un
comité se hacen textualmente.

**Importancia por permutación** — "¿qué variables está usando de verdad?". Se
revuelve una columna al azar, se vuelve a medir el error, y lo que empeora es
lo que esa variable aportaba. Se prefiere sobre la importancia interna de los
árboles porque esa última infla a las variables con muchos valores distintos
—una continua siempre ofrece más puntos de corte que una bandera de 0/1— y se
mide sobre el entrenamiento, no sobre datos nuevos.

**Dependencia parcial** — "¿y cómo la usa?". Barre una variable por todo su
rango dejando las demás como están, y dibuja qué le pasa al precio. Es lo que
convierte "la distancia al centro es importante" en "el primer kilómetro vale
cuatro veces lo que el décimo", que es una frase que un valuador puede
contradecir o confirmar con su experiencia — y que se pueda discutir es
precisamente el punto.

La función `comparar_deflactado` es la que produce el resultado central del
repo: entrena el mismo modelo dos veces, con y sin corregir la inflación, y
enseña cómo cambia de qué depende.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .features import COLUMNAS, matriz, objetivo
from .modelo import ModeloBoosting

SEMILLA = 20260924

# Cómo se lee cada columna en español.
NOMBRES = {
    "log_superficie": "superficie del predio",
    "log_forma": "forma del lote (frente vs fondo)",
    "log_dist_centro": "distancia al centro urbano",
    "log_dist_vialidad": "distancia a vialidad primaria",
    "log_densidad": "comercios en 1 km",
    "nivel_socioeconomico": "nivel socioeconómico de la zona",
    "esquina": "es esquina",
    "cus": "coeficiente de uso de suelo",
    "anios_desde_inicio": "año de la operación",
}
NOMBRES.update({f"ciudad_{c}": f"ciudad: {c}" for c in
                ["Aguascalientes", "Culiacan", "Leon", "Merida", "Puebla", "Queretaro"]})
NOMBRES.update({f"uso_{u}": f"uso: {u}" for u in
                ["Comercial", "Habitacional", "Industrial", "Mixto", "Oficinas"]})


def importancia(modelo: ModeloBoosting, d: pd.DataFrame, y: pd.Series,
                repeticiones: int = 8, semilla: int = SEMILLA) -> pd.DataFrame:
    """Importancia por permutación, medida sobre datos que el modelo no vio."""
    r = permutation_importance(
        modelo.estimador, matriz(d), y,
        n_repeats=repeticiones, random_state=semilla,
        scoring="neg_mean_absolute_error", n_jobs=-1)
    return (pd.DataFrame({
        "columna": COLUMNAS,
        "variable": [NOMBRES.get(c, c) for c in COLUMNAS],
        "importancia": r.importances_mean,
        "desv": r.importances_std,
    })
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True))


def comparar_deflactado(d: pd.DataFrame, semilla: int = SEMILLA) -> pd.DataFrame:
    """El experimento central: el mismo modelo, con y sin deflactar.

    Entrena las dos versiones sobre los mismos predios y mide la importancia de
    cada variable. Lo que hay que mirar no es el error —los dos modelos
    predicen escalas distintas y no son comparables punto por punto— sino
    **el lugar que ocupa "año de la operación"** en cada lista.

    Sobre precios nominales sube al podio: el modelo está prediciendo
    inflación. Deflactado se desploma, y lo que queda arriba es ubicación, que
    es lo que de verdad determina el valor del suelo.

    **Aquí, y solo aquí, la partición es aleatoria y no temporal.** No es un
    descuido ni una inconsistencia con el resto del repo: es un requisito del
    método. La importancia por permutación revuelve una columna y mide cuánto
    empeora el error; si el conjunto de evaluación es un solo año, la columna
    "año" es constante, revolverla no cambia nada, y sale con importancia cero
    aunque el modelo dependa por completo de ella. Para que la pregunta
    *¿de qué depende este modelo?* tenga respuesta, el año tiene que variar en
    los datos sobre los que se mide.

    La validación temporal sigue siendo la que mide qué tan bien predice
    (`validacion_temporal.py`). Esto mide otra cosa: de dónde saca lo que sabe.
    """
    rng = np.random.default_rng(semilla)
    mezcla = rng.permutation(len(d))
    corte = int(len(d) * 0.75)
    entrena = d.iloc[mezcla[:corte]]
    prueba = d.iloc[mezcla[corte:]]

    filas = []
    for etiqueta, deflactar in [("deflactado", True), ("nominal", False)]:
        y_e = objetivo(entrena, deflactar=deflactar)
        y_p = objetivo(prueba, deflactar=deflactar)
        modelo = ModeloBoosting(semilla=semilla).entrenar(entrena, y_e)
        imp = importancia(modelo, prueba, y_p, semilla=semilla)
        imp["entrenamiento"] = etiqueta
        imp["lugar"] = np.arange(1, len(imp) + 1)
        # Normalizada, para que las dos listas se puedan poner lado a lado.
        imp["peso_relativo"] = imp["importancia"] / imp["importancia"].max()
        filas.append(imp)

    return pd.concat(filas, ignore_index=True)


def dependencia_parcial(modelo: ModeloBoosting, d: pd.DataFrame,
                        columna: str, valores: np.ndarray) -> pd.DataFrame:
    """Cómo se mueve el precio cuando una variable barre su rango.

    Se recalcula sobre todo el padrón para cada valor y se promedia, en vez de
    usar un predio "promedio" que probablemente no existe.
    """
    X = matriz(d)
    precios = []
    for v in valores:
        X_mod = X.copy()
        X_mod[columna] = v
        precios.append(float(np.exp(modelo.estimador.predict(X_mod)).mean()))
    return pd.DataFrame({columna: valores, "precio_m2": precios})


def resumen_de_una_valuacion(fila: pd.Series, pred_log: float,
                             lo: float, hi: float) -> str:
    """Una línea en español para acompañar cada valuación."""
    centro = np.exp(pred_log)
    ancho = (np.exp(hi) - np.exp(lo)) / centro
    return (f"${centro:,.0f}/m² para un predio {fila['uso_suelo'].lower()} de "
            f"{fila['superficie_m2']:,.0f} m² en {fila['ciudad']}, a "
            f"{fila['dist_centro_km']:.1f} km del centro. "
            f"Rango del 90%: ${np.exp(lo):,.0f} a ${np.exp(hi):,.0f} "
            f"(±{ancho / 2:.0%}).")
