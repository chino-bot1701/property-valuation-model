"""
Generador del mercado sintético de suelo.

No hay un solo registro real detrás de este archivo. Las operaciones, los
precios, las ciudades y las características de cada predio se construyen con un
generador determinista.

Lo que sí es real es la **estructura** del precio del suelo, que es lo que el
modelo tiene que aprender:

- El precio por m² **baja** conforme el predio crece. Un lote de 50,000 m² no
  vale cincuenta veces uno de 1,000: hay menos compradores para el grande y el
  precio unitario se castiga. Si un modelo no captura esto, sobrevalúa todo lo
  grande.
- **La distancia al centro manda**, y no de forma lineal: el primer kilómetro
  cuesta mucho más que el décimo. Por eso entra en logaritmo.
- **Lo construible vale.** Dos predios idénticos con distinto coeficiente de
  uso de suelo (CUS) no valen lo mismo, porque lo que se compra es el derecho a
  edificar encima.
- El ruido **no es parejo**: los predios raros —muy grandes, muy alejados— se
  transan con mucha más dispersión que un lote comercial estándar. Esto importa
  para los intervalos: un modelo honesto debe abrir más el rango justo ahí.

Sobre eso se aplica la inflación, porque el precio que queda registrado en una
escritura es nominal, no real. Separar las dos cosas es el punto de partida del
repo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEMILLA = 20260924

N_OPERACIONES = 2_600
ANIO_INICIAL = 2018
ANIO_FINAL = 2025

# Ciudades ficticias del universo del portafolio. El nivel base es el precio
# por m² de referencia de cada plaza, en pesos constantes de 2025.
CIUDADES = {
    "Queretaro":      {"base": 9_200,  "apreciacion_real": 0.035},
    "Aguascalientes": {"base": 6_400,  "apreciacion_real": 0.021},
    "Leon":           {"base": 7_100,  "apreciacion_real": 0.018},
    "Merida":         {"base": 8_300,  "apreciacion_real": 0.047},
    "Puebla":         {"base": 6_900,  "apreciacion_real": 0.015},
    "Culiacan":       {"base": 5_200,  "apreciacion_real": 0.009},
}

# Peso de cada uso de suelo sobre el precio base, en logaritmo.
USOS = {
    "Comercial":    0.34,
    "Oficinas":     0.22,
    "Mixto":        0.10,
    "Habitacional": 0.00,
    "Industrial":  -0.26,
}

# Índice de precios, base 2018 = 100. Sintético, pero con la forma del INPC
# mexicano del periodo: inflación baja, el salto de 2022, y normalización.
INPC = {
    2018: 100.0, 2019: 103.6, 2020: 107.2, 2021: 114.9,
    2022: 123.9, 2023: 129.5, 2024: 135.2, 2025: 140.3,
}

# Superficie de referencia: los coeficientes de escala se miden contra esto.
SUPERFICIE_PIVOTE = 2_000.0


def factor_inflacion(anio: int | np.ndarray, anio_base: int = ANIO_FINAL):
    """Cuánto hay que multiplicar un precio de `anio` para llevarlo a `anio_base`."""
    serie = pd.Series(INPC)
    return serie.loc[anio_base] / pd.Series(anio).map(serie).to_numpy()


def _predios(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Las características físicas y de ubicación de cada predio."""
    ciudad = rng.choice(list(CIUDADES), size=n,
                        p=[0.22, 0.14, 0.17, 0.15, 0.19, 0.13])
    uso = rng.choice(list(USOS), size=n, p=[0.28, 0.12, 0.14, 0.30, 0.16])

    # Cola larga: muchos lotes medianos, pocos predios grandes.
    superficie = np.clip(np.exp(rng.normal(7.6, 1.05, n)), 250, 90_000)

    # El frente crece con la raíz de la superficie, con variación: un predio
    # puede ser cuadrado o una tira larga y angosta, y eso cambia su valor.
    frente = np.sqrt(superficie) * rng.uniform(0.55, 1.45, n)

    dist_centro = np.clip(rng.gamma(2.2, 3.1, n), 0.4, 28.0)

    # Cerca del centro hay más comercio. La correlación es real, no cosmética:
    # si el generador no la mete, el modelo aprende dos variables independientes
    # que en la vida real van juntas.
    densidad = np.clip(
        420 * np.exp(-dist_centro / 6.0) * rng.lognormal(0, 0.45, n), 0, 600)

    nivel_se = np.clip(
        np.round(rng.normal(3.4 - dist_centro * 0.045, 0.9, n)), 1, 5)

    dist_vialidad = np.clip(rng.gamma(1.7, 260, n), 15, 4_000)
    esquina = (rng.random(n) < 0.13).astype(int)

    # Coeficiente de uso de suelo: cuántos m² se pueden construir por m² de
    # terreno. Depende del uso permitido y de qué tan central sea el predio.
    cus_base = {"Comercial": 2.6, "Oficinas": 3.4, "Mixto": 2.2,
                "Habitacional": 1.5, "Industrial": 1.1}
    cus = np.clip(
        np.array([cus_base[u] for u in uso])
        * np.exp(-dist_centro / 22.0)
        * rng.lognormal(0, 0.30, n),
        0.4, 7.0)

    return pd.DataFrame({
        "ciudad": ciudad,
        "uso_suelo": uso,
        "superficie_m2": np.round(superficie, 1),
        "frente_m": np.round(frente, 1),
        "dist_centro_km": np.round(dist_centro, 2),
        "dist_vialidad_m": np.round(dist_vialidad, 0),
        "densidad_comercios_1km": np.round(densidad, 0),
        "nivel_socioeconomico": nivel_se.astype(int),
        "esquina": esquina,
        "cus": np.round(cus, 2),
    })


def _log_precio_real(d: pd.DataFrame, anios_transcurridos: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """El precio por m² en pesos constantes, en logaritmo.

    Esta función es la "verdad" que el modelo intentará recuperar. Está escrita
    en logaritmos porque los efectos del mercado son multiplicativos: una
    esquina no suma $800, agrega un porcentaje.
    """
    uso = d["uso_suelo"]
    base = np.log([CIUDADES[c]["base"] for c in d["ciudad"]])
    apreciacion = np.array([CIUDADES[c]["apreciacion_real"] for c in d["ciudad"]])

    # --- Interacciones. Son la razón de que un modelo de árboles tenga algo
    # que hacer aquí: los efectos no son los mismos en todas las categorías,
    # y un modelo lineal necesitaría que alguien escribiera cada término.

    # Lo construible vale mucho donde se puede explotar y casi nada en suelo
    # industrial, donde lo que se compra es superficie, no altura.
    peso_cus = uso.map({"Comercial": 0.44, "Oficinas": 0.52, "Mixto": 0.31,
                        "Habitacional": 0.17, "Industrial": 0.04}).to_numpy()

    # Cada ciudad decae distinto: unas son compactas y el centro lo es todo,
    # otras están extendidas y alejarse cuesta poco.
    decaimiento = d["ciudad"].map(
        {"Queretaro": -0.40, "Merida": -0.20, "Leon": -0.34,
         "Aguascalientes": -0.31, "Puebla": -0.37, "Culiacan": -0.19}).to_numpy()

    # Una esquina es un activo comercial: da dos frentes y visibilidad. En
    # suelo industrial o habitacional apenas cambia el precio.
    peso_esquina = uso.map({"Comercial": 0.24, "Oficinas": 0.17, "Mixto": 0.13,
                            "Habitacional": 0.04, "Industrial": 0.02}).to_numpy()

    # Tener comercio alrededor sube un local y le es indiferente a una nave.
    peso_densidad = uso.map({"Comercial": 0.30, "Oficinas": 0.21, "Mixto": 0.17,
                             "Habitacional": 0.11, "Industrial": 0.01}).to_numpy()

    log_p = (
        base
        + uso.map(USOS).to_numpy()
        + peso_cus * np.log1p(d["cus"].to_numpy())
        + decaimiento * np.log(d["dist_centro_km"].to_numpy())
        - 0.07 * np.log1p(d["dist_vialidad_m"].to_numpy() / 100.0)
        + peso_densidad * np.log1p(d["densidad_comercios_1km"].to_numpy() / 50.0)
        + 0.10 * (d["nivel_socioeconomico"].to_numpy() - 3)
        + peso_esquina * d["esquina"].to_numpy()
        # Economía de escala: el precio unitario cae con el tamaño del lote.
        - 0.11 * (np.log(d["superficie_m2"].to_numpy()) - np.log(SUPERFICIE_PIVOTE))
        # Un frente amplio para la superficie que tiene vale más.
        + 0.06 * np.log(d["frente_m"].to_numpy()
                        / np.sqrt(d["superficie_m2"].to_numpy()))
        # Apreciación REAL, la que queda después de descontar inflación.
        + apreciacion * anios_transcurridos
    )

    # Ruido heterocedástico: los predios grandes y los alejados se transan con
    # mucha más dispersión —hay menos comparables y menos compradores—. Es la
    # razón de que un intervalo de ancho fijo no sirva y haya que estimarlo
    # predio por predio.
    sigma = (0.095
             + 0.035 * np.clip(np.log(d["superficie_m2"].to_numpy() / 2_000), 0, None)
             + 0.007 * d["dist_centro_km"].to_numpy())
    return log_p + rng.normal(0, sigma)


def generar_operaciones(semilla: int = SEMILLA,
                        n: int = N_OPERACIONES) -> pd.DataFrame:
    """El comparable de mercado: operaciones cerradas, con precio de escritura.

    El precio que sale es **nominal**, como en la realidad: lo que se pagó el
    día de la operación, sin corregir por inflación. Deflactarlo es trabajo del
    pipeline, no del generador.
    """
    rng = np.random.default_rng(semilla)
    d = _predios(rng, n)

    anio = rng.integers(ANIO_INICIAL, ANIO_FINAL + 1, n)
    mes = rng.integers(1, 13, n)
    d["fecha"] = pd.to_datetime(
        {"year": anio, "month": mes, "day": rng.integers(1, 28, n)})
    d["anio"] = anio
    d["anios_desde_inicio"] = anio - ANIO_INICIAL

    log_real = _log_precio_real(d, d["anios_desde_inicio"].to_numpy(), rng)

    # Un 3% son operaciones atípicas: remates, ventas entre partes
    # relacionadas, urgencias. No se marcan ni se quitan — en un padrón real
    # nadie las trae etiquetadas, y el modelo tiene que convivir con ellas.
    atipica = rng.random(n) < 0.03
    log_real = log_real + np.where(atipica, rng.normal(-0.45, 0.25, n), 0.0)

    precio_m2_real = np.exp(log_real)
    inflacion = factor_inflacion(d["anio"].to_numpy())
    d["precio_m2_nominal"] = np.round(precio_m2_real / inflacion, 2)
    d["precio_total_nominal"] = np.round(
        d["precio_m2_nominal"] * d["superficie_m2"], 2)

    d = d.sort_values("fecha").reset_index(drop=True)
    d.insert(0, "operacion_id", [f"OP-{i:05d}" for i in range(1, len(d) + 1)])
    return d


def generar_portafolio(semilla: int = SEMILLA + 1, n: int = 64) -> pd.DataFrame:
    """Los predios que el grupo ya compró, con lo que pagó por ellos.

    Estos no sirven para entrenar: son sobre los que se aplica el modelo para
    ver cuáles se compraron caros y cuáles baratos. Una parte se compró
    deliberadamente fuera de precio, que es justo lo que hay que detectar.
    """
    rng = np.random.default_rng(semilla)
    d = _predios(rng, n)

    anio = rng.integers(ANIO_INICIAL, ANIO_FINAL + 1, n)
    d["fecha_adquisicion"] = pd.to_datetime(
        {"year": anio, "month": rng.integers(1, 13, n),
         "day": rng.integers(1, 28, n)})
    d["anio"] = anio
    d["anios_desde_inicio"] = anio - ANIO_INICIAL

    log_real = _log_precio_real(d, d["anios_desde_inicio"].to_numpy(), rng)

    # 12 predios comprados fuera de mercado: seis caros, seis baratos.
    desvio = np.zeros(n)
    idx = rng.permutation(n)
    desvio[idx[:6]] = rng.uniform(0.30, 0.48, 6)      # se pagó de más
    desvio[idx[6:12]] = -rng.uniform(0.30, 0.48, 6)   # se compró bien
    log_real = log_real + desvio
    # Etiqueta oculta, solo para evaluar. En una cartera real nadie trae
    # marcado cuál se compró fuera de precio: ese es justo el trabajo.
    d["_desvio_inyectado"] = np.round(desvio, 4)
    d["_fuera_de_precio"] = (desvio != 0).astype(int)

    inflacion = factor_inflacion(d["anio"].to_numpy())
    d["precio_m2_pagado"] = np.round(np.exp(log_real) / inflacion, 2)
    d["precio_total_pagado"] = np.round(
        d["precio_m2_pagado"] * d["superficie_m2"], 2)

    empresas = ["AAI", "ADP", "APE", "AYP", "IBS", "ILP",
                "EDC", "EDP", "BSM", "DIA", "OCI", "PPO"]
    d["empresa"] = rng.choice(empresas, n)
    d = d.sort_values("fecha_adquisicion").reset_index(drop=True)
    d.insert(0, "predio_id", [f"PR-{i:04d}" for i in range(1, len(d) + 1)])
    d.insert(1, "nombre", [f"Predio {n_}" for n_ in range(1, len(d) + 1)])
    return d


if __name__ == "__main__":
    ops = generar_operaciones()
    port = generar_portafolio()
    ops.to_csv("data/synthetic/operaciones.csv", index=False)
    port.to_csv("data/synthetic/portafolio.csv", index=False)
    print(f"{len(ops):,} operaciones {ops['anio'].min()}-{ops['anio'].max()} "
          f"en {ops['ciudad'].nunique()} ciudades")
    print(f"precio/m2 nominal: mediana ${ops['precio_m2_nominal'].median():,.0f}, "
          f"rango ${ops['precio_m2_nominal'].min():,.0f}-"
          f"${ops['precio_m2_nominal'].max():,.0f}")
    print(f"{len(port)} predios en el portafolio del grupo")
