"""
Pruebas del pipeline de valuación.

No prueban que el modelo sea bueno —para eso está `scripts/run_demo.py` con sus
aserciones—. Prueban las invariantes que, si se rompen, hacen que los buenos
números sean mentira:

- que los datos sean reproducibles,
- que **deflactar funcione**, porque de ahí cuelga todo lo demás,
- que la matriz no dependa de lo que traiga el lote de datos,
- que **cada modelo se entrene solo con años anteriores** al que valúa,
- que la calibración del intervalo **no toque** los datos de ajuste,
- que la cobertura prometida se cumpla.

La penúltima es la más fácil de romper sin darse cuenta: si el conjunto de
calibración se mezcla con el de entrenamiento, el intervalo sale angosto, las
métricas se ven mejor, y nada truena.

    pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import deflactar, evaluacion, features  # noqa: E402
from src.conformal import IntervaloConforme, cobertura  # noqa: E402
from src.explicacion import comparar_deflactado  # noqa: E402
from src.generar_mercado import (ANIO_FINAL, ANIO_INICIAL,  # noqa: E402
                                 generar_operaciones, generar_portafolio)
from src.modelo import LineaBaseMediana, ModeloBoosting, ModeloLineal  # noqa: E402
from src.portafolio import analizar  # noqa: E402
from src.validacion_temporal import entrenar_final, validar  # noqa: E402


@pytest.fixture(scope="module")
def ops() -> pd.DataFrame:
    return generar_operaciones()


@pytest.fixture(scope="module")
def resultados(ops) -> pd.DataFrame:
    return validar(ops, alpha=0.10)


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------

def test_el_generador_es_determinista():
    pd.testing.assert_frame_equal(generar_operaciones(n=300),
                                  generar_operaciones(n=300))


def test_el_mercado_cubre_todos_los_anios_y_ciudades(ops):
    assert set(ops["anio"]) == set(range(ANIO_INICIAL, ANIO_FINAL + 1))
    assert ops["ciudad"].nunique() == 6
    assert ops["uso_suelo"].nunique() == 5


def test_el_precio_por_m2_baja_con_el_tamano(ops):
    """La economía de escala del suelo. Si esto se invierte, el generador está
    mal y el modelo aprenderá algo que no pasa en el mercado."""
    real = deflactar.a_pesos_constantes(ops, "precio_m2_nominal")
    r = np.corrcoef(np.log(ops["superficie_m2"]), np.log(real))[0, 1]
    assert r < -0.05


def test_la_cartera_trae_compras_fuera_de_precio():
    port = generar_portafolio()
    assert port["_fuera_de_precio"].sum() == 12
    assert (port["_desvio_inyectado"] > 0).sum() == 6
    assert (port["_desvio_inyectado"] < 0).sum() == 6


# --------------------------------------------------------------------------
# Deflactar
# --------------------------------------------------------------------------

def test_deflactar_deja_el_ultimo_anio_intacto(ops):
    ultimo = ops[ops["anio"] == ANIO_FINAL]
    real = deflactar.a_pesos_constantes(ultimo, "precio_m2_nominal")
    np.testing.assert_allclose(real.to_numpy(),
                               ultimo["precio_m2_nominal"].to_numpy())


def test_deflactar_sube_los_precios_viejos(ops):
    viejo = ops[ops["anio"] == ANIO_INICIAL]
    real = deflactar.a_pesos_constantes(viejo, "precio_m2_nominal")
    assert (real.to_numpy() > viejo["precio_m2_nominal"].to_numpy()).all()


def test_el_indice_no_inventa_anios():
    with pytest.raises(KeyError):
        deflactar.factor(np.array([1999]))


def test_deflactar_quita_casi_toda_la_tendencia(ops):
    """La prueba de que el paso sirve: la correlación entre año y precio cae."""
    nominal = np.corrcoef(ops["anio"], np.log(ops["precio_m2_nominal"]))[0, 1]
    real = np.corrcoef(
        ops["anio"],
        np.log(deflactar.a_pesos_constantes(ops, "precio_m2_nominal")))[0, 1]
    assert nominal > 0.15
    assert abs(real) < nominal / 2


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def test_la_matriz_tiene_forma_fija_aunque_falten_categorias(ops):
    """Un lote de una sola ciudad debe producir la misma matriz.

    Sin esto, el modelo entrenado con seis ciudades truena al valuar un predio
    de una sola — que es exactamente lo que pasa en producción.
    """
    completa = features.matriz(ops)
    una_ciudad = features.matriz(ops[ops["ciudad"] == "Merida"])
    assert list(completa.columns) == list(una_ciudad.columns) == features.COLUMNAS


def test_la_matriz_rechaza_columnas_del_propietario(ops):
    X = features.matriz(ops.head(50))
    X["empresa"] = "AAI"
    with pytest.raises(ValueError, match="propietario"):
        features.verificar(X)


def test_la_matriz_rechaza_valores_no_finitos(ops):
    X = features.matriz(ops.head(50))
    X.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="no finitos"):
        features.verificar(X)


def test_el_objetivo_cambia_al_deflactar(ops):
    con = features.objetivo(ops, deflactar=True)
    sin = features.objetivo(ops, deflactar=False)
    assert not np.allclose(con, sin)
    # En el último año son idénticos: no hay nada que deflactar.
    ultimo = ops["anio"] == ANIO_FINAL
    np.testing.assert_allclose(con[ultimo].to_numpy(), sin[ultimo].to_numpy())


# --------------------------------------------------------------------------
# Validación temporal
# --------------------------------------------------------------------------

def test_cada_modelo_se_entrena_solo_con_el_pasado(resultados):
    assert (resultados["entrenado_hasta"] < resultados["anio"]).all()
    assert resultados["anio"].nunique() >= 4


def test_el_entrenamiento_crece_con_los_anios(resultados):
    """Ventana expansiva: cada año entrena con más historia que el anterior."""
    por_anio = resultados.groupby("anio")["n_entrenamiento"].first()
    assert por_anio.is_monotonic_increasing


def test_el_modelo_le_gana_a_la_regla_de_mesa(resultados):
    tabla = evaluacion.comparar(resultados).set_index("modelo")
    assert tabla.loc["ridge", "MdAPE"] < tabla.loc["mediana por ciudad y uso", "MdAPE"]
    assert tabla.loc["ridge", "PPE20"] > tabla.loc["mediana por ciudad y uso", "PPE20"]


def test_el_modelo_no_esta_sesgado(resultados):
    """Un sesgo sistemático al alza en una cartera de compras es peor que
    errar parejo: hace que todo parezca barato."""
    m = evaluacion.metricas_modelo(resultados["y"].to_numpy(),
                                   resultados["pred_lineal"].to_numpy())
    assert abs(m["sesgo_mediano"]) < 0.05


# --------------------------------------------------------------------------
# Intervalos
# --------------------------------------------------------------------------

def test_la_cobertura_alcanza_lo_prometido(resultados):
    mi = evaluacion.metricas_intervalo(resultados, alpha=0.10)
    assert mi["cobertura_observada"] >= 0.87


def test_los_intervalos_no_son_de_ancho_fijo(resultados):
    """El punto de normalizar: el rango tiene que depender del predio."""
    from src.conformal import ancho_relativo
    anchos = ancho_relativo(resultados["lo"].to_numpy(), resultados["hi"].to_numpy())
    assert anchos.std() / anchos.mean() > 0.15


def test_el_intervalo_se_abre_en_los_predios_grandes(resultados):
    from src.conformal import ancho_relativo
    a = pd.Series(ancho_relativo(resultados["lo"].to_numpy(),
                                 resultados["hi"].to_numpy()))
    grandes = resultados["superficie_m2"] > resultados["superficie_m2"].quantile(0.8)
    assert a[grandes.to_numpy()].median() > a[~grandes.to_numpy()].median()


def test_mas_confianza_da_intervalos_mas_anchos(ops):
    anchos = []
    for alpha in (0.30, 0.10):
        r = validar(ops, alpha=alpha)
        anchos.append(evaluacion.metricas_intervalo(r, alpha=alpha)["ancho_mediano"])
    assert anchos[1] > anchos[0]


def test_calibrar_con_los_datos_de_ajuste_da_un_intervalo_mentiroso(ops):
    """La razón de que los dos conjuntos sean disjuntos, medida.

    Se usa el boosting a propósito. Con ridge el efecto es despreciable —apenas
    sobreajusta, así que sus residuos en muestra se parecen a los de fuera— y
    eso por sí solo ya dice algo: **la trampa la paga el modelo flexible**.
    Con el boosting, calibrar sobre los datos de ajuste encoge el intervalo
    casi a la mitad y hunde la cobertura muy por debajo de lo prometido, sin
    que nada truene ni avise.
    """
    entrena = ops[ops["anio"] < 2024]
    calibra = ops[ops["anio"] == 2024]
    prueba = ops[ops["anio"] == 2025]
    y = features.objetivo(ops)
    ye, yc = y.loc[entrena.index], y.loc[calibra.index]
    yp = y.loc[prueba.index].to_numpy()

    punto = ModeloBoosting().entrenar(entrena, ye)
    bien = IntervaloConforme().entrenar(punto, entrena, ye, calibra, yc)
    mal = IntervaloConforme().entrenar(punto, entrena, ye, entrena, ye)

    lo_b, hi_b = bien.intervalo(punto, prueba)
    lo_m, hi_m = mal.intervalo(punto, prueba)

    assert (hi_m - lo_m).mean() < 0.75 * (hi_b - lo_b).mean()
    assert cobertura(yp, lo_b, hi_b) > 0.82
    assert cobertura(yp, lo_m, hi_m) < 0.75


# --------------------------------------------------------------------------
# El experimento de deflactar y la cartera
# --------------------------------------------------------------------------

def test_sin_deflactar_el_anio_pesa_mucho_mas(ops):
    comp = comparar_deflactado(ops.sample(1_400, random_state=3))
    peso = {e: float(comp[(comp["entrenamiento"] == e)
                          & (comp["columna"] == "anios_desde_inicio")]
                     ["peso_relativo"].iloc[0])
            for e in ("nominal", "deflactado")}
    assert peso["nominal"] > 3 * peso["deflactado"]


def test_la_cartera_se_valua_y_se_ordena(ops):
    port = generar_portafolio()
    modelo, intervalo, _ = entrenar_final(ops)
    a = analizar(port, modelo, intervalo)
    assert len(a) == len(port)
    assert a["brecha_pct"].is_monotonic_decreasing
    assert set(a["veredicto"]) <= {"revisar: por encima del rango",
                                   "compra por debajo del rango",
                                   "dentro de mercado"}
    # Los predios marcados tienen que estar realmente fuera de su rango.
    marcados = a[a["fuera_de_rango"]]
    assert ((marcados["pagado_m2_real"] > marcados["justo_m2_hi"])
            | (marcados["pagado_m2_real"] < marcados["justo_m2_lo"])).all()


def test_la_linea_base_cae_de_vuelta_cuando_faltan_comparables(ops):
    y = features.objetivo(ops)
    base = LineaBaseMediana().entrenar(ops, y)
    raro = ops.head(1).copy()
    raro["ciudad"] = "Ciudad que no existe"   # plaza que el modelo nunca vio
    raro["uso_suelo"] = "Comercial"
    assert np.isfinite(base.predecir(raro)).all()


def test_los_dos_modelos_predicen_en_la_escala_correcta(ops):
    y = features.objetivo(ops)
    for modelo in (ModeloLineal().entrenar(ops, y), ModeloBoosting().entrenar(ops, y)):
        pred = np.exp(modelo.predecir(ops.head(200)))
        assert (pred > 200).all() and (pred < 200_000).all()
