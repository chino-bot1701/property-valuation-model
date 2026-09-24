"""
El valuador, como lo usaría el área de inversiones.

    streamlit run app/valuador.py

Tres pantallas, en el orden en que se toma una decisión: cuánto vale este
predio, qué tengo ya comprado y a qué precio, y por qué debería creerle al
modelo.

El modelo se entrena en el arranque (poco más de un segundo). Las métricas de
validación se leen de `data/synthetic/`, porque recorrer la ventana temporal
completa toma bastante más y no cambia entre visitas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import evaluacion, portafolio as pf  # noqa: E402
from src.conformal import ancho_relativo  # noqa: E402
from src.explicacion import NOMBRES  # noqa: E402
from src.generar_mercado import (ANIO_FINAL, ANIO_INICIAL, CIUDADES,  # noqa: E402
                                 USOS, generar_operaciones, generar_portafolio)
from src.validacion_temporal import entrenar_final  # noqa: E402

DATOS = RAIZ / "data" / "synthetic"
ALPHA = 0.10

st.set_page_config(page_title="Valuador de suelo", page_icon="📐", layout="wide")


@st.cache_resource(show_spinner="Entrenando el modelo sobre el mercado sintético…")
def preparar(alpha: float = ALPHA):
    ops = generar_operaciones()
    port = generar_portafolio()
    modelo, intervalo, _ = entrenar_final(ops, alpha=alpha)
    return ops, port, modelo, intervalo


@st.cache_data
def validacion() -> pd.DataFrame | None:
    ruta = DATOS / "validacion.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


@st.cache_data
def importancias() -> pd.DataFrame | None:
    ruta = RAIZ / "docs" / "importancias.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


ops, port, modelo, intervalo = preparar()

st.title("Valuador de suelo")
st.caption(
    "Portafolio inmobiliario · **implementación de referencia sobre datos "
    "sintéticos.** Ningún registro real, ni anonimizado: el mercado completo "
    "lo genera `src/generar_mercado.py` con una semilla fija."
)

tab_valuar, tab_cartera, tab_modelo = st.tabs(
    ["Valuar un predio", "La cartera", "Por qué creerle"])


# ==========================================================================
# 1. Valuar un predio
# ==========================================================================

with tab_valuar:
    izq, der = st.columns([2, 3])

    with izq:
        st.subheader("El predio")
        c1, c2 = st.columns(2)
        ciudad = c1.selectbox("Ciudad", sorted(CIUDADES))
        uso = c2.selectbox("Uso de suelo", sorted(USOS))

        superficie = st.slider("Superficie (m²)", 250, 60_000, 2_000, step=50)
        frente = st.slider("Frente (m)", 8, 320, 45, step=1)

        c3, c4 = st.columns(2)
        dist_centro = c3.slider("Distancia al centro (km)", 0.5, 25.0, 6.0, step=0.5)
        dist_vialidad = c4.slider("A vialidad primaria (m)", 20, 3_000, 400, step=20)

        c5, c6 = st.columns(2)
        densidad = c5.slider("Comercios en 1 km", 0, 500, 120, step=10)
        nse = c6.select_slider("Nivel socioeconómico", [1, 2, 3, 4, 5], value=3)

        c7, c8 = st.columns(2)
        cus = c7.slider("Coeficiente de uso de suelo", 0.4, 7.0, 2.2, step=0.1,
                        help="m² construibles por m² de terreno. Es de lo que "
                             "más mueve el precio en suelo comercial.")
        esquina = c8.toggle("Es esquina", value=False)

        confianza = st.select_slider(
            "Confianza del intervalo", ["70%", "80%", "90%"], value="90%",
            help="Más confianza, intervalo más ancho. No es gratis.")

    predio = pd.DataFrame([{
        "ciudad": ciudad, "uso_suelo": uso,
        "superficie_m2": float(superficie), "frente_m": float(frente),
        "dist_centro_km": float(dist_centro), "dist_vialidad_m": float(dist_vialidad),
        "densidad_comercios_1km": float(densidad),
        "nivel_socioeconomico": int(nse), "esquina": int(esquina),
        "cus": float(cus), "anio": ANIO_FINAL,
        "anios_desde_inicio": ANIO_FINAL - ANIO_INICIAL,
    }])

    # `preparar` está cacheada por nivel de confianza: cambiar el selector
    # reutiliza un modelo ya entrenado en lugar de volver a ajustarlo.
    alpha_sel = {"70%": 0.30, "80%": 0.20, "90%": 0.10}[confianza]
    _, _, _, intervalo_sel = preparar(alpha_sel)

    centro = float(np.exp(modelo.predecir(predio))[0])
    lo, hi = intervalo_sel.intervalo(modelo, predio)
    lo, hi = float(np.exp(lo[0])), float(np.exp(hi[0]))
    ancho = (hi - lo) / centro

    with der:
        st.subheader("La valuación")
        k1, k2 = st.columns(2)
        k1.metric("Precio estimado", f"${centro:,.0f} /m²")
        k2.metric("Valor del predio", f"${centro * superficie / 1e6:,.1f} M")

        st.markdown(
            f"Rango del **{confianza}**: de **\\${lo:,.0f}** a **\\${hi:,.0f}** por m² "
            f"— es decir ±{ancho / 2:.0%} alrededor del punto central. "
            f"En total, entre \\${lo * superficie / 1e6:,.1f} M y "
            f"\\${hi * superficie / 1e6:,.1f} M."
        )
        st.progress(min(ancho / 1.2, 1.0),
                    text="qué tan incierta es esta valuación")

        # Comparables: operaciones parecidas del mercado sintético.
        similares = ops[(ops["ciudad"] == ciudad) & (ops["uso_suelo"] == uso)].copy()
        if len(similares) >= 5:
            similares["cercania"] = (
                np.abs(np.log(similares["superficie_m2"] / superficie))
                + np.abs(np.log(similares["dist_centro_km"] / dist_centro)))
            comparables = similares.nsmallest(8, "cercania")
            st.caption(f"Los 8 comparables más parecidos de los "
                       f"{len(similares)} registrados en {ciudad}")
            st.dataframe(
                comparables[["anio", "superficie_m2", "dist_centro_km", "cus",
                             "esquina", "precio_m2_nominal"]]
                .rename(columns={"anio": "Año", "superficie_m2": "m²",
                                 "dist_centro_km": "km al centro", "cus": "CUS",
                                 "esquina": "Esquina",
                                 "precio_m2_nominal": "Precio/m² pagado"}),
                hide_index=True, width="stretch")
        else:
            st.info("Pocos comparables para esa combinación de ciudad y uso. "
                    "El intervalo ya lo refleja: es más ancho.")


# ==========================================================================
# 2. La cartera
# ==========================================================================

with tab_cartera:
    nivel = st.select_slider(
        "Qué tan estricto es el tamiz", ["70%", "80%", "90%"], value="90%",
        help="El intervalo con el que se decide si una compra amerita revisión. "
             "Más estricto marca menos y se equivoca menos.")
    _, _, _, intervalo_cartera = preparar({"70%": 0.30, "80%": 0.20, "90%": 0.10}[nivel])

    analisis = pf.analizar(port, modelo, intervalo_cartera)
    res = pf.resumen(analisis)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Invertido", f"${res['invertido_real'] / 1e6:,.0f} M",
              help="En pesos constantes de hoy.")
    k2.metric("Valor hoy", f"${res['valor_hoy'] / 1e6:,.0f} M")
    k3.metric("Plusvalía real", f"{res['plusvalia_real_cartera']:+.1%}",
              help="Por encima de la inflación. No es el número que sale de "
                   "comparar pesos de 2018 con pesos de 2025.")
    k4.metric("Predios a revisar", res["por_encima_del_rango"])

    if res["por_encima_del_rango"]:
        st.info(
            f"Quitando los {res['por_encima_del_rango'] + res['por_debajo_del_rango']} "
            f"predios que el modelo manda a revisar, la plusvalía real de la "
            f"cartera sería **{res['plusvalia_sin_marcados']:+.1%}** en lugar de "
            f"**{res['plusvalia_real_cartera']:+.1%}**. Un puñado de operaciones "
            f"explica casi toda la diferencia."
        )

    solo_marcados = st.checkbox("Solo los que ameritan revisión", value=False)
    tabla = analisis[analisis["fuera_de_rango"]] if solo_marcados else analisis

    st.dataframe(
        tabla[["predio_id", "ciudad", "uso_suelo", "superficie_m2", "anio",
               "pagado_m2_real", "justo_m2", "justo_m2_lo", "justo_m2_hi",
               "brecha_pct", "veredicto"]]
        .rename(columns={
            "predio_id": "Predio", "ciudad": "Ciudad", "uso_suelo": "Uso",
            "superficie_m2": "m²", "anio": "Año",
            "pagado_m2_real": "Pagado/m²", "justo_m2": "Modelo/m²",
            "justo_m2_lo": "Mínimo", "justo_m2_hi": "Máximo",
            "brecha_pct": "Brecha", "veredicto": "Veredicto"}),
        hide_index=True, width="stretch", height=420,
        column_config={
            "Pagado/m²": st.column_config.NumberColumn(format="dollar"),
            "Modelo/m²": st.column_config.NumberColumn(format="dollar"),
            "Mínimo": st.column_config.NumberColumn(format="dollar"),
            "Máximo": st.column_config.NumberColumn(format="dollar"),
            "Brecha": st.column_config.NumberColumn(format="percent"),
        })

    st.caption(
        "**La brecha sola no decide.** Un predio puede estar 47% por encima del "
        "punto central y seguir dentro de mercado, si es del tipo que el modelo "
        "valúa con poca certeza. Lo que manda a revisión es salirse del rango, "
        "no el tamaño de la diferencia."
    )


# ==========================================================================
# 3. Por qué creerle
# ==========================================================================

with tab_modelo:
    r = validacion()
    if r is None:
        st.warning("Corre `python scripts/run_demo.py` para generar las métricas "
                   "de validación.")
        st.stop()

    st.subheader("Los tres modelos, sobre los mismos predios")
    comp = evaluacion.comparar(r)
    st.dataframe(
        comp.rename(columns={"modelo": "Modelo", "MdAPE": "Error mediano",
                             "sesgo_mediano": "Sesgo",
                             "mejora_vs_base": "Mejora vs. la regla de mesa"}),
        hide_index=True, width="stretch",
        column_config={
            "Error mediano": st.column_config.NumberColumn(format="percent"),
            "PPE10": st.column_config.NumberColumn(format="percent"),
            "PPE20": st.column_config.NumberColumn(format="percent"),
            "Sesgo": st.column_config.NumberColumn(format="percent"),
            "Mejora vs. la regla de mesa": st.column_config.NumberColumn(
                format="percent")})
    st.caption(
        "**PPE10 y PPE20** son las métricas de la industria de valuación: qué "
        "proporción de las estimaciones cae dentro de ±10% y ±20% del precio "
        "real. El **gradient boosting pierde contra la regresión lineal**, y por "
        "eso el que se despliega es el lineal. Con 2,600 comparables y un "
        "mercado cuya estructura es sobre todo multiplicativa, el modelo "
        "flexible no tiene de dónde sacar ventaja y sí tiene de dónde "
        "sobreajustar."
    )

    st.divider()
    st.subheader("Los intervalos cumplen lo que prometen")
    mi = evaluacion.metricas_intervalo(r, alpha=ALPHA)
    k1, k2, k3 = st.columns(3)
    k1.metric("Cobertura prometida", f"{mi['cobertura_objetivo']:.0%}")
    k2.metric("Cobertura observada", f"{mi['cobertura_observada']:.1%}")
    k3.metric("Ancho mediano", f"±{mi['ancho_mediano'] / 2:.0%}")

    anual = evaluacion.por_anio(r, alpha=ALPHA)
    izq, der = st.columns(2)
    izq.caption("Cobertura año por año — validación con ventana expansiva")
    izq.bar_chart(anual.set_index("anio")["cobertura"], height=250)
    der.caption("Error mediano año por año")
    der.bar_chart(anual.set_index("anio")["MdAPE"], height=250)

    anchos = ancho_relativo(r["lo"].to_numpy(), r["hi"].to_numpy())
    st.caption("El intervalo se abre donde hay menos comparables: cada punto es "
               "un predio valuado fuera de muestra")
    st.scatter_chart(
        pd.DataFrame({"superficie (m²)": r["superficie_m2"],
                      "ancho del intervalo": anchos}),
        x="superficie (m²)", y="ancho del intervalo", height=280)

    st.divider()
    st.subheader("Qué pasa si no se descuenta la inflación")
    imp = importancias()
    if imp is not None:
        izq, der = st.columns(2)
        for col, etiqueta, titulo in [
                (izq, "nominal", "Entrenado en pesos corrientes"),
                (der, "deflactado", "Entrenado en pesos constantes")]:
            sub = imp[imp["entrenamiento"] == etiqueta].head(8)
            col.caption(titulo)
            col.bar_chart(sub.set_index("variable")["peso_relativo"],
                          horizontal=True, height=300)
        fila_n = imp[(imp["entrenamiento"] == "nominal")
                     & (imp["columna"] == "anios_desde_inicio")].iloc[0]
        fila_d = imp[(imp["entrenamiento"] == "deflactado")
                     & (imp["columna"] == "anios_desde_inicio")].iloc[0]
        st.markdown(
            f"Sobre precios corrientes, **el año de la operación** queda en el "
            f"lugar **{int(fila_n['lugar'])}** de 20 con un peso de "
            f"**{fila_n['peso_relativo']:.2f}**. Descontada la inflación cae al "
            f"lugar **{int(fila_d['lugar'])}** con peso "
            f"**{fila_d['peso_relativo']:.2f}** — "
            f"**{fila_n['peso_relativo'] / fila_d['peso_relativo']:.0f} veces menos**. "
            f"Ese primer modelo no estaba valuando suelo: estaba prediciendo "
            f"inflación, y lo habría seguido haciendo sin que las métricas se "
            f"quejaran."
        )
