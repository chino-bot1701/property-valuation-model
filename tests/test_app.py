"""
El valuador tiene que levantar sin credenciales y sin explotar.

`AppTest` corre el script de Streamlit de verdad, sin navegador. No prueba que
se vea bonito; prueba que renderiza, que las tres pestañas traen contenido y
que mover los controles no rompe nada — que es lo que se descubre tarde cuando
la demo se enseña en vivo.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "valuador.py"
FIXTURE = APP.parents[1] / "data" / "synthetic" / "validacion.csv"


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    return at


def test_levanta_sin_excepciones(app):
    assert not app.exception


def test_declara_que_los_datos_son_sinteticos(app):
    texto = " ".join(e.value for e in app.caption)
    assert "sintéticos" in texto
    assert "Ningún registro real" in texto


def test_entrega_una_valuacion(app):
    etiquetas = [m.label for m in app.metric]
    assert "Precio estimado" in etiquetas
    assert "Valor del predio" in etiquetas
    precio = next(m for m in app.metric if m.label == "Precio estimado")
    assert "$" in precio.value


def test_muestra_el_intervalo_con_los_dos_extremos(app):
    textos = " ".join(m.value for m in app.markdown)
    assert "Rango del" in textos
    assert "por m²" in textos


def test_la_cartera_trae_predios_y_veredictos(app):
    assert len(app.dataframe) >= 2
    cartera = next(df.value for df in app.dataframe if "Veredicto" in df.value.columns)
    assert len(cartera) > 0
    assert cartera["Veredicto"].isin(
        ["revisar: por encima del rango", "compra por debajo del rango",
         "dentro de mercado"]).all()


@pytest.mark.skipif(not FIXTURE.exists(),
                    reason="faltan las métricas; corre `python scripts/run_demo.py`")
def test_la_pestana_del_modelo_trae_las_metricas(app):
    etiquetas = [m.label for m in app.metric]
    assert "Cobertura prometida" in etiquetas
    assert "Cobertura observada" in etiquetas


def test_mover_el_predio_cambia_la_valuacion():
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    antes = next(m for m in at.metric if m.label == "Precio estimado").value

    # Alejarlo del centro tiene que bajarle el precio. Si no cambia nada, el
    # control está desconectado del modelo y la demo es una maqueta.
    distancia = next(s for s in at.slider if "Distancia al centro" in s.label)
    distancia.set_value(24.0).run()

    assert not at.exception
    despues = next(m for m in at.metric if m.label == "Precio estimado").value
    assert antes != despues
