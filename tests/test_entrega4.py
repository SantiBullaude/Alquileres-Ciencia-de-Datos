"""Recorridos de la app y selección de avisos comparables."""
import sys
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))
from analisis_entrega4 import comparables  # noqa: E402


class ComparablesTest(unittest.TestCase):
    def test_filtra_y_excluye_el_aviso(self):
        datos = pd.DataFrame([
            {"clave": "a", "localidad": "Capital", "moneda": "ARS", "superficie_cubierta_m2": 50, "dormitorios": 2},
            {"clave": "b", "localidad": "Capital", "moneda": "ARS", "superficie_cubierta_m2": 55, "dormitorios": 2},
            {"clave": "c", "localidad": "Capital", "moneda": "USD", "superficie_cubierta_m2": 50, "dormitorios": 2},
            {"clave": "d", "localidad": "Capital", "moneda": "ARS", "superficie_cubierta_m2": 65, "dormitorios": 2},
            {"clave": "e", "localidad": "Capital", "moneda": "ARS", "superficie_cubierta_m2": 50, "dormitorios": 1},
        ])
        resultado = comparables(datos, "Capital", "ARS", 50, 2, excluir="a")
        self.assertEqual(resultado["clave"].tolist(), ["b"])
        self.assertTrue(comparables(datos, "Capital", "ARS", 50, None).empty)


class AppEntrega4Test(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(RAIZ / "app" / "streamlit_app.py"), default_timeout=90).run()
        self.assertFalse(self.app.exception)

    def campo(self, tipo, nombre):
        return next(c for c in getattr(self.app, tipo) if c.label == nombre)

    def test_simulacion_y_conversion(self):
        self.campo("number_input", "Precio mensual publicado (en la moneda elegida)").set_value(600000.0)
        self.campo("button", "Predecir").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.success)
        next(c for c in self.app.selectbox if c.label == "Moneda del aviso" and c.key != "explora_moneda").select("USD")
        self.campo("number_input", "Precio mensual publicado (en la moneda elegida)").set_value(400.0)
        self.campo("button", "Predecir").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("616.000" in m.value for m in self.app.markdown))

    def test_zonas_por_localidad_y_correlacion(self):
        datos = pd.read_parquet(RAIZ / "app" / "data" / "dataset_exploracion.parquet")
        localidad = next(c for c in self.app.selectbox if c.label == "Localidad" and c.key != "explora_localidad")
        localidad.select("Las Heras").run()
        self.assertFalse(self.app.exception)
        zona = self.campo("selectbox", "Zona geografica")
        disponibles = set(datos.loc[datos.localidad.eq("Las Heras"), "zona_geo"].dropna().astype(int).astype(str))
        self.assertIn(str(zona.value), disponibles)
        distancia = datos[(datos.moneda == "ARS") & datos.dist_centro_km.between(0, 30)].dropna(subset=["dist_centro_km", "precio_m2"])
        correlacion = distancia.dist_centro_km.corr(distancia.precio_m2, method="spearman")
        self.assertTrue(any(f"Spearman: {correlacion:.3f}" in c.value for c in self.app.caption))

    def test_navegacion_y_exploracion(self):
        self.app.multiselect(key="explora_localidad").set_value(["Capital", "Las Heras"]).run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("1.558 avisos con estos filtros" in c.value for c in self.app.caption))
        self.app.multiselect(key="explora_moneda").set_value(["ARS", "USD"]).run()
        self.assertFalse(self.app.exception)
        self.app.button(key="limpiar_exploracion").click().run()
        self.assertEqual(self.app.multiselect(key="explora_localidad").value, [])
        self.assertTrue(any("3.364 avisos con estos filtros" in c.value for c in self.app.caption))
        self.campo("radio", "La historia se lee en orden. Acá las preguntas las hacés vos.").set_value("Los casos").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["navegacion"], "Los casos")

if __name__ == "__main__":
    unittest.main()
