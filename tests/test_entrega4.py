"""Recorridos de la app y selección de avisos comparables."""
import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np
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

    def test_el_cierre_responde_con_los_cortes_del_modelo(self):
        """La seccion final existe, se puede navegar y repite el corte que calculo build_app."""
        metrics = json.loads((RAIZ / "app" / "data" / "metrics.json").read_text(encoding="utf-8"))
        corte = f"{metrics['cortes_residuo']['alto']:.1%}".replace(".", ",").replace("%", " %")
        self.assertIn("La respuesta", [h.value for h in self.app.header])
        cierres = [sh.value for sh in self.app.subheader if "Un alquiler está sobrevalorado" in sh.value]
        self.assertEqual(len(cierres), 1)
        # el corte va en el cuerpo, no en el titulo
        self.assertTrue(any(corte in m.value for m in self.app.markdown),
                        f"ningun texto trae el corte {corte}")
        radio = self.campo("radio", "La historia se lee en orden. Acá las preguntas las hacés vos.")
        radio.set_value("La respuesta").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["navegacion"], "La respuesta")

    def test_la_cochera_se_explica_donde_corresponde(self):
        """El titular de sobreprecios no promete la cochera; el de medianas si la incluye."""
        titulares = [sh.value for sh in self.app.subheader]
        sobreprecios = next(t for t in titulares if "duplican los sobreprecios" in t)
        self.assertNotIn("cochera", sobreprecios.lower())
        self.assertTrue(any(t.startswith("Qué sube y qué baja el precio") for t in titulares))

    def test_las_cifras_salen_de_los_datos(self):
        """Las cifras del cuerpo se recalculan: nada escrito a mano."""
        datos = pd.read_parquet(RAIZ / "app" / "data" / "dataset_exploracion.parquet")
        pesos = datos[datos.moneda.eq("ARS")].copy()
        pesos["mensual"] = pesos.precio_m2 * pesos.superficie_cubierta_m2
        pesos = pesos.dropna(subset=["superficie_cubierta_m2", "mensual"])
        pendiente = np.polyfit(np.log10(pesos.superficie_cubierta_m2), np.log10(pesos.mensual), 1)[0]
        crecimiento = f"{2 ** pendiente - 1:.0%}".replace("%", " %")
        textos = [m.value for m in self.app.markdown]
        self.assertTrue(any(f"crece un **{crecimiento}**" in t for t in textos),
                        f"ningun texto trae {crecimiento}")
        fuera = f"{int((~datos.etiqueta.eq('En linea')).sum()):,}".replace(",", ".")
        self.assertTrue(any(fuera in t for t in textos), f"ningun texto trae {fuera} avisos fuera de linea")

    def test_los_titulos_no_llevan_cifras(self):
        """Los numeros van en el cuerpo: los titulos afirman el hallazgo en palabras."""
        titulos = [sh.value for sh in self.app.subheader] + [h.value for h in self.app.header]
        con_cifras = [t for t in titulos if re.search(r"[0-9]", t)]
        self.assertEqual(con_cifras, [], f"estos titulos traen cifras: {con_cifras}")

if __name__ == "__main__":
    unittest.main()
