"""Brujula Inmobiliaria - Entrega 4.

Copa de martini: el tallo (El proyecto / Los datos / Lo que encontramos / El
modelo) cuenta los hallazgos en orden; la copa (Explora / Proba el modelo)
deja que el usuario pregunte lo suyo.

Los numeros de este archivo no estan escritos a mano: vienen de
app/data/metrics.json, que genera build_app.py corriendo el mismo pipeline
que notebooks/entrega3_modelado.ipynb con la misma semilla. Si el pipeline
suma avisos, correr build_app.py de nuevo actualiza todo lo que se muestra
aca.
"""

import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import r2_score

import pipeline_comun  # noqa: F401 - necesario para que joblib.load resuelva el pipeline
from analisis_entrega4 import ajuste, casos, comparables, mostrar_comparables, ubicacion
from paleta import COLORES_LOCALIDAD, COLORES_PORTAL, COLORES_MONEDA, COLORES_COMPARACION, COLORES_MODELO, escala

DATA = Path(__file__).resolve().parent / "data"

# --- paleta: igual criterio en toda la app -----------------------------------
COLOR_SOBREVALORADO = COLORES_COMPARACION["Sobrevalorado"]   # status: critical
COLOR_EN_LINEA = COLORES_COMPARACION["En linea"]        # status: neutral
COLOR_OPORTUNIDAD = COLORES_COMPARACION["Oportunidad"]     # status: good
ESCALA_ETIQUETA = alt.Scale(
    domain=["Sobrevalorado", "En linea", "Oportunidad"],
    range=[COLOR_SOBREVALORADO, COLOR_EN_LINEA, COLOR_OPORTUNIDAD],
)
COLOR_ARS = COLORES_MONEDA["ARS"]   # categorico slot 1
COLOR_USD = COLORES_MONEDA["USD"]   # categorico slot 2
ESCALA_MONEDA = alt.Scale(domain=["ARS", "USD"], range=[COLOR_ARS, COLOR_USD])
COLOR_ELEGIDO = "#2a78d6"
COLOR_RESTO = "#c3c2b7"

ETIQUETAS_COLUMNAS = {
    "superficie_cubierta_m2": "Superficie cubierta (m2)",
    "superficie_total_m2": "Superficie total (m2)",
    "dormitorios": "Dormitorios",
    "banos": "Banos",
    "antiguedad": "Antiguedad (anios)",
    "cochera": "Cochera",
    "estado_conservacion": "Estado de conservacion",
    "valor_expensas": "Expensas ($)",
    "tiene_expensas": "Tiene expensas",
    "acepta_mascotas": "Acepta mascotas",
    "localidad": "Localidad",
    "zona_geo": "Zona geografica",
    "dist_centro_km": "Distancia al centro (km)",
    "densidad_1km": "Densidad en 1 km (avisos vecinos)",
    "tipo_anunciante": "Tipo de anunciante",
    "fuente": "Portal de origen",
    "fecha_scraping": "Fecha de scraping",
    "moneda": "Moneda del aviso",
}


def tabla(df):
    """st.table, pero sin los .0000 de los floats ya redondeados en build_app.py."""
    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].map(lambda x: f"{x:,.0f}".replace(",", ".") if pd.notna(x) else "")
    return st.table(df)


def miles(x):
    """Separador de miles con punto (es-AR), sin tocar el resto del texto:
    nunca usar .replace(",", ".") sobre un bloque de markdown completo, porque
    tambien convierte las comas de puntuacion/listas en puntos."""
    return f"{x:,.0f}".replace(",", ".")


def coma(x, decimales=1, signo=False):
    """Coma decimal (es-AR) para un numero suelto, sin afectar el resto del texto."""
    spec = f"{{:{'+' if signo else ''}.{decimales}%}}"
    return spec.format(x).replace(".", ",")


@st.cache_resource
def cargar_todo():
    with open(DATA / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(DATA / "rangos_columnas.json", encoding="utf-8") as f:
        rangos = json.load(f)
    modelo = joblib.load(DATA / "modelo_final.joblib")
    explo = pd.read_parquet(DATA / "dataset_exploracion.parquet")
    return metrics, rangos, modelo, explo


metrics, rangos, modelo, explo = cargar_todo()
explo = explo.copy()
explo["precio_mensual_ars"] = explo["precio_m2"] * explo["superficie_cubierta_m2"]

st.set_page_config(page_title="¿Esta sobrevalorado?", layout="wide")

st.markdown("""
<style>
.bloque-cierra {font-size: 0.85rem; color: #898781; margin-top: -0.5rem;}
.stVegaLiteChart:has(> .vega-bindings select) {display: flex; flex-direction: column;}
.stVegaLiteChart:has(> .vega-bindings select) > .vega-bindings {order: -1; margin-bottom: 0.75rem;}
</style>
""", unsafe_allow_html=True)

SECCIONES = ["El proyecto", "Los datos", "Lo que encontramos", "El modelo", "Los casos", "Explora", "Proba el modelo"]
ANCLAS = {"El proyecto": "proyecto", "Los datos": "datos", "Lo que encontramos": "hallazgos", "El modelo": "modelo", "Los casos": "casos", "Explora": "explora", "Proba el modelo": "simulador"}


def ir_a_seccion():
    st.session_state["destino_pendiente"] = ANCLAS[st.session_state["navegacion"]]


st.sidebar.radio("La historia se lee en orden. Acá las preguntas las hacés vos.", SECCIONES,
                 key="navegacion", on_change=ir_a_seccion)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Proyecto Integrador - Ciencia de Datos 2026 (UTN FRM)\n\n"
    "Grupo 5k10-09 - Gustavo Santino Giovannini, Rodrigo Luis Vargas, "
    "Renzo Sahade, Garcia Zacarias, Santiago Bullaude"
)


# =============================================================================
# EL PROYECTO (portada)
# =============================================================================
def portada():
    st.title("¿Un alquiler en Mendoza está sobrevalorado, en línea, o es una oportunidad?")
    conteo = explo["etiqueta"].value_counts()
    dentro = int(conteo.get("En linea", 0))
    alto = int(conteo.get("Sobrevalorado", 0))
    bajo = int(conteo.get("Oportunidad", 0))
    st.markdown(
        f"Entrenamos un modelo para estimar cuánto piden por departamentos parecidos. "
        f"En {miles(alto + bajo)} de los {miles(len(explo))} avisos, la diferencia con esa "
        "referencia fue lo bastante grande como para revisar el precio."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("En línea", miles(dentro))
    c1.caption("Avisos dentro de los cortes de comparación.")
    c2.metric("Por encima", miles(alto))
    c2.caption("Avisos por encima del corte superior.")
    c3.metric("Por debajo", miles(bajo))
    c3.caption("Avisos por debajo del corte inferior.")
    st.caption(
        f"Avisos de Inmoclick e InmoUP recogidos entre el {metrics['dataset']['ventana_inicio']} "
        f"y el {metrics['dataset']['ventana_fin']}. Los cortes identifican diferencias "
        "frente a los precios publicados en ese período."
    )


# =============================================================================
# LOS DATOS
# =============================================================================
def los_datos():
    st.header("Los datos")
    st.subheader("Tres localidades concentran la mayoría de los avisos")
    cantidades = explo["localidad"].fillna("Sin localidad").value_counts().rename_axis("localidad").reset_index(name="avisos")
    principales = cantidades.head(3)
    localidades = ", ".join(principales.localidad.iloc[:-1]) + " y " + principales.localidad.iloc[-1]
    localidades = localidades.replace("Guaymallen", "Guaymallén")
    n_principales = principales.avisos.sum()
    st.markdown(
        f"Reunimos avisos de Inmoclick e InmoUP entre el {metrics['dataset']['ventana_inicio']} "
        f"y el {metrics['dataset']['ventana_fin']}. Cada fila representa una publicación. "
        "Una propiedad puede aparecer en los dos portales."
    )
    st.markdown(
        f"{localidades} suman **{miles(n_principales)} avisos** "
        f"({coma(n_principales / len(explo), decimales=0)} de la muestra). "
        "En localidades con pocos avisos cuesta más encontrar departamentos parecidos para comparar."
    )
    grafico = alt.Chart(cantidades).mark_bar().encode(
        x=alt.X("avisos:Q", title="Cantidad de avisos"),
        y=alt.Y("localidad:N", title=None, sort="-x", axis=alt.Axis(labelLimit=150)),
        color=alt.Color("localidad:N", scale=escala(COLORES_LOCALIDAD), legend=None),
        tooltip=[alt.Tooltip("localidad:N", title="Localidad"), alt.Tooltip("avisos:Q", title="Avisos")],
    ).properties(height=300)
    st.altair_chart(grafico, width="stretch")
    sin_coordenadas = explo[["latitud", "longitud"]].isna().any(axis=1).sum()
    st.markdown(
        f"Después de la limpieza quedaron {miles(len(explo))} de los "
        f"{miles(metrics['dataset']['filas_totales'])} avisos. "
        f"{miles(sin_coordenadas)} de los que quedaron carecen de coordenadas y se excluyen del mapa. "
        "Para comparar precios por metro cuadrado, pasamos los avisos en dólares a pesos."
    )

    with st.expander("Cómo lo sabemos"):
        st.markdown(f"""
**De dónde salen y cómo se obtienen.** Pipeline de Airflow con 5 tareas
(`extract_listings -> parse_raw -> transform_clean -> quality_check -> export_csv`),
scrapeando Inmoclick e InmoUP respetando el `robots.txt` de cada portal. El
CSV usado es `{metrics['dataset']['csv']}`, acumulado entre el
{metrics['dataset']['ventana_inicio']} y el {metrics['dataset']['ventana_fin']}.

**Qué es una fila.** Un aviso de departamento en alquiler publicado en uno de
los dos portales, en el momento del scraping. No una propiedad: el mismo
departamento publicado en Inmoclick y en InmoUP son dos filas.

**Qué se filtró, paso a paso:**
""")
        tabla_filtro = pd.DataFrame(metrics["pasos_filtro"])
        tabla(tabla_filtro.set_index("paso"))
        st.markdown(f"""
Los avisos en dólares se convierten con el dólar MEP del 13/10/2026
(${miles(metrics['dataset']['tc_usd'])}), porque en toda la ventana se movió
menos de 2%: el supuesto de una sola cotización se sostiene. Entran
{metrics['dataset']['avisos_usd_convertidos']} avisos más que si se
descartaran.

**Qué obstáculos encontramos.** Los avisos usan dos monedas y algunos tienen
superficies o precios sospechosos. También hay coordenadas erróneas y una
misma propiedad puede aparecer en ambos portales. Convertimos las monedas,
filtramos los registros sospechosos, anulamos coordenadas imposibles y agrupamos
posibles duplicados al separar entrenamiento y prueba.

**Qué columnas se descartaron por fuga:**
""")
        tabla(pd.DataFrame(metrics["columnas_descartadas_fuga"]).set_index("columna"))

        st.markdown("**Cuántos avisos nuevos trajo cada corrida del pipeline:**")
        evo = pd.DataFrame(metrics["evolucion_dataset"])
        chart_evo = alt.Chart(evo).mark_bar(color=COLOR_ARS).encode(
            x=alt.X("fecha_scraping:N", title="Fecha de scraping (primera vez que se vio el aviso)"),
            y=alt.Y("avisos_nuevos:Q", title="Avisos nuevos"),
            tooltip=["fecha_scraping", "avisos_nuevos", "avisos_acumulados"],
        ).properties(height=220)
        st.altair_chart(chart_evo, width="stretch")

    st.markdown('<p class="bloque-cierra">Cierra la Entrega 1.</p>', unsafe_allow_html=True)


# =============================================================================
# LO QUE ENCONTRAMOS
# =============================================================================
def precios_por_portal():
    with st.expander("Comparar los precios entre portales"):
        resumen = explo.groupby("fuente", as_index=False).agg(
            avisos=("precio_m2", "size"), mediana=("precio_m2", "median"))
        grafico = alt.Chart(resumen).mark_bar().encode(
            color=alt.Color("fuente:N", scale=escala(COLORES_PORTAL), legend=None),
            x=alt.X("mediana:Q", title="Mediana del precio publicado ($/m²)"),
            y=alt.Y("fuente:N", title="Portal", sort="-x"),
            tooltip=[alt.Tooltip("fuente:N", title="Portal"),
                     alt.Tooltip("avisos:Q", title="Avisos"),
                     alt.Tooltip("mediana:Q", title="Mediana ($/m²)", format=",.0f")]
        ).properties(height=170)
        st.altair_chart(grafico, width="stretch")
        st.caption("La mediana describe los avisos de cada portal; por sí sola no explica la diferencia de precio.")

        st.markdown("**Comparar portales en una misma localidad y rango de superficie:**")
        localidades = sorted(explo["localidad"].dropna().unique())
        localidad = st.selectbox("Localidad para comparar portales", localidades,
                                 index=localidades.index("Capital") if "Capital" in localidades else 0)
        rango = st.slider("Superficie para comparar (m²)", 10, 300, (40, 90))
        grupo = explo[explo["localidad"].eq(localidad) & explo["superficie_cubierta_m2"].between(*rango)]
        if grupo.empty:
            st.caption("No hay avisos para esa selección.")
        else:
            por_portal = grupo.groupby("fuente").agg(avisos=("clave", "size"), mediana=("precio_m2", "median"))
            tabla(por_portal)
            st.caption("La comparación controla localidad y superficie. Pueden quedar diferencias de moneda, estado y otras características.")


def lo_que_encontramos():
    st.header("Lo que encontramos")
    st.subheader("El alquiler mensual suele subir menos que la superficie")
    st.markdown(
        "Los departamentos más grandes suelen pedir más por mes, aunque el precio crece "
        "menos que la superficie. Por eso, aun al dividir el alquiler por los metros cuadrados, "
        "conviene comparar departamentos de tamaño parecido."
    )

    datos_superficie = explo[explo["moneda"] == "ARS"].dropna(subset=["superficie_cubierta_m2", "precio_mensual_ars"])
    base = alt.Chart(datos_superficie.sample(n=min(1500, len(datos_superficie)), random_state=42))
    lineal = base.mark_circle(size=14, opacity=0.3, color=COLOR_ARS).encode(
        x=alt.X("superficie_cubierta_m2:Q", title="superficie cubierta (m²)"),
        y=alt.Y("precio_mensual_ars:Q", title="Precio mensual ($)"),
    ).properties(height=320, title="Escala lineal: la nube se abre en curva")
    log = base.mark_circle(size=14, opacity=0.3, color=COLOR_ARS).encode(
        x=alt.X("superficie_cubierta_m2:Q", scale=alt.Scale(type="log"), title="superficie cubierta (m², log)"),
        y=alt.Y("precio_mensual_ars:Q", scale=alt.Scale(type="log"), title="Precio mensual ($, log)"),
    ).properties(height=320, title="Escala log-log: la curva se endereza")
    st.altair_chart(lineal | log, width="stretch")
    st.caption(f"Se muestran hasta 1.500 avisos en pesos de los {miles(len(datos_superficie))} incluidos en este análisis.")

    distancia = explo[(explo["moneda"] == "ARS") & explo["dist_centro_km"].between(0, 30)].dropna(subset=["dist_centro_km", "precio_m2"])
    correlacion_distancia = distancia["dist_centro_km"].corr(distancia["precio_m2"], method="spearman")
    with st.expander("Cómo lo sabemos"):
        superficie = datos_superficie["superficie_cubierta_m2"]
        precio = datos_superficie["precio_mensual_ars"]
        pearson = superficie.corr(precio)
        superficie_log, precio_log = np.log10(superficie), np.log10(precio)
        correlacion_log = superficie_log.corr(precio_log)
        pendiente_log = np.polyfit(superficie_log, precio_log, 1)[0]
        st.markdown(
            f"**Superficie y alquiler mensual.** La hipótesis de una relación no lineal se sostuvo. "
            f"En {miles(len(datos_superficie))} avisos en pesos, la correlación de Pearson pasó de "
            f"{pearson:.3f} a {correlacion_log:.3f} al aplicar logaritmo en ambos ejes. "
            f"La pendiente en esa escala fue {pendiente_log:.2f}. Se usaron los mismos avisos "
            "en los dos cálculos y la misma muestra de puntos en ambos gráficos."
        )
        st.markdown(
            f"**Distancia y precio por m².** La hipótesis de que alejarse del centro abarata "
            f"claramente el metro cuadrado tuvo poco respaldo. En {miles(len(distancia))} avisos "
            f"en pesos a hasta 30 km de Plaza Independencia, la correlación de Spearman fue "
            f"{correlacion_distancia:.3f}, muy cerca de cero."
        )

    with st.expander("Las otras hipótesis: moneda, ambientes y dormitorios"):
        st.markdown("Los importes en pesos y en dólares usan escalas distintas. Antes de compararlos convertimos los avisos en dólares a pesos con una misma cotización.")
        por_moneda = explo.groupby("moneda", as_index=False).agg(avisos=("clave", "size"), mediana=("precio_m2", "median"))
        st.altair_chart(alt.Chart(por_moneda).mark_bar().encode(
            color=alt.Color("moneda:N", scale=escala(COLORES_MONEDA), legend=None),
            x=alt.X("mediana:Q", title="Mediana convertida a pesos ($/m²)"),
            y=alt.Y("moneda:N", title="Moneda original"),
            tooltip=["moneda:N", alt.Tooltip("avisos:Q", title="Avisos"), alt.Tooltip("mediana:Q", title="Mediana ($/m²)", format=",.0f")]
        ).properties(height=130), width="stretch")
        st.caption("Las diferencias también pueden deberse al tipo de inmueble ofrecido en cada moneda.")
        origen = Path(__file__).resolve().parent.parent / metrics["dataset"]["csv"]
        if origen.exists():
            ambientes = pd.read_csv(origen, usecols=["clave", "ambientes"])
            cruces = explo[["clave", "dormitorios"]].merge(ambientes, on="clave", how="left", validate="one_to_one").dropna(subset=["ambientes", "dormitorios"])
            pares = cruces.groupby(["ambientes", "dormitorios"], as_index=False).size()
            st.markdown("La relación entre ambientes y dormitorios se ve en los avisos que informan ambos datos. Los ambientes pueden incluir espacios que no son dormitorios.")
            st.altair_chart(alt.Chart(pares).mark_rect().encode(
                x=alt.X("dormitorios:O", title="Dormitorios"), y=alt.Y("ambientes:O", title="Ambientes"),
                color=alt.Color("size:Q", title="Avisos", scale=alt.Scale(scheme="blues")),
                tooltip=[alt.Tooltip("ambientes:O", title="Ambientes"), alt.Tooltip("dormitorios:O", title="Dormitorios"), alt.Tooltip("size:Q", title="Avisos")]
            ).properties(height=250), width="stretch")
            st.caption(f"{miles(len(cruces))} avisos informan ambos datos; el resto queda fuera de esta comparación.")
        else:
            st.caption("No está disponible el archivo original para comparar ambientes y dormitorios.")


    st.subheader("La distancia al centro, por sí sola, explica poco del precio")
    st.markdown("La relación global es débil. Puede haber diferencias entre barrios y entre departamentos de distinto tamaño.")
    grafico_distancia = alt.Chart(distancia).mark_circle(size=14, opacity=.3, color=COLOR_ARS).encode(
        x=alt.X("dist_centro_km:Q", title="Distancia a Plaza Independencia (km)"),
        y=alt.Y("precio_m2:Q", title="Precio publicado ($/m²)"),
        tooltip=[alt.Tooltip("localidad:N", title="Localidad"),
                 alt.Tooltip("dist_centro_km:Q", title="Distancia (km)", format=".1f"),
                 alt.Tooltip("precio_m2:Q", title="Precio ($/m²)", format=",.0f")]
    ).properties(height=280).interactive()
    st.altair_chart(grafico_distancia, width="stretch")
    st.caption(f"{miles(len(distancia))} avisos en pesos a hasta 30 km del centro. Correlación de Spearman: {correlacion_distancia:.3f}.")
    ubicacion(explo, ESCALA_ETIQUETA)
    precios_por_portal()

    st.markdown('<p class="bloque-cierra">Cierra la Entrega 2.</p>', unsafe_allow_html=True)


# =============================================================================
# EL MODELO
# =============================================================================
def el_modelo():
    st.header("El modelo")
    test = metrics["test"]
    st.subheader(f"El modelo erra {test['mejora_vs_mejor_piso_pct']:.0%} menos que la mejor regla de medianas "
                 f"que conocemos, y {test['mejora_pct']:.0%} menos que la mediana global")
    st.markdown(
        "Un piso que solo predice la mediana global no sabe ni cuántos m² tiene "
        "el departamento. Por eso armamos una **escalera de pisos** cada vez más "
        "exigentes (medianas por tramo de superficie y por superficie × moneda) "
        "y comparamos contra cuatro familias de modelos con la misma validación "
        "por firma de inmueble; gana un **HistGradientBoosting** entrenado para "
        "minimizar el error absoluto."
    )

    st.markdown("Esta referencia se calcula de la misma manera para todos los avisos. Puede fallar si faltan características o si una zona está poco representada; tampoco conoce el precio de un contrato cerrado.")

    esc = pd.DataFrame(metrics["escalera_pisos"])
    orden_esc = esc["escalon"].tolist()
    esc["grupo_color"] = esc["escalon"].map(lambda v: v.split(":", 1)[0] if not v.startswith("Modelo:") else "HistGradientBoosting")
    chart_esc = alt.Chart(esc).mark_bar().encode(
        x=alt.X("test_mae:Q", title="MAE en test ($/m²) — menos es mejor"),
        y=alt.Y("escalon:N", sort=orden_esc, title=None),
        color=alt.Color("grupo_color:N", scale=escala(COLORES_MODELO), legend=None),
        tooltip=[
            alt.Tooltip("escalon:N", title="Escalón"),
            alt.Tooltip("regla:N", title="Regla"),
            alt.Tooltip("cv_mae:Q", title="MAE validación (15 folds)", format=",.0f"),
            alt.Tooltip("test_mae:Q", title="MAE test", format=",.0f"),
            alt.Tooltip("test_err_rel:Q", title="Error relativo mediano (test)", format=".1%"),
        ],
    ).properties(height=210, title="Escalera de pisos vs modelo final")
    st.altair_chart(chart_esc, width="stretch")

    comp = metrics["comparacion_modelos"]
    filas = []
    for m in comp:
        for v in m["folds"]:
            filas.append({"modelo": m["modelo"], "mae": v})
    df_folds = pd.DataFrame(filas)
    orden = [m["modelo"] for m in sorted(comp, key=lambda d: d["mae_mean"], reverse=True)]
    df_folds["grupo_color"] = df_folds["modelo"].map(lambda v: v.split(" ", 1)[0] if v.startswith(("P0 ", "P1 ", "P3 ")) else v)
    chart = alt.Chart(df_folds).mark_boxplot(size=22).encode(
        x=alt.X("mae:Q", title="MAE por fold ($/m²) — menos es mejor"),
        y=alt.Y("modelo:N", sort=orden, title=None),
        color=alt.Color("grupo_color:N", scale=escala(COLORES_MODELO), legend=None),
    ).properties(height=300, title="15 folds agrupados por firma de inmueble, mismos folds para todos")
    st.altair_chart(chart, width="stretch")
    st.caption("Cada color identifica el mismo modelo o regla en ambos gráficos.")

    with st.expander("Cómo lo sabemos"):
        st.markdown(f"""
**Métrica y por qué.** MAE (error absoluto medio) en $/m²: se lee en pesos
("le erramos por X pesos por m² en promedio"). Penaliza los errores de
forma lineal; los valores extremos también influyen en esta medida.

**La escalera de pisos.** Con MAE, la constante óptima dentro de un grupo es su
mediana. Probamos pisos con 0, 1 y 2 columnas (los cuantiles de superficie se
calculan **solo con el fold de entrenamiento**, y si un grupo tiene menos de 20
avisos cae al nivel anterior):
- **P0 (mediana global, 0 col):** {miles(metrics['piso']['mae_mean'])} ± {miles(metrics['piso']['mae_std'])} $/m² en CV ({miles(test['mae_piso'])} en test).
- **P1 (mediana por decil de superficie, 1 col):** {miles(metrics['piso_superficie']['mae_mean'])} ± {miles(metrics['piso_superficie']['mae_std'])} $/m² en CV ({miles(test['mae_piso_superficie'])} en test).
- **P3 (mediana por superficie × moneda, 2 col, mejor piso):** {miles(metrics['mejor_piso']['mae_mean'])} ± {miles(metrics['mejor_piso']['mae_std'])} $/m² en CV ({miles(test['mae_mejor_piso'])} en test).

**Cómo se armó la validación.** El mismo departamento puede estar publicado
en los dos portales; si una copia cae en entrenamiento y la otra en
validación, el modelo "se copia". Por eso el corte de test y los folds de
validación cruzada agrupan por **firma de inmueble** (dirección normalizada
+ localidad + superficie): {metrics['split']['pct_avisos_con_firma_compartida']:.0%}
de los avisos comparte firma con otro. Firmas compartidas entre train y
test: {metrics['split']['firmas_compartidas_train_test']}.
Las zonas se calcularon con las coordenadas de todo el conjunto antes de separar;
no usaron los precios de test. Haría falta otra evaluación para medir cómo
funciona el modelo en zonas o períodos nuevos.

**La tabla de comparación**, con media y desvío de los 15 folds:
""")
        tabla_comp = pd.DataFrame(comp)[["modelo", "mae_mean", "mae_std", "mejora_vs_piso", "mejora_vs_mejor_piso"]]
        tabla_comp["mejora_vs_piso"] = tabla_comp["mejora_vs_piso"].map("{:+.1%}".format)
        tabla_comp["mejora_vs_mejor_piso"] = tabla_comp["mejora_vs_mejor_piso"].map("{:+.1%}".format)
        tabla(tabla_comp.set_index("modelo"))

        st.markdown("**Fold contra fold, contra el elegido:**")
        tabla(pd.DataFrame(metrics["pares_contra_mejor"]).set_index("contra"))

        st.markdown(f"""
**Columnas.** De las 18 candidatas, salen {len(metrics['columnas_salen'])}
porque sacarlas no empeora el MAE en más de 5 $/m² o ayudan en menos de la
mitad de los folds: {", ".join(metrics['columnas_salen'])}. Quedan
{len(metrics['columnas_finales'])}: {", ".join(metrics['columnas_finales'])}.
Con todas las columnas el MAE era {miles(metrics['mae_con_todas'])}; con las
finales, {miles(metrics['mae_con_finales'])}.

**El ajuste del elegido.** Grilla sobre tasa de aprendizaje, cantidad de
árboles, tamaño de hoja y tipo de pérdida. Mejor combinación:
{metrics['grid_mejor_params']}, con MAE {miles(metrics['grid_mejor_mae'])}
(contra {miles(metrics['mae_sin_ajustar_finales'])} sin ajustar: la grilla
ayuda poco, menos que el desvío entre folds).

**El resultado en test**, medido una sola vez: MAE
**{miles(test['mae_modelo'])} $/m²** contra **{miles(test['mae_mejor_piso'])}**
del mejor piso P3 ({test['mejora_vs_mejor_piso_pct']:.0%} menos error) y
**{miles(test['mae_piso'])}** de la mediana global ({test['mejora_pct']:.0%}
menos error). Error relativo mediano:
{coma(test['error_relativo_mediano_modelo'])} contra
{coma(test['error_relativo_mediano_mejor_piso'])} del mejor piso y
{coma(test['error_relativo_mediano_piso'])} de la mediana global. En validación
cruzada el elegido daba {miles(test['mae_cv_elegido'])}, una cifra cercana
a la observada en test.

**Dónde se equivoca**, por tramo de precio real:
""")
        tabla(pd.DataFrame(metrics["por_tramo"]).set_index("tramo"))
        st.markdown("Por localidad:")
        tabla(pd.DataFrame(metrics["por_localidad"]).set_index("localidad"))
        st.markdown("Por moneda de publicación:")
        tabla(pd.DataFrame(metrics["por_moneda"]).set_index("moneda"))

        st.markdown("**En qué se apoya** (importancia por permutación, sobre el test):")
        imp = pd.DataFrame(metrics["importancia_permutacion"])
        chart_imp = alt.Chart(imp).mark_bar(color=COLOR_ELEGIDO).encode(
            x=alt.X("aumento_mae:Q", title="aumento del MAE al desordenar la columna ($/m²)"),
            y=alt.Y("columna:N", sort="-x", title=None),
        ).properties(height=260)
        st.altair_chart(chart_imp, width="stretch")

        fuga = metrics["fuga_superficie"]
        st.markdown(f"""
**El aporte de la superficie al modelo.**
1. *Existe al predecir:* la publica el aviso.
2. *No contiene la respuesta:* la pendiente log-log entre precio y superficie es
   **{fuga['pendiente_log_precio_vs_log_superficie']}** (< 1): en esta relación promedio, al duplicarse los m² el precio aumenta cerca
   del 64%. Esto no permite anticipar el cambio de precio de un aviso individual.
3. *Sin superficie sigue ganando:* al quitarla el MAE pasa a
   **{miles(fuga['mae_sin_superficie'])}** ± {miles(fuga['desvio_sin_superficie'])} $/m²,
   muy por debajo de la mediana global ({miles(fuga['mae_piso'])}).
4. *Con una sola columna no alcanza:* un HistGradientBoosting entrenado **solo
   con `superficie_cubierta_m2`** da **{miles(fuga['mae_solo_superficie'])}** ±
   {miles(fuga['desvio_solo_superficie'])} $/m² en CV ({miles(fuga['mae_solo_superficie_test'])}
   en test), prácticamente empatado con el piso P1 de mediana por superficie
   ({miles(metrics['piso_superficie']['mae_mean'])}). Las otras 12 columnas bajan
   el error de {miles(fuga['mae_solo_superficie'])} a {miles(fuga['mae_modelo_final_cv'])} $/m²
   (ganan en 15 de 15 folds).

**Por qué mantenemos un único par de cortes ({coma(metrics['cortes_residuo']['bajo'], signo=True)} / {coma(metrics['cortes_residuo']['alto'], signo=True)}) para el veredicto.**
Al cruzar el veredicto con el quintil del **precio real**, el quintil más caro
concentra más avisos "Sobrevalorados" y el más barato más "Oportunidades",
porque agrupar por el precio real junta en los extremos a los avisos que
quedan por encima o por debajo de la estimación. En cambio, al cruzar por el quintil del
**precio predicho** (lo único que conoce la app antes de saber cuánto pide el
aviso), el reparto es equilibrado en todos los tramos (entre 69% y 87% en línea):
""")
        c_r, c_p = st.columns(2)
        with c_r:
            st.caption("Por quintil de precio real (incluye el residuo)")
            df_cr = pd.DataFrame(metrics["cruce_veredicto_tramo_real"]).set_index("tramo")
            st.table(df_cr.map("{:.0%}".format))
        with c_p:
            st.caption("Por quintil de precio predicho (lo que ve la app)")
            df_cp = pd.DataFrame(metrics["cruce_veredicto_tramo_predicho"]).set_index("tramo_predicho")
            st.table(df_cp.map("{:.0%}".format))

    st.markdown('<p class="bloque-cierra">Cierra la Entrega 3.</p>', unsafe_allow_html=True)


# =============================================================================
# EXPLORA (copa)
# =============================================================================
def grafico_exploracion(datos):
    opciones = {
        "Superficie cubierta (m²)": "superficie_cubierta_m2",
        "Precio mensual (ARS)": "precio_mensual_ars",
        "Precio por m² (ARS)": "precio_m2",
        "Referencia por m² (ARS)": "precio_m2_predicho",
        "Diferencia relativa": "residuo_relativo",
        "Distancia al centro (km)": "dist_centro_km",
        "Dormitorios": "dormitorios",
    }
    nombres, columnas = list(opciones), list(opciones.values())
    eje_x = alt.param(name="eje_x", value=columnas[0], bind=alt.binding_select(options=columnas, labels=nombres, name="Eje horizontal: "))
    eje_y = alt.param(name="eje_y", value=columnas[2], bind=alt.binding_select(options=columnas, labels=nombres, name="Eje vertical: "))
    seleccion = alt.selection_interval(name="seleccion", empty=True)
    base = alt.Chart(datos).transform_calculate(valor_x="datum[eje_x]", valor_y="datum[eje_y]")
    puntos = base.mark_circle(size=42, opacity=.75).encode(
        x=alt.X("valor_x:Q", title="Eje horizontal", scale=alt.Scale(zero=False)),
        y=alt.Y("valor_y:Q", title="Eje vertical", scale=alt.Scale(zero=False)),
        color=alt.Color("etiqueta:N", scale=ESCALA_ETIQUETA, legend=alt.Legend(title="Comparación con la referencia", orient="bottom", columns=1)),
        opacity=alt.condition(seleccion, alt.value(.8), alt.value(.12)),
        tooltip=[alt.Tooltip("direccion:N", title="Dirección"), alt.Tooltip("localidad:N", title="Localidad"),
                 alt.Tooltip("superficie_cubierta_m2:Q", title="Superficie (m²)"),
                 alt.Tooltip("dormitorios:Q", title="Dormitorios", format=",.0f"),
                 alt.Tooltip("precio:Q", title="Precio publicado", format=",.0f"),
                 alt.Tooltip("moneda:N", title="Moneda"),
                 alt.Tooltip("precio_m2:Q", title="Publicado (ARS/m²)", format=",.0f"),
                 alt.Tooltip("precio_m2_predicho:Q", title="Referencia (ARS/m²)", format=",.0f"),
                 alt.Tooltip("residuo_relativo:Q", title="Diferencia", format="+.1%"),
                 alt.Tooltip("etiqueta:N", title="Comparación")],
    ).properties(height=400).add_params(eje_x, eje_y, seleccion)
    resumen = alt.Chart(datos).transform_filter(seleccion).mark_bar().encode(
        x=alt.X("count():Q", title="Cantidad de avisos"),
        y=alt.Y("etiqueta:N", title=None, sort=["Sobrevalorado", "En linea", "Oportunidad"]),
        color=alt.Color("etiqueta:N", scale=ESCALA_ETIQUETA, legend=None),
        tooltip=[alt.Tooltip("etiqueta:N", title="Comparación"), alt.Tooltip("count():Q", title="Avisos")],
    ).properties(height=110, title="Avisos dentro de la selección")
    return alt.vconcat(puntos, resumen).resolve_scale(color="shared")


def explora():
    st.header("Explorá los avisos")
    st.markdown("Elegí qué avisos querés ver y compará sus características. Pasá el cursor por un punto para consultar el precio y la referencia del modelo.")
    st.markdown(
        "Para seguir uno de los casos, filtrá su localidad y buscá departamentos de "
        "tamaño parecido. Después elegí **Referencia por m²** en el eje horizontal y "
        "**Precio por m²** en el vertical para comparar lo que se pide con lo que estima el modelo."
    )
    st.caption("Podés elegir varias opciones en cada filtro. Si lo dejás vacío, se muestran todas.")
    sup_max = float(explo["superficie_cubierta_m2"].max())

    def limpiar_filtros():
        for clave in ("explora_localidad", "explora_moneda", "explora_portal", "explora_comparacion"):
            st.session_state[clave] = []
        st.session_state["explora_superficie"] = (0.0, sup_max)

    col_localidad, col_moneda, col_portal = st.columns(3)
    with col_localidad:
        localidades = st.multiselect("Localidad", sorted(explo["localidad"].dropna().unique()),
                                     placeholder="Todas las localidades", key="explora_localidad")
    with col_moneda:
        monedas = st.multiselect("Moneda del aviso", sorted(explo["moneda"].dropna().unique()),
                                  placeholder="Todas las monedas", key="explora_moneda")
    with col_portal:
        portales = st.multiselect("Portal", sorted(explo["fuente"].dropna().unique()),
                                  placeholder="Todos los portales", key="explora_portal")
    col_superficie, col_comparacion = st.columns(2)
    with col_superficie:
        rango_sup = st.slider("Superficie cubierta (m²)", 0.0, sup_max, (0.0, sup_max), key="explora_superficie")
    with col_comparacion:
        comparaciones = st.multiselect("Comparación con la referencia", ["Sobrevalorado", "En linea", "Oportunidad"],
                                       placeholder="Todos los avisos", key="explora_comparacion",
                                       format_func=lambda v: "En línea" if v == "En linea" else v)
    st.button("Limpiar filtros", on_click=limpiar_filtros, key="limpiar_exploracion")
    mascara = explo["superficie_cubierta_m2"].between(*rango_sup)
    if localidades:
        mascara &= explo["localidad"].isin(localidades)
    if monedas:
        mascara &= explo["moneda"].isin(monedas)
    if portales:
        mascara &= explo["fuente"].isin(portales)
    if comparaciones:
        mascara &= explo["etiqueta"].isin(comparaciones)
    filtrado = explo.loc[mascara]
    st.caption(f"{miles(len(filtrado))} avisos con estos filtros, de {miles(len(explo))} en total.")
    if filtrado.empty:
        st.info("No encontramos avisos con esa combinación. Probá cambiar algún filtro.")
        return
    st.markdown("Cambiá los ejes en los controles del gráfico. Arrastrá sobre los puntos para seleccionar una zona: las barras de abajo cuentan esos avisos. Hacé doble clic para volver a ver todos.")
    st.altair_chart(grafico_exploracion(filtrado), width="stretch")
    with st.expander("Ver los avisos de estos filtros"):
        orden = st.selectbox("Ordenar por", ["Mayor precio por m²", "Menor precio por m²", "Mayor diferencia con la referencia", "Menor diferencia con la referencia"])
        columna = "precio_m2" if "precio" in orden else "residuo_relativo"
        vista = filtrado.sort_values(columna, ascending=orden.startswith("Menor")).head(50).copy()
        vista["residuo_relativo"] = vista["residuo_relativo"] * 100
        st.dataframe(vista[["localidad", "superficie_cubierta_m2", "dormitorios", "precio_m2", "residuo_relativo", "etiqueta", "url"]].rename(columns={"localidad": "Localidad", "superficie_cubierta_m2": "m²", "dormitorios": "Dormitorios", "precio_m2": "Precio (ARS/m²)", "residuo_relativo": "Diferencia", "etiqueta": "Comparación", "url": "Aviso"}), hide_index=True, width="stretch", column_config={"Diferencia": st.column_config.NumberColumn(format="%.1f%%"), "Aviso": st.column_config.LinkColumn("Aviso", display_text="Abrir")})
        st.caption("Se muestran hasta 50 avisos de los filtros elegidos. La selección del gráfico modifica las barras, no esta tabla.")
    st.caption("La referencia de cada aviso se calculó sin usar ese aviso para entrenar. Los cortes se obtuvieron de las diferencias en validación.")


# =============================================================================
# PROBA EL MODELO (copa)
# =============================================================================
def proba_el_modelo():
    st.header("Probá el modelo")
    st.markdown(
        "Cargá las características de un departamento para estimar su alquiler por m². "
        "El formulario ofrece categorías y rangos que vienen del conjunto de entrenamiento."
    )

    st.markdown(
        "Probá con un aviso que estés mirando. Cargá sus características y el precio publicado. "
        "Después, cambiá solo el precio para ver cuándo la diferencia supera los cortes y "
        "compará el resultado con avisos parecidos."
    )

    finales = metrics["columnas_finales"]
    valores = {}
    cols_ui = st.columns(2)
    i = 0
    orden_formulario = [c for c in finales if c != "zona_geo"]
    if "zona_geo" in finales:
        orden_formulario.insert(orden_formulario.index("localidad") + 1, "zona_geo")
    for col in orden_formulario:
        info = rangos.get(col)
        if info is None:
            continue
        etiqueta = ETIQUETAS_COLUMNAS.get(col, col)
        destino = cols_ui[i % 2]
        i += 1
        with destino:
            if info["tipo"] == "numerica":
                lo, hi, med = info["min"], info["max"], info["mediana"]
                # dormitorios/banos/antiguedad son conteos enteros, pero quedan
                # en float64 por los NaN del dataset: si los 3 valores son
                # enteros, el slider tambien lo es (si no, Streamlit deja
                # elegir decimales que no tienen sentido, ej. "2,5 banos").
                if float(lo).is_integer() and float(hi).is_integer() and float(med).is_integer():
                    valores[col] = st.slider(etiqueta, min_value=int(lo), max_value=int(hi), value=int(med))
                else:
                    valores[col] = st.slider(etiqueta, min_value=round(lo, 1), max_value=round(hi, 1),
                                              value=round(med, 1))
            elif info["tipo"] == "categorica":
                etiquetas_cat = info.get("etiquetas")
                if etiquetas_cat:
                    # zona_geo: el codigo de KMeans no dice nada por si solo,
                    # se muestra con la localidad mas comun de ese cluster
                    # (el valor real que recibe el pipeline sigue siendo el
                    # codigo, no el texto).
                    categorias = info["categorias"]
                    if col == "zona_geo" and valores.get("localidad"):
                        zonas_localidad = explo.loc[explo["localidad"] == valores["localidad"], "zona_geo"].dropna()
                        disponibles = set(zonas_localidad.astype(int).astype(str))
                        filtradas = [c for c in categorias if c in disponibles]
                        if filtradas:
                            mas_frecuente = str(int(zonas_localidad.mode().iloc[0]))
                            categorias = sorted(filtradas, key=lambda c: c != mas_frecuente)
                    valores[col] = st.selectbox(etiqueta, categorias,
                                                 format_func=lambda c, etiquetas=etiquetas_cat: etiquetas.get(c, c))
                else:
                    valores[col] = st.selectbox(etiqueta, info["categorias"])
            elif info["tipo"] == "fecha":
                valores[col] = info["max"]  # fecha de scraping: se fija a la corrida mas reciente, no la carga el usuario

    precio_pedido = st.number_input(
        "Precio mensual publicado (en la moneda elegida)",
        min_value=0.0, value=0.0, step=1000.0,
    )

    if st.button("Predecir", type="primary"):
        if (str(valores.get("zona_geo")) == "-1" and valores.get("dist_centro_km", 0) <= 30) or (str(valores.get("zona_geo")) != "-1" and valores.get("dist_centro_km", 0) > 30):
            st.warning("La zona y la distancia al centro no coinciden. Revisá esos valores para obtener una comparación coherente.")
        fila = pd.DataFrame([valores])
        for col in finales:
            if col not in fila.columns:
                fila[col] = np.nan
        pred = float(modelo.predict(fila)[0])
        st.success(f"El modelo estima **{pred:,.0f} $/m²** para este aviso.".replace(",", "."))

        test = metrics["test"]
        st.caption(f"En los avisos de prueba, el error absoluto medio fue ${miles(test['mae_modelo'])}/m². Es un promedio y no un intervalo para este aviso.")

        superficie = valores.get("superficie_cubierta_m2")
        if precio_pedido > 0 and superficie:
            precio_ars = precio_pedido * metrics["dataset"]["tc_usd"] if valores.get("moneda") == "USD" else precio_pedido
            precio_m2_real = precio_ars / superficie
            residuo = (precio_m2_real - pred) / pred
            corte_bajo, corte_alto = metrics["cortes_residuo"]["bajo"], metrics["cortes_residuo"]["alto"]
            if residuo > corte_alto:
                veredicto, color = "Sobrevalorado", COLOR_SOBREVALORADO
            elif residuo < corte_bajo:
                veredicto, color = "Oportunidad", COLOR_OPORTUNIDAD
            else:
                veredicto, color = "En línea con la referencia", COLOR_EN_LINEA
            st.markdown(
                f"Pide **{precio_m2_real:,.0f} $/m²** ({residuo:+.0%} contra lo esperado): "
                f"<span style='color:{color}; font-weight:700'>{veredicto}</span>".replace(",", "."),
                unsafe_allow_html=True,
            )

            st.write(f"Referencia: ${miles(pred * superficie)} por mes. Precio publicado: ${miles(precio_ars)} por mes, expresado en pesos.")
            if valores.get("moneda") == "USD":
                st.caption(f"Convertimos el precio con la cotización de ${miles(metrics['dataset']['tc_usd'])} por dólar usada en el dataset.")
            st.caption("El precio puede depender de características que no figuran en el aviso o que el modelo no conoce.")
            grupo = comparables(explo, valores.get("localidad"), valores.get("moneda"), superficie, valores.get("dormitorios"))
            if not grupo.empty:
                st.write(f"Mediana de los comparables: ${miles(grupo.precio_m2.median())}/m².")
            mostrar_comparables(grupo)
        else:
            st.caption("Ingresá un precio mensual para compararlo con la referencia y otros avisos.")

        for col, val in valores.items():
            info = rangos.get(col, {})
            if info.get("tipo") == "numerica" and not (info["min"] <= val <= info["max"]):
                st.warning(f"{ETIQUETAS_COLUMNAS.get(col, col)} = {val} está fuera de lo que vio el entrenamiento "
                           f"({info['min']}–{info['max']}). La predicción extrapola.")



def ancla(nombre):
    st.markdown(f'<span id="{nombre}"></span>', unsafe_allow_html=True)


ancla("proyecto")
portada()
ancla("datos")
los_datos()
ancla("hallazgos")
lo_que_encontramos()
ancla("modelo")
el_modelo()
ajuste(explo, metrics["test"])
ancla("casos")
casos(explo, metrics["cortes_residuo"])
ancla("explora")
explora()
ancla("simulador")
proba_el_modelo()

if "destino_pendiente" in st.session_state:
    destino = st.session_state.pop("destino_pendiente")
    components.html(f"""<script>
        const target = window.parent.document.getElementById('{destino}');
        if (target) target.scrollIntoView({{behavior: 'smooth', block: 'start'}});
    </script>""", height=0)
