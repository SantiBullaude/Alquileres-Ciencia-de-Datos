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

import pipeline_comun  # noqa: F401 - necesario para que joblib.load resuelva el pipeline

DATA = Path(__file__).resolve().parent / "data"

# --- paleta: igual criterio en toda la app -----------------------------------
COLOR_SOBREVALORADO = "#d03b3b"   # status: critical
COLOR_EN_LINEA = "#9e9d98"        # status: neutral
COLOR_OPORTUNIDAD = "#0ca30c"     # status: good
ESCALA_ETIQUETA = alt.Scale(
    domain=["Sobrevalorado", "En linea", "Oportunidad"],
    range=[COLOR_SOBREVALORADO, COLOR_EN_LINEA, COLOR_OPORTUNIDAD],
)
COLOR_ARS = "#2a78d6"   # categorico slot 1
COLOR_USD = "#eb6834"   # categorico slot 2
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
    "zona_geo": "Zona geografica (cluster)",
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

st.set_page_config(page_title="¿Esta sobrevalorado?", layout="wide")

st.markdown("""
<style>
.bloque-cierra {font-size: 0.85rem; color: #898781; margin-top: -0.5rem;}
</style>
""", unsafe_allow_html=True)

SECCIONES = ["El proyecto", "Los datos", "Lo que encontramos", "El modelo", "Explora", "Proba el modelo"]
seccion = st.sidebar.radio("La historia se lee en orden. Acá las preguntas las hacés vos.", SECCIONES)
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
    st.markdown(
        "A partir de los datos que publica el aviso —superficie, ubicación, "
        "dormitorios, cochera, moneda— el modelo estima cuánto **debería** "
        "costar el alquiler, y la diferencia con lo que pide de verdad separa "
        "los avisos sobrevalorados de las oportunidades."
    )

    test = metrics["test"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Avisos analizados", f"{metrics['dataset']['filas_modeladas']:,}".replace(",", "."))
        st.caption("Departamentos en alquiler de Inmoclick e InmoUP, Mendoza, "
                   f"entre el {metrics['dataset']['ventana_inicio']} y el {metrics['dataset']['ventana_fin']}.")
    with c2:
        st.metric("Menos error que no mirar nada", f"{test['mejora_pct']:.0%}")
        st.caption(f"MAE del modelo: {test['mae_modelo']:,.0f} $/m² · del piso (la mediana): "
                   f"{test['mae_piso']:,.0f} $/m² (medido en el 20% de test, una sola vez).".replace(",", "."))
    with c3:
        st.metric("Error relativo típico", f"{test['error_relativo_mediano_modelo']:.0%}")
        st.caption(f"Contra {test['error_relativo_mediano_piso']:.0%} del piso. "
                   "Por eso la app no marca sobrevalorado a un aviso que pide solo un 5% más.")

    st.info(
        "La pregunta del proyecto. El modelo es supervisado (regresión sobre "
        "`precio_m2`); el resultado se lee como **veredicto**: sobrevalorado, "
        "en línea con el mercado, u oportunidad. El detalle completo está en "
        "`Documentacion/Propuestas_Proyecto_Integrador_Inmobiliario.docx`."
    )


# =============================================================================
# LOS DATOS
# =============================================================================
def los_datos():
    st.header("Los datos")
    st.subheader("Hay dos monedas mezcladas en la misma columna, y una vale mil veces más que la otra")
    st.markdown(
        "La mayoría de los avisos publica el precio en pesos, pero uno de cada "
        "once lo publica en dólares. Sin convertir, un `precio_m2` de 10.000 y "
        "uno de 10 conviven en la misma columna: no son dos mercados distintos, "
        "son dos **unidades** distintas."
    )

    h1 = metrics["hallazgos_eda"]["h1_monedas"]
    datos_moneda = explo[["moneda", "precio_m2"]].dropna().copy()
    chart = alt.Chart(datos_moneda).mark_circle(size=18, opacity=0.35).encode(
        x=alt.X("precio_m2:Q", scale=alt.Scale(type="log"), title="precio_m2, sin convertir (escala log)"),
        y=alt.Y("moneda:N", title=None, sort=["ARS", "USD"]),
        color=alt.Color("moneda:N", scale=ESCALA_MONEDA, legend=alt.Legend(title="Moneda")),
        yOffset="jitter:Q",
        tooltip=[alt.Tooltip("moneda:N", title="Moneda"), alt.Tooltip("precio_m2:Q", title="precio_m2", format=",.1f")],
    ).transform_calculate(jitter="(random()-0.5)*0.6").properties(height=220)
    st.altair_chart(chart, width="stretch")
    st.caption(
        f"Separación estandarizada entre ambas sin convertir: **{h1['separacion_sin_convertir']}** "
        f"(>0,8 es zona verde: dos poblaciones bien distintas). ARS: {h1['n_ars']:,} avisos, mediana "
        f"{h1['mediana_ars']:,.0f} $/m². USD: {h1['n_usd']} avisos, mediana {h1['mediana_usd']} USD/m².".replace(",", ".")
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
(${metrics['dataset']['tc_usd']:,.0f}), porque en toda la ventana se movió
menos de 2%: el supuesto de una sola cotización se sostiene. Entran
{metrics['dataset']['avisos_usd_convertidos']} avisos más que si se
descartaran.

**Qué columnas se descartaron por fuga:**
""".replace(",", "."))
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
def lo_que_encontramos():
    st.header("Lo que encontramos")
    st.subheader("El precio no sube en línea recta con la superficie: achicarse a la mitad no abarata a la mitad")
    st.markdown(
        "Un departamento de 100 m² no cuesta el doble que uno de 50: cuesta "
        "menos que el doble. La relación entre superficie y precio es curva, "
        "y en escala logarítmica se endereza."
    )

    h2 = metrics["hallazgos_eda"]["h2_superficie_sostenida"]
    base = alt.Chart(explo.dropna(subset=["superficie_cubierta_m2", "precio_m2"]).sample(
        n=min(1500, len(explo)), random_state=42))
    lineal = base.mark_circle(size=14, opacity=0.3, color=COLOR_ARS).encode(
        x=alt.X("superficie_cubierta_m2:Q", title="superficie cubierta (m²)"),
        y=alt.Y("precio_m2:Q", title="precio_m2 ($)"),
    ).properties(height=320, title="Escala lineal: la nube se abre en curva")
    log = base.mark_circle(size=14, opacity=0.3, color=COLOR_ARS).encode(
        x=alt.X("superficie_cubierta_m2:Q", scale=alt.Scale(type="log"), title="superficie cubierta (m², log)"),
        y=alt.Y("precio_m2:Q", scale=alt.Scale(type="log"), title="precio_m2 ($, log)"),
    ).properties(height=320, title="Escala log-log: la curva se endereza")
    st.altair_chart(lineal | log, width="stretch")

    with st.expander("Cómo lo sabemos"):
        st.markdown(f"""
**Una hipótesis que se sostuvo: la superficie explica el precio, pero en
escala logarítmica.** En escala lineal, Spearman da {h2['spearman_lineal']}
y Pearson {h2['pearson_lineal']} contra `precio` — una brecha de
{h2['brecha_lineal']} (zona amarilla: la nube sube ordenada pero curva).
Pasando ambas variables a log10, Pearson sube a {h2['pearson_log']} y la
brecha cae a {h2['brecha_log']} (zona verde: la curvatura desaparece).
Medido sobre {h2['n']:,} avisos en pesos.

**Una que no: la distancia al centro, sola, no explica el precio por m².**
Spearman entre `dist_centro_km` y `precio_m2` da
**{metrics['hallazgos_eda']['h3_distancia_refutada']['spearman']}**: zona
roja, refutada. (La sección "El modelo" retoma esta variable separando por
tamaño de departamento, donde el efecto sí aparece en los más chicos.)

**Cómo se construyó el gráfico.** Transformación `log10` en ambos ejes del
panel derecho; sin esa transformación, la asimetría de ambas variables
aplasta la mayoría de los puntos contra el origen.
""".replace(",", "."))

    st.markdown('<p class="bloque-cierra">Cierra la Entrega 2.</p>', unsafe_allow_html=True)


# =============================================================================
# EL MODELO
# =============================================================================
def el_modelo():
    st.header("El modelo")
    test = metrics["test"]
    st.subheader(f"Con los datos del aviso, el modelo le gana al piso por un {test['mejora_pct']:.0%}, "
                 "y lo que más usa es la superficie")
    st.markdown(
        "Se compararon cuatro familias con la misma validación y las mismas "
        "columnas; gana un **HistGradientBoosting** entrenado para minimizar "
        "el error absoluto."
    )

    comp = metrics["comparacion_modelos"]
    filas = []
    for m in comp:
        for v in m["folds"]:
            filas.append({"modelo": m["modelo"], "mae": v})
    df_folds = pd.DataFrame(filas)
    orden = [m["modelo"] for m in sorted(comp, key=lambda d: d["mae_mean"], reverse=True)]
    chart = alt.Chart(df_folds).mark_boxplot(size=24).encode(
        x=alt.X("mae:Q", title="MAE por fold ($/m²) — menos es mejor"),
        y=alt.Y("modelo:N", sort=orden, title=None),
        color=alt.condition(alt.datum.modelo == "HistGradientBoosting",
                             alt.value(COLOR_ELEGIDO), alt.value(COLOR_RESTO)),
    ).properties(height=260, title="15 folds agrupados por firma de inmueble, mismos folds para los 5 modelos")
    st.altair_chart(chart, width="stretch")
    st.caption("Azul: el modelo elegido. Gris: las demás familias probadas, incluido el piso.")

    with st.expander("Cómo lo sabemos"):
        st.markdown(f"""
**Métrica y por qué.** MAE (error absoluto medio) en $/m²: se lee en pesos
("le erramos por X pesos por m² en promedio") y no se deja arrastrar por
avisos mal cargados, a diferencia de RMSE.

**El piso.** `DummyRegressor` con la mediana del entrenamiento: con MAE, la
constante que menos erra es la mediana, no el promedio. MAE =
{metrics['piso']['mae_mean']:,.0f} ± {metrics['piso']['mae_std']:,.0f} $/m².

**Cómo se armó la validación.** El mismo departamento puede estar publicado
en los dos portales; si una copia cae en entrenamiento y la otra en
validación, el modelo "se copia". Por eso el corte de test y los folds de
validación cruzada agrupan por **firma de inmueble** (dirección normalizada
+ localidad + superficie), nunca al azar: {metrics['split']['pct_avisos_con_firma_compartida']:.0%}
de los avisos comparte firma con otro. Firmas compartidas entre train y
test: {metrics['split']['firmas_compartidas_train_test']}.

**La tabla de comparación**, con media y desvío de los 15 folds:
""".replace(",", "."))
        tabla_comp = pd.DataFrame(comp)[["modelo", "mae_mean", "mae_std", "mejora_vs_piso"]]
        tabla_comp["mejora_vs_piso"] = tabla_comp["mejora_vs_piso"].map("{:.1%}".format)
        tabla(tabla_comp.set_index("modelo"))

        st.markdown("**Fold contra fold, contra el elegido:**")
        tabla(pd.DataFrame(metrics["pares_contra_mejor"]).set_index("contra"))

        st.markdown(f"""
**Columnas.** De las 18 candidatas, salen {len(metrics['columnas_salen'])}
porque sacarlas no empeora el MAE en más de 5 $/m² o ayudan en menos de la
mitad de los folds: {", ".join(metrics['columnas_salen'])}. Quedan
{len(metrics['columnas_finales'])}: {", ".join(metrics['columnas_finales'])}.
Con todas las columnas el MAE era {metrics['mae_con_todas']:,.0f}; con las
finales, {metrics['mae_con_finales']:,.0f}.

**El ajuste del elegido.** Grilla sobre tasa de aprendizaje, cantidad de
árboles, tamaño de hoja y tipo de pérdida. Mejor combinación:
{metrics['grid_mejor_params']}, con MAE {metrics['grid_mejor_mae']:,.0f}
(contra {metrics['mae_sin_ajustar_finales']:,.0f} sin ajustar: la grilla
ayuda poco, menos que el desvío entre folds).

**El resultado en test**, medido una sola vez: MAE
**{test['mae_modelo']:,.0f} $/m²** contra **{test['mae_piso']:,.0f}** del
piso ({test['mejora_pct']:.0%} menos error). Error relativo mediano:
{test['error_relativo_mediano_modelo']:.1%} contra
{test['error_relativo_mediano_piso']:.1%} del piso. En validación cruzada
el elegido daba {test['mae_cv_elegido']:,.0f}: muy cerca del test, la
validación no estaba mintiendo.

**Dónde se equivoca**, por tramo de precio real:
""".replace(",", "."))
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
**La superficie no es fuga.** Existe al predecir (la publica el aviso), no
contiene la respuesta (pendiente log-log de {fuga['pendiente_log_precio_vs_log_superficie']}:
el precio sube menos que proporcional a la superficie, no la reconstruye) y
sin ella el modelo sigue ganándole al piso ({fuga['mae_sin_superficie']:,.0f}
contra {fuga['mae_piso']:,.0f}).
""".replace(",", "."))

    st.markdown('<p class="bloque-cierra">Cierra la Entrega 3.</p>', unsafe_allow_html=True)


# =============================================================================
# EXPLORA (copa)
# =============================================================================
def explora():
    st.header("Explorá los avisos")
    st.markdown(
        "Empezá por el panorama: todos los avisos modelados, precio por m² "
        "contra superficie. Después achicá el universo con los controles, y "
        "pasá el mouse sobre un aviso para ver qué dijo el modelo."
    )

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        localidades = sorted(explo["localidad_agrupada"].dropna().unique())
        sel_localidad = st.multiselect("Localidad", localidades, default=localidades)
    with col_f2:
        sup_min, sup_max = float(explo["superficie_cubierta_m2"].min()), float(explo["superficie_cubierta_m2"].quantile(0.99))
        rango_sup = st.slider("Superficie cubierta (m²)", 0.0, float(sup_max), (0.0, float(sup_max)))
    with col_f3:
        monedas = sorted(explo["moneda"].dropna().unique())
        sel_moneda = st.multiselect("Moneda", monedas, default=monedas)

    filtrado = explo[
        explo["localidad_agrupada"].isin(sel_localidad)
        & explo["superficie_cubierta_m2"].between(rango_sup[0], rango_sup[1])
        & explo["moneda"].isin(sel_moneda)
    ]
    st.caption(f"{len(filtrado):,} avisos en el filtro actual, de {len(explo):,} totales.".replace(",", "."))

    chart = alt.Chart(filtrado).mark_circle(size=35, opacity=0.5).encode(
        x=alt.X("superficie_cubierta_m2:Q", scale=alt.Scale(type="log"), title="superficie cubierta (m², log)"),
        y=alt.Y("precio_m2:Q", scale=alt.Scale(type="log"), title="precio_m2 ($, log)"),
        color=alt.Color("etiqueta:N", scale=ESCALA_ETIQUETA, legend=alt.Legend(title="Veredicto del modelo")),
        tooltip=[
            alt.Tooltip("direccion:N", title="Dirección"),
            alt.Tooltip("localidad:N", title="Localidad"),
            alt.Tooltip("superficie_cubierta_m2:Q", title="Superficie (m²)"),
            alt.Tooltip("dormitorios:Q", title="Dormitorios"),
            alt.Tooltip("precio:Q", title="Precio publicado", format=",.0f"),
            alt.Tooltip("precio_m2:Q", title="precio_m2 real", format=",.0f"),
            alt.Tooltip("precio_m2_predicho:Q", title="precio_m2 estimado por el modelo", format=",.0f"),
            alt.Tooltip("residuo_relativo:Q", title="Diferencia vs el modelo", format="+.1%"),
            alt.Tooltip("etiqueta:N", title="Veredicto"),
        ],
    ).properties(height=480).interactive()
    st.altair_chart(chart, width="stretch")

    st.caption(
        "Cada aviso lo predice un modelo que no lo vio entrenar con ese dato "
        "(predicción fuera de fold o sobre el test). El veredicto sale de "
        f"comparar el residuo relativo contra los cortes {metrics['cortes_residuo']['bajo']:+.1%} / "
        f"{metrics['cortes_residuo']['alto']:+.1%} (percentiles 10 y 90 del error en validación)."
    )


# =============================================================================
# PROBA EL MODELO (copa)
# =============================================================================
def proba_el_modelo():
    st.header("Probá el modelo")
    st.markdown(
        "Armá un aviso con sus atributos reales y mirá qué precio por m² "
        "estimaría el modelo. Los controles van del mínimo al máximo que vio "
        "el entrenamiento, así que no se puede cargar un valor que el modelo "
        "nunca conoció."
    )

    finales = metrics["columnas_finales"]
    valores = {}
    cols_ui = st.columns(2)
    i = 0
    for col in finales:
        info = rangos.get(col)
        if info is None:
            continue
        etiqueta = ETIQUETAS_COLUMNAS.get(col, col)
        destino = cols_ui[i % 2]
        i += 1
        with destino:
            if info["tipo"] == "numerica":
                valores[col] = st.slider(etiqueta, min_value=round(info["min"], 1), max_value=round(info["max"], 1),
                                          value=round(info["mediana"], 1))
            elif info["tipo"] == "categorica":
                valores[col] = st.selectbox(etiqueta, info["categorias"])
            elif info["tipo"] == "fecha":
                valores[col] = info["max"]  # fecha de scraping: se fija a la corrida mas reciente, no la carga el usuario

    precio_pedido = st.number_input(
        "Opcional: ¿cuánto pide de verdad este aviso por mes? (para calcular el veredicto)",
        min_value=0.0, value=0.0, step=1000.0,
    )

    if st.button("Predecir", type="primary"):
        fila = pd.DataFrame([valores])
        for col in finales:
            if col not in fila.columns:
                fila[col] = np.nan
        pred = float(modelo.predict(fila)[0])
        st.success(f"El modelo estima **{pred:,.0f} $/m²** para este aviso.".replace(",", "."))

        test = metrics["test"]
        st.caption(f"Error típico del modelo en test: ±{test['mae_modelo']:,.0f} $/m² "
                   f"({test['error_relativo_mediano_modelo']:.0%}).".replace(",", "."))

        superficie = valores.get("superficie_cubierta_m2")
        if precio_pedido > 0 and superficie:
            precio_m2_real = precio_pedido / superficie
            residuo = (precio_m2_real - pred) / pred
            corte_bajo, corte_alto = metrics["cortes_residuo"]["bajo"], metrics["cortes_residuo"]["alto"]
            if residuo > corte_alto:
                veredicto, color = "Sobrevalorado", COLOR_SOBREVALORADO
            elif residuo < corte_bajo:
                veredicto, color = "Oportunidad", COLOR_OPORTUNIDAD
            else:
                veredicto, color = "En línea con el mercado", COLOR_EN_LINEA
            st.markdown(
                f"Pide **{precio_m2_real:,.0f} $/m²** ({residuo:+.0%} contra lo esperado): "
                f"<span style='color:{color}; font-weight:700'>{veredicto}</span>".replace(",", "."),
                unsafe_allow_html=True,
            )

        for col, val in valores.items():
            info = rangos.get(col, {})
            if info.get("tipo") == "numerica" and not (info["min"] <= val <= info["max"]):
                st.warning(f"{ETIQUETAS_COLUMNAS.get(col, col)} = {val} está fuera de lo que vio el entrenamiento "
                           f"({info['min']}–{info['max']}). La predicción extrapola.")


PAGINAS = {
    "El proyecto": portada,
    "Los datos": los_datos,
    "Lo que encontramos": lo_que_encontramos,
    "El modelo": el_modelo,
    "Explora": explora,
    "Proba el modelo": proba_el_modelo,
}
PAGINAS[seccion]()
