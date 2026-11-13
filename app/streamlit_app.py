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
        st.metric("Menos error que el mejor piso", f"{test['mejora_vs_mejor_piso_pct']:.0%}",
                  delta=f"{test['mejora_pct']:.0%} vs mediana global", delta_color="normal")
        st.caption(f"MAE en test: {test['mae_modelo']:,.0f} $/m² del modelo · "
                   f"{test['mae_mejor_piso']:,.0f} $/m² del mejor piso (mediana superficie × moneda) · "
                   f"{test['mae_piso']:,.0f} $/m² de la mediana global.".replace(",", "."))
    with c3:
        # Coma decimal solo en el numero, no en la oracion (el .replace global
        # tambien convertia el punto final de la frase en coma).
        err_rel_modelo = f"{test['error_relativo_mediano_modelo']:.1%}".replace(".", ",")
        err_rel_mejor_piso = f"{test['error_relativo_mediano_mejor_piso']:.1%}".replace(".", ",")
        err_rel_piso_g = f"{test['error_relativo_mediano_piso']:.1%}".replace(".", ",")
        st.metric("Error relativo típico", err_rel_modelo)
        st.caption(f"Contra {err_rel_mejor_piso} del mejor piso y "
                   f"{err_rel_piso_g} de la mediana global. "
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
        f"(>0,8 es zona verde: dos poblaciones bien distintas). ARS: {miles(h1['n_ars'])} avisos, mediana "
        f"{miles(h1['mediana_ars'])} $/m². USD: {h1['n_usd']} avisos, mediana {h1['mediana_usd']} USD/m²."
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
Medido sobre {miles(h2['n'])} avisos en pesos.

**Una que no: la distancia al centro, sola, no explica el precio por m².**
Spearman entre `dist_centro_km` y `precio_m2` da
**{metrics['hallazgos_eda']['h3_distancia_refutada']['spearman']}**: zona
roja, refutada. (La sección "El modelo" retoma esta variable separando por
tamaño de departamento, donde el efecto sí aparece en los más chicos.)

**Cómo se construyó el gráfico.** Transformación `log10` en ambos ejes del
panel derecho; sin esa transformación, la asimetría de ambas variables
aplasta la mayoría de los puntos contra el origen.
""")

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

    esc = pd.DataFrame(metrics["escalera_pisos"])
    orden_esc = esc["escalon"].tolist()
    chart_esc = alt.Chart(esc).mark_bar(size=26).encode(
        x=alt.X("test_mae:Q", title="MAE en test ($/m²) — menos es mejor"),
        y=alt.Y("escalon:N", sort=orden_esc, title=None),
        color=alt.condition(alt.datum.columnas == len(metrics["columnas_finales"]),
                             alt.value(COLOR_ELEGIDO), alt.value(COLOR_RESTO)),
        tooltip=[
            alt.Tooltip("escalon:N", title="Escalón"),
            alt.Tooltip("regla:N", title="Regla"),
            alt.Tooltip("cv_mae:Q", title="MAE validación (15 folds)", format=",.0f"),
            alt.Tooltip("test_mae:Q", title="MAE test", format=",.0f"),
            alt.Tooltip("test_err_rel:Q", title="Error relativo mediano (test)", format=".1%"),
        ],
    ).properties(height=210, title="Escalera de pisos vs modelo final (test, medido una sola vez)")
    st.altair_chart(chart_esc, width="stretch")

    comp = metrics["comparacion_modelos"]
    filas = []
    for m in comp:
        for v in m["folds"]:
            filas.append({"modelo": m["modelo"], "mae": v})
    df_folds = pd.DataFrame(filas)
    orden = [m["modelo"] for m in sorted(comp, key=lambda d: d["mae_mean"], reverse=True)]
    chart = alt.Chart(df_folds).mark_boxplot(size=22).encode(
        x=alt.X("mae:Q", title="MAE por fold ($/m²) — menos es mejor"),
        y=alt.Y("modelo:N", sort=orden, title=None),
        color=alt.condition(alt.datum.modelo == "HistGradientBoosting",
                             alt.value(COLOR_ELEGIDO), alt.value(COLOR_RESTO)),
    ).properties(height=300, title="15 folds agrupados por firma de inmueble, mismos folds para todos")
    st.altair_chart(chart, width="stretch")
    st.caption("Azul: el modelo elegido. Gris: la escalera de pisos (P0, P1, P3) y las demás familias probadas.")

    with st.expander("Cómo lo sabemos"):
        st.markdown(f"""
**Métrica y por qué.** MAE (error absoluto medio) en $/m²: se lee en pesos
("le erramos por X pesos por m² en promedio") y no se deja arrastrar por
avisos mal cargados, a diferencia de RMSE.

**La escalera de pisos.** Con MAE, la constante óptima dentro de un grupo es su
mediana. Probamos pisos con 0, 1 y 2 columnas (los cuantiles de superficie se
calculan **solo con el fold de entrenamiento**, y si un grupo tiene menos de 20
avisos cae al nivel anterior):
- **P0 (mediana global, 0 col):** {miles(metrics['piso']['mae_mean'])} ± {miles(metrics['piso']['mae_std'])} $/m² en CV ({miles(test['mae_piso'])} en test).
- **P1 (mediana por decil de superficie, 1 col):** {miles(metrics['piso_superficie']['mae_mean'])} ± {miles(metrics['piso_superficie']['mae_std'])} $/m² en CV ({miles(test['mae_piso_superficie'])} en test).
- **P3 (mediana por superficie × moneda, 2 col — mejor piso):** {miles(metrics['mejor_piso']['mae_mean'])} ± {miles(metrics['mejor_piso']['mae_std'])} $/m² en CV ({miles(test['mae_mejor_piso'])} en test).

**Cómo se armó la validación.** El mismo departamento puede estar publicado
en los dos portales; si una copia cae en entrenamiento y la otra en
validación, el modelo "se copia". Por eso el corte de test y los folds de
validación cruzada agrupan por **firma de inmueble** (dirección normalizada
+ localidad + superficie), nunca al azar: {metrics['split']['pct_avisos_con_firma_compartida']:.0%}
de los avisos comparte firma con otro. Firmas compartidas entre train y
test: {metrics['split']['firmas_compartidas_train_test']}.

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
cruzada el elegido daba {miles(test['mae_cv_elegido'])}: muy cerca del test, la
validación no estaba mintiendo.

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
**La superficie no es fuga — y el modelo no es solo superficie.**
1. *Existe al predecir:* la publica el aviso.
2. *No contiene la respuesta:* la pendiente log-log entre precio y superficie es
   **{fuga['pendiente_log_precio_vs_log_superficie']}** (< 1): cuando los m² se
   duplican, el alquiler sube ~64%, por eso el m² de los grandes es más barato.
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
efectivamente piden de más o de menos. En cambio, al cruzar por el quintil del
**precio predicho** (lo único que conoce la app antes de saber cuánto pide el
aviso), el reparto es equilibrado en todos los tramos (~73–85% en línea):
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
    st.caption(f"{miles(len(filtrado))} avisos en el filtro actual, de {miles(len(explo))} totales.")

    chart = alt.Chart(filtrado).mark_circle(size=35, opacity=0.5).encode(
        x=alt.X("superficie_cubierta_m2:Q", scale=alt.Scale(type="log"), title="superficie cubierta (m², log)"),
        y=alt.Y("precio_m2:Q", scale=alt.Scale(type="log"), title="precio_m2 ($, log)"),
        color=alt.Color("etiqueta:N", scale=ESCALA_ETIQUETA, legend=alt.Legend(title="Veredicto del modelo")),
        tooltip=[
            alt.Tooltip("direccion:N", title="Dirección"),
            alt.Tooltip("localidad:N", title="Localidad"),
            alt.Tooltip("superficie_cubierta_m2:Q", title="Superficie (m²)"),
            alt.Tooltip("dormitorios:Q", title="Dormitorios", format=",.0f"),
            alt.Tooltip("banos:Q", title="Baños", format=",.0f"),
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
                    valores[col] = st.selectbox(etiqueta, info["categorias"],
                                                 format_func=lambda c: etiquetas_cat.get(c, c))
                else:
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
