"""Colores compartidos para categorías que aparecen en distintos gráficos."""
import altair as alt

COLORES_LOCALIDAD = {
    "Capital": "#4f86bd", "Godoy Cruz": "#c87938", "Guaymallen": "#9173aa",
    "Las Heras": "#bf6154", "Lujan de Cuyo": "#4d9486", "Maipu": "#b49a45",
    "General alvear": "#7e9451", "Rivadavia": "#a76442", "San Martin": "#687db5",
    "San Rafael": "#997b67", "Tunuyan": "#6d9698",
}
COLORES_PORTAL = {"inmoclick": "#4f86bd", "inmoup": "#c87938"}
COLORES_MONEDA = {"ARS": "#4f86bd", "USD": "#c87938"}
COLORES_COMPARACION = {"Sobrevalorado": "#c85c50", "En linea": "#4f86bd", "Oportunidad": "#4d9a70"}
COLORES_MODELO = {"P0": "#9173aa", "P1": "#b49a45", "P3": "#c87938", "Lineal": "#bf6154", "Lineal log(y)": "#4d9486", "Random Forest": "#7e9451", "HistGradientBoosting": "#4f86bd"}
COLORES_OTROS = ["#4f86bd", "#c87938", "#9173aa", "#4d9486", "#bf6154", "#b49a45", "#7e9451", "#a76442", "#687db5", "#997b67", "#6d9698"]


# Resaltar y apagar (Clase 10): lo que el titulo nombra va en color; el resto,
# en gris. Sigue estando, pero deja de competir por la atencion.
COLOR_FOCO = "#2a78d6"
COLOR_FOCO_2 = "#c87938"
GRIS_APAGADO = "#c3c2b7"


def escala(colores):
    return alt.Scale(domain=list(colores), range=list(colores.values()))


def resaltar(valores, focos):
    """Escala que pinta solo los valores que el titulo nombra.

    `focos` es una lista (todos con el color de foco) o un dict {valor: color}
    cuando el titulo compara dos cosas distintas.
    """
    if not isinstance(focos, dict):
        focos = {valor: COLOR_FOCO for valor in focos}
    dominio = [str(valor) for valor in valores]
    focos = {str(clave): color for clave, color in focos.items()}
    return alt.Scale(domain=dominio, range=[focos.get(valor, GRIS_APAGADO) for valor in dominio])


# Nombre que ve el usuario para cada veredicto. La clave interna del dataset
# sigue siendo "En linea"; en pantalla decimos "Precio acorde" porque describe
# lo que significa: el aviso pide lo mismo que sus equivalentes.
NOMBRE_COMPARACION = {"Sobrevalorado": "Sobrevalorado", "En linea": "Precio acorde",
                      "Oportunidad": "Oportunidad"}
ORDEN_COMPARACION = list(NOMBRE_COMPARACION.values())
ESCALA_COMPARACION = alt.Scale(
    domain=ORDEN_COMPARACION,
    range=[COLORES_COMPARACION[clave] for clave in NOMBRE_COMPARACION],
)


def mostrar(grafico, **kwargs):
    """st.altair_chart con la misma tipografia en todos los graficos.

    Letra mas grande que la de Vega por defecto (11 px) para que ejes,
    leyendas y titulos se lean en una proyeccion o en una pantalla chica.
    """
    import streamlit as st

    grafico = (
        grafico.configure_axis(labelFontSize=13, titleFontSize=14, titlePadding=10, labelLimit=260)
        .configure_legend(labelFontSize=13, titleFontSize=14, symbolSize=140, labelLimit=260)
        .configure_title(fontSize=16, anchor="start", offset=12)
        .configure_header(labelFontSize=13, titleFontSize=14)
    )
    return st.altair_chart(grafico, **kwargs)
