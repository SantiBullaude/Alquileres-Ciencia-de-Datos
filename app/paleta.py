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


def escala(colores):
    return alt.Scale(domain=list(colores), range=list(colores.values()))
