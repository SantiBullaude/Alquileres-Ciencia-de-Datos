"""Datos y gráficos para la historia de la Entrega 4."""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import r2_score
from paleta import (COLORES_LOCALIDAD, COLORES_PORTAL, COLORES_OTROS, COLORES_COMPARACION,
                    COLOR_FOCO, COLOR_FOCO_2, GRIS_APAGADO, NOMBRE_COMPARACION, escala, mostrar,
                    resaltar)

DATA = Path(__file__).resolve().parent / "data"


def miles(valor):
    return f"{valor:,.0f}".replace(",", ".")


def porcentaje(valor):
    return f"{valor:+.1%}".replace(".", ",").replace("%", " %")


def dec(valor, decimales=1):
    """Numero decimal con coma (es-AR), para veces/correlaciones."""
    return f"{valor:.{decimales}f}".replace(".", ",")


def pct(valor, decimales=0):
    """Porcentaje es-AR (coma decimal y espacio antes del signo), sin signo mas."""
    return f"{valor:.{decimales}%}".replace(".", ",").replace("%", " %")


ACENTOS = {"Guaymallen": "Guaymallén", "Lujan de Cuyo": "Luján de Cuyo", "Maipu": "Maipú",
           "Las Heras": "Las Heras", "San Martin": "San Martín", "Tunuyan": "Tunuyán",
           "General alvear": "General Alvear"}


TOTAL_PASOS = 6


def paso(numero, texto):
    """Ubica la seccion dentro de la historia: que paso es y que le aporta a la pregunta."""
    st.caption(f"Paso {numero} de {TOTAL_PASOS} hacia la respuesta · {texto}")


def para_la_pregunta(texto):
    """Cierre de cada hallazgo: por que lo mostramos y que hace con eso el resto de la historia."""
    st.markdown(f"> **Qué aporta a la pregunta.** {texto}")


def nombre(localidad):
    """Nombre de localidad con acento para los textos (el dato crudo viene sin tildes)."""
    return ACENTOS.get(str(localidad), str(localidad))


def comparables(datos, localidad, moneda, superficie, dormitorios, excluir=None):
    """Misma localidad, moneda y dormitorios; superficie con tolerancia del 20 %."""
    if not superficie or pd.isna(dormitorios):
        return datos.iloc[0:0].copy()
    mascara = (datos["localidad"].eq(localidad) & datos["moneda"].eq(moneda)
               & datos["superficie_cubierta_m2"].between(superficie * .8, superficie * 1.2)
               & datos["dormitorios"].eq(dormitorios))
    if excluir is not None:
        mascara &= datos["clave"].ne(excluir)
    resultado = datos.loc[mascara].copy()
    resultado["distancia_superficie"] = (resultado["superficie_cubierta_m2"] - superficie).abs()
    return resultado.sort_values("distancia_superficie")


def mostrar_comparables(grupo):
    if grupo.empty:
        st.info("No encontramos avisos con estas características en la muestra.")
        return
    st.caption(f"Encontramos {miles(len(grupo))} avisos de la misma localidad, moneda y cantidad de dormitorios, con superficie hasta 20 % menor o mayor. Mostramos los ocho más próximos en superficie.")
    vista = grupo.assign(precio_ars=grupo.precio_m2 * grupo.superficie_cubierta_m2)[
        ["localidad", "superficie_cubierta_m2", "dormitorios", "precio_ars", "precio_m2", "fuente", "url"]
    ].head(8).rename(columns={"localidad": "Localidad", "superficie_cubierta_m2": "m²",
                               "dormitorios": "Dormitorios", "precio_ars": "Precio mensual en pesos",
                               "precio_m2": "Precio por m²", "fuente": "Portal", "url": "Aviso"})
    st.dataframe(vista, hide_index=True, width="stretch", column_config={
        "Precio mensual en pesos": st.column_config.NumberColumn(format="$ %.0f"),
        "Precio por m²": st.column_config.NumberColumn(format="$ %.0f"),
        "Aviso": st.column_config.LinkColumn("Aviso", display_text="Abrir")})
    st.caption("La comparación controla localidad, moneda, dormitorios y superficie (±20 %). Un mismo departamento puede estar publicado en ambos portales.")
    if len(grupo) < 10:
        st.warning("Hay menos de diez avisos comparables. Usá la mediana con cuidado.")


def grafico_comparacion_aviso(publicado, predicho, mediana_comp=None, n_comp=0, veredicto="En linea"):
    """Barra horizontal que compara precio publicado, referencia del modelo y mediana de comparables."""
    color_aviso = COLORES_COMPARACION.get(veredicto, COLORES_COMPARACION["En linea"])
    filas = [
        {"referencia": "1. Precio publicado", "valor": float(publicado), "color": color_aviso},
        {"referencia": "2. Referencia del modelo", "valor": float(predicho), "color": "#4f86bd"},
    ]
    if mediana_comp is not None and pd.notna(mediana_comp) and n_comp > 0:
        filas.append({
            "referencia": f"3. Mediana de {miles(n_comp)} comparables",
            "valor": float(mediana_comp),
            "color": "#7e9451",
        })
    df_comp = pd.DataFrame(filas)
    orden = df_comp["referencia"].tolist()
    escala_col = alt.Scale(domain=orden, range=df_comp["color"].tolist())
    barras = alt.Chart(df_comp).mark_bar(height=22).encode(
        x=alt.X("valor:Q", title="Precio por m² ($/m²)"),
        y=alt.Y("referencia:N", sort=orden, title=None, axis=alt.Axis(labelLimit=210)),
        color=alt.Color("referencia:N", scale=escala_col, legend=None),
        tooltip=[
            alt.Tooltip("referencia:N", title="Referencia"),
            alt.Tooltip("valor:Q", title="Precio ($/m²)", format=",.0f"),
        ],
    ).properties(height=38 * len(filas) + 25)
    return barras


def nivel_de_precios(datos, metrics, escala_moneda, escala_portal):
    """Cuanto se pide hoy, y cuanto cambia el precio segun el portal y la moneda."""
    mensual = datos.precio_m2 * datos.superficie_cubierta_m2
    p25, mediana, p75 = mensual.quantile(.25), mensual.median(), mensual.quantile(.75)
    st.subheader("La mayoría de los alquileres se mueve en una banda de precios estrecha")
    st.markdown(
        f"La mitad de los avisos pide entre **${miles(p25)}** y **${miles(p75)}** por mes. "
        f"Antes de juzgar un aviso hay que saber contra qué se lo compara: en la ventana relevada, el alquiler "
        f"mensual mediano de un departamento en el Gran Mendoza es de **${miles(mediana)}** "
        f"(**${miles(datos.precio_m2.median())}/m²**), con el 10 % más barato por debajo de "
        f"**${miles(mensual.quantile(.1))}** y el 10 % más caro por encima de **${miles(mensual.quantile(.9))}**."
    )
    histograma = alt.Chart(datos.assign(mensual=mensual).query("mensual <= @mensual.quantile(0.99)")).mark_bar(
        color=COLORES_PORTAL["inmoclick"]).encode(
        x=alt.X("mensual:Q", bin=alt.Bin(maxbins=40), title="Alquiler mensual publicado ($)"),
        y=alt.Y("count():Q", title="Cantidad de avisos"),
        tooltip=[alt.Tooltip("count():Q", title="Avisos")],
    ).properties(height=300, title=["Distribución del alquiler mensual", "(sin el 1 % más caro)"])
    regla = alt.Chart(pd.DataFrame({"mediana": [float(mediana)]})).mark_rule(
        color="#d03b3b", strokeDash=[5, 5], size=2).encode(x="mediana:Q")

    portales = datos.groupby("fuente", as_index=False).agg(
        avisos=("clave", "size"), mediana_m2=("precio_m2", "median"),
        superficie=("superficie_cubierta_m2", "median"))
    portales["mediana_mensual"] = [
        float((datos.loc[datos.fuente.eq(f), "precio_m2"] * datos.loc[datos.fuente.eq(f), "superficie_cubierta_m2"]).median())
        for f in portales.fuente]
    base_portales = alt.Chart(portales).encode(
        x=alt.X("mediana_m2:Q", title="Mediana del precio publicado ($/m²)"),
        y=alt.Y("fuente:N", title=None, sort="-x", axis=alt.Axis(labelLimit=120)),
    )
    grafico_portales = (
        base_portales.mark_bar().encode(
            color=alt.Color("fuente:N", scale=escala_portal, legend=None),
            tooltip=[alt.Tooltip("fuente:N", title="Portal"), alt.Tooltip("avisos:Q", title="Avisos"),
                     alt.Tooltip("mediana_m2:Q", title="Mediana ($/m²)", format=",.0f"),
                     alt.Tooltip("mediana_mensual:Q", title="Mediana mensual ($)", format=",.0f"),
                     alt.Tooltip("superficie:Q", title="Superficie mediana (m²)", format=",.0f")])
        + base_portales.mark_text(align="right", dx=-8, color="white", fontWeight="bold", fontSize=14).encode(
            text=alt.Text("mediana_m2:Q", format=",.0f"))
    ).properties(height=alt.Step(60), title=["Precio por m² según el portal", "donde se publica"])

    c_hist, c_portal = st.columns([1.2, 1])
    with c_hist:
        mostrar(histograma + regla, width="stretch")
    with c_portal:
        mostrar(grafico_portales, width="stretch")

    caro = portales.sort_values("mediana_m2", ascending=False).iloc[0]
    barato = portales.sort_values("mediana_m2", ascending=False).iloc[-1]
    brecha_portal = caro.mediana_m2 / barato.mediana_m2 - 1
    usd, ars = datos[datos.moneda.eq("USD")], datos[datos.moneda.eq("ARS")]
    st.markdown(
        f"**Sí cambia según dónde se publique, pero poco:** la mediana de **{caro.fuente}** "
        f"(${miles(caro.mediana_m2)}/m²) está un {pct(brecha_portal, 1)} por encima de la de **{barato.fuente}** "
        f"(${miles(barato.mediana_m2)}/m²). Es una brecha chica y en parte se explica por el tipo de aviso: en "
        f"{barato.fuente} los departamentos son más grandes (mediana de {miles(barato.superficie)} m² contra "
        f"{miles(caro.superficie)} m²) y rinden menos por metro. En alquiler mensual la relación se invierte "
        f"(${miles(barato.mediana_mensual)} contra ${miles(caro.mediana_mensual)}). "
        f"**La diferencia que sí importa es la moneda:** los {miles(len(usd))} avisos en dólares piden "
        f"{pct(usd.precio_m2.median() / ars.precio_m2.median() - 1)} más por m² que los "
        f"{miles(len(ars))} en pesos."
    )
    para_la_pregunta(
        "Para decir que un aviso está caro primero hay que saber cuánto se pide en general: esta banda es la vara "
        "inicial. Y como la moneda cambia mucho el precio por m² (el portal casi no), el modelo recibe la moneda "
        "como dato y los comparables se buscan siempre dentro de la misma moneda."
    )
    st.info(
        "**Lo que estos datos no pueden responder.** Con una ventana de cinco semanas "
        f"({metrics['dataset']['ventana_inicio']} a {metrics['dataset']['ventana_fin']}) y sin datos de ingresos "
        "ni una serie histórica de precios, no podemos afirmar si el mercado mendocino **en general** está caro o "
        "barato: no hay contra qué compararlo. Todo lo que sigue compara cada aviso **contra el resto del mercado "
        "en el mismo momento**, que es la pregunta que estos datos sí responden."
    )


def ubicacion(datos, escala_etiqueta, aporte_zona=None):
    tabla = (datos.groupby("localidad", as_index=False)
             .agg(avisos=("clave", "size"), mediana=("precio_m2", "median"),
                  arriba=("etiqueta", lambda s: s.eq("Sobrevalorado").mean()),
                  abajo=("etiqueta", lambda s: s.eq("Oportunidad").mean()))
             .query("avisos >= 50").sort_values("mediana", ascending=False))
    lider, ultima = tabla.iloc[0], tabla.iloc[-1]
    brecha_loc = lider.mediana / ultima.mediana - 1
    st.subheader(f"{nombre(lider.localidad)} lidera el precio por m² y {nombre(ultima.localidad)} "
                 "cierra la tabla")
    st.markdown(
        "El precio por metro cuadrado cambia fuerte según el departamento: la mediana va de "
        f"**${miles(ultima.mediana)}/m²** en {nombre(ultima.localidad)} a **${miles(lider.mediana)}/m²** en "
        f"{nombre(lider.localidad)}, un **{pct(brecha_loc)} más**. Pero la localidad sola no alcanza, porque "
        "dentro de una misma localidad conviven barrios muy distintos."
    )
    grafico = alt.Chart(tabla).mark_bar().encode(
        color=alt.Color("localidad:N", legend=None, scale=resaltar(
            tabla.localidad, {lider.localidad: COLOR_FOCO, ultima.localidad: COLOR_FOCO_2})),
        x=alt.X("mediana:Q", title="Mediana del precio publicado ($/m²)"),
        y=alt.Y("localidad:N", title=None, sort="-x"),
        tooltip=["localidad:N", alt.Tooltip("avisos:Q", title="Avisos"),
                 alt.Tooltip("mediana:Q", title="Mediana ($/m²)", format=",.0f")]
    ).properties(height=280, title="Mediana por localidad (al menos 50 avisos)")

    mapa = json.loads((DATA / "departamentos_gran_mendoza.geojson").read_text(encoding="utf-8"))
    geo = datos.dropna(subset=["latitud", "longitud"]).copy()
    zona = geo[geo.longitud.between(-69.1, -68.5) & geo.latitud.between(-33.17, -32.62)].copy()
    zona["orden_mapa"] = zona.etiqueta.ne("En linea").astype(int)
    # Encuadre: el rectangulo donde cae el 98 % de los avisos (sin los extremos), no los
    # departamentos enteros. Lujan y Las Heras llegan a la cordillera y dejaban los puntos
    # amontonados en una esquina del mapa.
    oeste, este = zona.longitud.quantile([.01, .99])
    sur, norte = zona.latitud.quantile([.01, .99])
    encuadre = {"type": "Feature", "properties": {}, "geometry": {"type": "MultiPoint", "coordinates": [
        [float(oeste), float(sur)], [float(este), float(norte)]]}}
    limites = alt.Chart(alt.Data(values=mapa["features"])).mark_geoshape(
        fill="#e8e8e4", stroke="#a5aaa5", strokeWidth=0.8, clip=True).encode(
            tooltip=[alt.Tooltip("properties.departamento:N", title="Departamento")])
    puntos = alt.Chart(zona).mark_circle(size=38, opacity=.75, stroke="white", strokeWidth=0.4, clip=True).encode(
        longitude="longitud:Q", latitude="latitud:Q",
        color=alt.Color("comparacion:N", scale=escala_etiqueta, title="Comparación",
                        legend=alt.Legend(orient="bottom", direction="horizontal")),
        # los que se apartan van arriba, para que no los tape la masa de avisos con precio acorde
        order=alt.Order("orden_mapa:Q"),
        tooltip=[alt.Tooltip("localidad:N", title="Localidad"),
                 alt.Tooltip("comparacion:N", title="Comparación"),
                 alt.Tooltip("precio_m2:Q", title="Publicado ($/m²)", format=",.0f"),
                 alt.Tooltip("precio_m2_predicho:Q", title="Referencia ($/m²)", format=",.0f"),
                 alt.Tooltip("residuo_relativo:Q", title="Diferencia", format="+.1%")])
    mapa_chart = (limites + puntos).project(type="mercator", fit=encuadre).properties(
        height=420, title="Gran Mendoza: dónde cae cada aviso"
    )

    c_bar, c_map = st.columns([1, 1.25])
    with c_bar:
        mostrar(grafico, width="stretch")
        st.caption(
            f"Localidades con al menos 50 avisos. {nombre(lider.localidad)} (${miles(lider.mediana)}/m²) supera "
            f"en un {pct(brecha_loc)} a {nombre(ultima.localidad)} (${miles(ultima.mediana)}/m²)."
        )
    with c_map:
        mostrar(mapa_chart, width="stretch")
        st.caption(f"{miles(len(zona))} avisos geolocalizados, coloreados según su diferencia con el modelo. "
                   "Límites: [departamentos de Argentina](https://github.com/mgaitan/departamentos_argentina). Los puntos muestran cómo se distribuyen los avisos con precio acorde, sobrevalorados y oportunidades; "
                   "los que se apartan se dibujan encima. El encuadre muestra la zona donde está el 98 % de los avisos.")

    zonas(datos, aporte_zona)


def zonas(datos, aporte_zona=None):
    """Por que el proyecto agrupa coordenadas en zonas, con la evidencia que lo respalda."""
    por_zona = (datos.dropna(subset=["zona_geo"]).groupby("zona_geo", as_index=False)
                .agg(avisos=("clave", "size"), mediana=("precio_m2", "median"),
                     distancia=("dist_centro_km", "median"),
                     principal=("localidad", lambda s: nombre(s.value_counts().index[0]))))
    por_zona = por_zona.query("avisos >= 30").sort_values("mediana", ascending=False)
    brecha_zonas = por_zona.mediana.max() / por_zona.mediana.min() - 1
    partidas = datos.dropna(subset=["zona_geo"]).groupby("localidad").zona_geo.nunique()
    repartidas = int((partidas > 1).sum())

    st.subheader("Por eso agrupamos en zonas: una localidad no es un mercado homogéneo")
    st.markdown(
        f"Entre la zona más barata y la más cara hay un **{pct(brecha_zonas)}** de diferencia. "
        f"Un aviso trae latitud y longitud, pero esas coordenadas sueltas no le dicen nada al modelo. "
        f"Agrupamos los avisos con **KMeans en 6 zonas** dentro de los 30 km de Plaza Independencia y dejamos "
        f"el interior provincial como una zona aparte. La decisión se justifica con dos hechos: "
        f"**{repartidas} de las {len(partidas)} localidades se reparten en más de una zona** (los avisos de una "
        f"misma localidad no comparten precio), y entre zonas la mediana va de "
        f"**${miles(por_zona.mediana.min())}/m²** a **${miles(por_zona.mediana.max())}/m²**."
    )
    etiquetas = por_zona.assign(
        zona=lambda t: t.apply(
            lambda f: ("Interior provincial" if f.zona_geo == -1
                       else f"Zona {int(f.zona_geo)} · {f.principal} ({miles(f.distancia)} km)"), axis=1))
    grafico_zonas = alt.Chart(etiquetas).mark_bar().encode(
        x=alt.X("mediana:Q", title="Mediana del precio publicado ($/m²)"),
        y=alt.Y("zona:N", title=None, sort="-x", axis=alt.Axis(labelLimit=260)),
        color=alt.Color("zona:N", legend=None, scale=resaltar(
            etiquetas.zona, {etiquetas.zona.iloc[0]: COLOR_FOCO, etiquetas.zona.iloc[-1]: COLOR_FOCO_2})),
        tooltip=[alt.Tooltip("zona:N", title="Zona"), alt.Tooltip("avisos:Q", title="Avisos"),
                 alt.Tooltip("distancia:Q", title="Distancia mediana al centro (km)", format=".1f"),
                 alt.Tooltip("mediana:Q", title="Mediana ($/m²)", format=",.0f")],
    ).properties(height=30 * len(etiquetas) + 60, title="Mediana por zona geográfica (zonas con al menos 30 avisos)")
    mostrar(grafico_zonas, width="stretch")
    para_la_pregunta(
        "La segunda mitad de la pregunta es **dónde** se concentran los desfases. Para responderla necesitamos "
        "saber cuánto cambia el precio entre localidades y entre zonas: un aviso de Chacras no se puede juzgar con "
        "la vara de Las Heras. Por eso el modelo recibe la localidad, la zona, la distancia y la densidad de avisos "
        "vecinos, y los comparables se buscan siempre en la misma localidad."
    )
    if aporte_zona is not None:
        st.caption(
            f"Honestidad sobre el aporte: una vez que el modelo ya tiene la distancia al centro y la densidad "
            f"de avisos vecinos, la columna `zona_geo` solo suma ${miles(aporte_zona)}/m² de error evitado "
            "(lo medimos por permutación, en «El modelo»). Las zonas explican el mapa y separan el interior "
            "provincial, pero el peso geográfico del modelo lo cargan la distancia y la densidad."
        )


def ajuste(datos, test_metrics):
    prueba = datos[datos.particion.eq("test")].copy()
    limite = max(prueba.precio_m2.max(), prueba.precio_m2_predicho.max())
    puntos = alt.Chart(prueba).mark_circle(size=42, opacity=.55, color="#2a78d6", stroke="white",
                                           strokeWidth=0.4).encode(
        x=alt.X("precio_m2:Q", title="Precio publicado ($/m²)", scale=alt.Scale(domain=[0, limite])),
        y=alt.Y("precio_m2_predicho:Q", title="Precio estimado por el modelo ($/m²)", scale=alt.Scale(domain=[0, limite])),
        tooltip=[alt.Tooltip("localidad:N", title="Localidad"),
                 alt.Tooltip("precio_m2:Q", title="Publicado ($/m²)", format=",.0f"),
                 alt.Tooltip("precio_m2_predicho:Q", title="Estimado ($/m²)", format=",.0f"),
                 alt.Tooltip("residuo_relativo:Q", title="Diferencia", format="+.1%")])
    diagonal = alt.Chart(pd.DataFrame({"x": [0, limite], "y": [0, limite]})).mark_line(
        color="#d03b3b", strokeDash=[5, 5]).encode(x="x:Q", y="y:Q")

    tramos = pd.qcut(prueba.precio_m2, 5, labels=["Q1 (más barato)", "Q2", "Q3 (central)", "Q4", "Q5 (más caro)"], duplicates="drop")
    prueba["quintil_real"] = tramos
    prueba["error_absoluto"] = (prueba.precio_m2 - prueba.precio_m2_predicho).abs()
    error_por_tramo = prueba.groupby("quintil_real", observed=True).agg(
        mae=("error_absoluto", "mean"),
        mediana_real=("precio_m2", "median"),
        avisos=("clave", "size"),
    ).reset_index()

    mae_central = float(error_por_tramo.loc[error_por_tramo["quintil_real"] == "Q3 (central)", "mae"].iloc[0])
    mae_caro = float(error_por_tramo.loc[error_por_tramo["quintil_real"] == "Q5 (más caro)", "mae"].iloc[0])
    veces = mae_caro / mae_central
    st.subheader("El ajuste es firme en el tramo central y se abre en el extremo caro")
    st.markdown(
        "El modelo se entrena para representar a la gran mayoría de los alquileres estándar: sobre la diagonal "
        "roja el precio publicado coincide con la estimación. En los **quintiles centrales (Q2 a Q4)** el error "
        f"absoluto medio baja hasta **${miles(mae_central)}/m²**, mientras que en el **quintil más caro (Q5)** "
        f"sube a **${miles(mae_caro)}/m²**: ahí el modelo es **{dec(veces)} veces menos preciso**, porque "
        "aparecen departamentos con lujos no estructurados o precios muy por encima de su zona."
    )

    orden_q = ["Q1 (más barato)", "Q2", "Q3 (central)", "Q4", "Q5 (más caro)"]
    colores_q = resaltar(orden_q, {"Q3 (central)": COLOR_FOCO,
                                   "Q5 (más caro)": COLORES_COMPARACION["Sobrevalorado"]})
    chart_quintiles = alt.Chart(error_por_tramo).mark_bar().encode(
        x=alt.X("mae:Q", title="Error absoluto medio — MAE ($/m²)"),
        y=alt.Y("quintil_real:N", sort=orden_q, title="Quintil del precio publicado"),
        color=alt.Color("quintil_real:N", scale=colores_q, legend=None),
        tooltip=[
            alt.Tooltip("quintil_real:N", title="Tramo"),
            alt.Tooltip("avisos:Q", title="Avisos en prueba"),
            alt.Tooltip("mediana_real:Q", title="Precio mediano ($/m²)", format=",.0f"),
            alt.Tooltip("mae:Q", title="MAE ($/m²)", format=",.0f"),
        ],
    ).properties(height=360, title="El ajuste es firme en el medio y se abre en el extremo caro")

    col_diag, col_q = st.columns([1.15, 1])
    with col_diag:
        mostrar(
            (puntos + diagonal).properties(
                height=360, title=f"Precio publicado vs. estimado ({miles(len(prueba))} avisos de prueba)"
            ).interactive(),
            width="stretch",
        )
    with col_q:
        mostrar(chart_quintiles, width="stretch")

    st.caption(
        f"{miles(len(prueba))} avisos de prueba. R²: {r2_score(prueba.precio_m2, prueba.precio_m2_predicho):.2f}; "
        f"error absoluto medio global: ${miles(test_metrics['mae_modelo'])}/m². "
        "En el tramo medio el modelo ofrece una referencia sólida; en los extremos señala avisos para auditar con comparables."
    )

    st.markdown("##### Por qué tomamos esta estimación como el precio objetivo del aviso")
    st.markdown(
        "Lo que llamamos *precio objetivo* no es nuestra opinión sobre cuánto debería valer un departamento: "
        "es **lo que pide el mercado por departamentos equivalentes**, calculado con cuatro reglas que nos "
        "sacan del medio.\n\n"
        f"1. **Ningún aviso participa de su propia estimación.** Cada punto del gráfico lo predijo un modelo "
        f"entrenado sin ese aviso (predicción fuera de fold o sobre el test), así que la referencia no puede "
        f"copiarse del precio que queremos juzgar.\n"
        "2. **El criterio es uno solo para todos.** Los mismos cortes y el mismo modelo se aplican a un "
        "monoambiente de Las Heras y a un piso de Chacras; no hay ajustes por caso.\n"
        "3. **Le ganamos a la referencia ingenua.** Un tasador que solo mirara la mediana de su tramo de "
        f"superficie y moneda erraría ${miles(test_metrics['mae_mejor_piso'])}/m²; el modelo erra "
        f"${miles(test_metrics['mae_modelo'])}/m² ({pct(test_metrics['mejora_vs_mejor_piso_pct'])} menos).\n"
        "4. **No está sesgado por tramo de precio.** Agrupando por el precio que el modelo estima —lo único "
        "que se conoce antes de ver cuánto pide el aviso— la diferencia mediana se mantiene cerca de cero en "
        "los cinco tramos (tabla de abajo). El modelo no marca «caro» sistemáticamente a los departamentos "
        "caros ni «barato» a los baratos.\n\n"
        "De ahí el argumento central del proyecto: si le dimos al modelo todo lo que el aviso publica y aun "
        "así el precio se aparta de la referencia, **lo que falta explicar está en el aviso, no en el modelo**."
    )

    prueba["tramo_estimado"] = pd.qcut(prueba.precio_m2_predicho, 5, labels=False, duplicates="drop") + 1
    tabla = prueba.groupby("tramo_estimado", observed=True).agg(
        avisos=("clave", "size"),
        diferencia_mediana=("residuo_relativo", "median"),
        mae=("error_absoluto", "mean"),
    ).reset_index()
    tabla["diferencia_mediana"] = tabla.diferencia_mediana.map(porcentaje)
    tabla["mae"] = tabla.mae.map(miles)
    with st.expander("Cómo lo sabemos · Ajuste según el tramo del precio estimado"):
        st.dataframe(
            tabla.rename(columns={
                "tramo_estimado": "Quintil del precio estimado",
                "avisos": "Avisos",
                "diferencia_mediana": "Diferencia relativa mediana",
                "mae": "MAE ($/m²)",
            }),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Al agrupar por el precio que estima el modelo (lo único observable antes de conocer cuánto pide el aviso), "
            "la diferencia mediana se mantiene cercana a cero en los cinco tramos."
        )


def caso_extremo(datos, lado, minimo=10, banda=(30, 130)):
    """Aviso de prueba mas extremo del lado pedido que tenga al menos `minimo` comparables."""
    prueba = datos[datos.particion.eq("test") & datos.superficie_cubierta_m2.between(*banda)
                   & datos.dormitorios.notna()]
    for _, aviso in prueba.sort_values("residuo_relativo", ascending=(lado == "Oportunidad")).iterrows():
        if aviso.etiqueta != lado:
            continue
        grupo = comparables(datos, aviso.localidad, aviso.moneda, aviso.superficie_cubierta_m2,
                            aviso.dormitorios, aviso.clave)
        if len(grupo) >= minimo:
            return aviso, grupo
    return None, None


def que_mueve_el_precio(datos, csv_origen=None, banda=(40, 90)):
    """Que caracteristicas suben y bajan la mediana del precio por m2, dentro de una banda de superficie."""
    muestra = datos[datos.superficie_cubierta_m2.between(*banda)].copy()
    texto = pd.Series("", index=muestra.index)
    if csv_origen is not None and Path(csv_origen).exists():
        extra = pd.read_csv(csv_origen, usecols=["clave", "descripcion"])
        texto = (muestra[["clave"]].merge(extra, on="clave", how="left")
                 .descripcion.fillna("").str.lower().set_axis(muestra.index))
    tiene_cochera = muestra.cochera.notna() & ~muestra.cochera.str.lower().isin(["sin cochera", "indistinto"])
    candidatos = {
        "Publicado en dólares": muestra.moneda.eq("USD"),
        "Se entrega amoblado o equipado": texto.str.contains(r"amoblad|equipad|muebles", regex=True),
        "A estrenar o nuevo": texto.str.contains(r"estrenar|nuevo|nueva", regex=True),
        "Tiene cochera": tiene_cochera,
        "Estado de conservación excelente": muestra.estado_conservacion.eq("Excelente"),
        "A menos de 2 km del centro": muestra.dist_centro_km.lt(2),
        "A más de 7 km del centro": muestra.dist_centro_km.gt(7),
        "Paga expensas": muestra.valor_expensas.fillna(0).gt(0),
        "Monoambiente o 1 dormitorio": muestra.dormitorios.le(1),
        "3 dormitorios o más": muestra.dormitorios.ge(3),
    }
    filas = []
    for etiqueta, mascara in candidatos.items():
        if mascara.sum() < 30 or (~mascara).sum() < 30:
            continue
        con, sin = muestra.loc[mascara, "precio_m2"].median(), muestra.loc[~mascara, "precio_m2"].median()
        filas.append({"factor": etiqueta, "efecto": float(con / sin - 1), "avisos": int(mascara.sum()),
                      "mediana_con": float(con), "mediana_sin": float(sin),
                      "sentido": "Sube el precio" if con >= sin else "Baja el precio"})
    if not filas:
        return
    efectos = pd.DataFrame(filas).sort_values("efecto", ascending=False)
    suben = efectos[efectos.efecto > 0].head(2)
    bajan = efectos[efectos.efecto < 0]

    st.subheader(f"Qué sube y qué baja el precio: {suben.factor.iloc[0].lower()} y "
                 f"{suben.factor.iloc[1].lower()} son los que más mueven la aguja")
    st.markdown(
        f"Suben la mediana un **{porcentaje(suben.efecto.iloc[0])}** y un "
        f"**{porcentaje(suben.efecto.iloc[1])}** respectivamente. "
        f"Comparando solo departamentos de {banda[0]} a {banda[1]} m² —para que la superficie no distorsione el "
        "precio por m²—, cada barra muestra cuánto cambia la mediana del precio por metro cuadrado cuando el aviso "
        "tiene esa característica frente a cuando no la tiene."
        + (f" El único factor que **baja** la mediana es **{bajan.factor.iloc[-1].lower()}** "
           f"({porcentaje(bajan.efecto.iloc[-1])}): los departamentos más grandes rinden menos por m²."
           if not bajan.empty else "")
    )
    grafico = alt.Chart(efectos).mark_bar().encode(
        x=alt.X("efecto:Q", title="Diferencia en la mediana del precio por m²", axis=alt.Axis(format="+%")),
        y=alt.Y("factor:N", sort="-x", title=None, axis=alt.Axis(labelLimit=250)),
        color=alt.Color("sentido:N", title=None, scale=alt.Scale(
            domain=["Sube el precio", "Baja el precio"],
            range=[COLORES_COMPARACION["Sobrevalorado"], COLORES_COMPARACION["Oportunidad"]])),
        tooltip=[alt.Tooltip("factor:N", title="Característica"),
                 alt.Tooltip("avisos:Q", title="Avisos con la característica"),
                 alt.Tooltip("mediana_con:Q", title="Mediana con ($/m²)", format=",.0f"),
                 alt.Tooltip("mediana_sin:Q", title="Mediana sin ($/m²)", format=",.0f"),
                 alt.Tooltip("efecto:Q", title="Diferencia", format="+.1%")],
    ).properties(height=30 * len(efectos) + 60, title="Efecto de cada característica sobre la mediana del precio por m²")
    mostrar(grafico, width="stretch")
    st.caption(
        f"Medido sobre {miles(len(muestra))} avisos de {banda[0]} a {banda[1]} m². Son diferencias de mediana entre "
        "grupos, no efectos aislados: las características aparecen juntas (un departamento a estrenar también suele "
        "tener cochera), así que las barras se superponen entre sí."
    )


def casos(datos, cortes, csv_origen=None):
    fuera = int((~datos.etiqueta.eq("En linea")).sum())
    st.header("Los casos: dónde y por qué algunos avisos se apartan de la referencia")
    paso(4, "aplicamos la referencia del modelo a cada aviso y vemos cuáles se apartan, dónde y por qué.")
    st.markdown(
        f"Son **{miles(fuera)} avisos** sobre {miles(len(datos))}. "
        "Para responder cuándo un alquiler está sobrevalorado o es una oportunidad, combinamos dos pasos: "
        f"primero identificamos los avisos que superan los cortes de validación (**{porcentaje(cortes['bajo'])}** "
        f"y **{porcentaje(cortes['alto'])}** frente al modelo) y después los contrastamos con **vecinos comparables** "
        "(misma localidad, moneda y dormitorios, con superficie dentro del ±20 %). Cuando ambas referencias coinciden, "
        "el desfase queda confirmado."
    )

    localidades = (
        datos.assign(arriba=datos.etiqueta.eq("Sobrevalorado"), abajo=datos.etiqueta.eq("Oportunidad"))
        .groupby("localidad", as_index=False)
        .agg(avisos=("clave", "size"), mediana=("precio_m2", "median"), arriba=("arriba", "mean"), abajo=("abajo", "mean"))
        .query("avisos >= 50")
        .sort_values("arriba", ascending=False)
    )
    primera, segunda = localidades.iloc[0], localidades.iloc[1]
    st.subheader(f"{nombre(primera.localidad)} y {nombre(segunda.localidad)} concentran la mayor "
                 "proporción de avisos sobrevalorados")
    porcentajes = localidades.melt(
        id_vars=["localidad", "avisos"], value_vars=["arriba", "abajo"], var_name="comparacion", value_name="proporcion"
    )
    porcentajes["comparacion"] = porcentajes["comparacion"].map({"arriba": "Por encima (Sobrevalorado)", "abajo": "Por debajo (Oportunidad)"})
    graf = alt.Chart(porcentajes).mark_bar().encode(
        x=alt.X("proporcion:Q", title="Proporción de avisos en la localidad", axis=alt.Axis(format="%")),
        y=alt.Y("localidad:N", sort=localidades.localidad.tolist(), title=None),
        yOffset=alt.YOffset("comparacion:N"),
        color=alt.Color(
            "comparacion:N",
            title="Respecto de la referencia",
            scale=alt.Scale(
                domain=["Por encima (Sobrevalorado)", "Por debajo (Oportunidad)"],
                range=[COLORES_COMPARACION["Sobrevalorado"], COLORES_COMPARACION["Oportunidad"]],
            ),
        ),
        opacity=alt.condition(
            alt.FieldOneOfPredicate(field="localidad",
                                    oneOf=[primera.localidad, segunda.localidad]),
            alt.value(1), alt.value(0.3)),
        tooltip=[
            alt.Tooltip("localidad:N", title="Localidad"),
            alt.Tooltip("avisos:Q", title="Avisos totales"),
            alt.Tooltip("comparacion:N", title="Grupo"),
            alt.Tooltip("proporcion:Q", title="Proporción", format=".1%"),
        ],
    ).properties(height=250)
    mostrar(graf, width="stretch")
    if not localidades.empty:
        homogeneas = localidades.assign(
            en_linea=1 - localidades.arriba - localidades.abajo).sort_values("en_linea", ascending=False)
        mas, segunda_mas = homogeneas.iloc[0], homogeneas.iloc[1]
        st.markdown(
            f"En **{nombre(primera.localidad)}** ({pct(primera.arriba, 1)} de {miles(primera.avisos)} avisos por "
            f"encima del corte) y **{nombre(segunda.localidad)}** ({pct(segunda.arriba, 1)}) conviven "
            "departamentos estándar con zonas de alto poder adquisitivo (Quinta y Sexta Sección, Chacras de "
            f"Coria, Dalvian). En mercados más homogéneos como **{nombre(mas.localidad)}** "
            f"({pct(mas.en_linea, 1)} con precio acorde) y **{nombre(segunda_mas.localidad)}** "
            f"({pct(segunda_mas.en_linea, 1)}), los precios se apartan mucho menos."
        )

    factores_sobre = [
        {"factor": "Publicado en USD (vs. ARS)", "grupo": "Con la característica", "proporcion": float(datos.loc[datos.moneda.eq("USD"), "etiqueta"].eq("Sobrevalorado").mean())},
        {"factor": "Publicado en USD (vs. ARS)", "grupo": "Resto de los avisos", "proporcion": float(datos.loc[datos.moneda.eq("ARS"), "etiqueta"].eq("Sobrevalorado").mean())},
    ]
    if csv_origen is not None and Path(csv_origen).exists():
        extra = pd.read_csv(csv_origen, usecols=["clave", "descripcion", "piscina", "amoblado"])
        cruce_desc = datos[["clave", "etiqueta"]].merge(extra, on="clave", how="left")
        texto = cruce_desc["descripcion"].fillna("").str.lower()
        amob = texto.str.contains(r"amoblad|equipad|muebles", regex=True) | cruce_desc["amoblado"].astype(str).str.lower().eq("true")
        estrenar = texto.str.contains(r"estrenar|nuevo|nueva", regex=True)
        factores_sobre.extend([
            {"factor": "Amoblado o equipado en el aviso", "grupo": "Con la característica", "proporcion": float(cruce_desc.loc[amob, "etiqueta"].eq("Sobrevalorado").mean())},
            {"factor": "Amoblado o equipado en el aviso", "grupo": "Resto de los avisos", "proporcion": float(cruce_desc.loc[~amob, "etiqueta"].eq("Sobrevalorado").mean())},
            {"factor": "A estrenar / nuevo en el aviso", "grupo": "Con la característica", "proporcion": float(cruce_desc.loc[estrenar, "etiqueta"].eq("Sobrevalorado").mean())},
            {"factor": "A estrenar / nuevo en el aviso", "grupo": "Resto de los avisos", "proporcion": float(cruce_desc.loc[~estrenar, "etiqueta"].eq("Sobrevalorado").mean())},
        ])
    df_factores = pd.DataFrame(factores_sobre)

    def tasa(factor, grupo):
        fila = df_factores[df_factores.factor.eq(factor) & df_factores.grupo.eq(grupo)]
        return float(fila.proporcion.iloc[0]) if not fila.empty else float("nan")

    fuera_usd = float((~datos.loc[datos.moneda.eq("USD"), "etiqueta"].eq("En linea")).mean())
    brecha_usd = datos.loc[datos.moneda.eq("USD"), "precio_m2"].median() / datos.loc[datos.moneda.eq("ARS"), "precio_m2"].median() - 1
    tasa_extra = tasa("Amoblado o equipado en el aviso", "Con la característica")
    tasa_base = tasa("Amoblado o equipado en el aviso", "Resto de los avisos")
    st.subheader("Publicar en dólares y los detalles que no entran en la ficha duplican los sobreprecios")
    st.markdown(
        f"Los avisos en dólares piden un **{pct(brecha_usd)} más por m²**. "
        "¿Qué tienen en común los avisos donde el precio supera ampliamente al modelo? Al cruzar las predicciones "
        "con las características del aviso y con el texto libre de la publicación, aparecen dos explicaciones claras: "
        f"**el segmento en dólares** (donde {pct(fuera_usd)} de los avisos se sale de los cortes) y **atributos de "
        "confort que la tabla estructurada no captura** (como entregarse *amoblado/equipado* o ser *a estrenar*, "
        f"que llevan la tasa de avisos por encima del corte de {pct(tasa_base, 1)} a {pct(tasa_extra, 1)})."
    )
    chart_factores = alt.Chart(df_factores).mark_bar().encode(
        x=alt.X("proporcion:Q", title="% de avisos marcados por encima del corte (Sobrevalorados)", axis=alt.Axis(format="%")),
        y=alt.Y("factor:N", title=None, sort="-x", axis=alt.Axis(labelLimit=220)),
        yOffset=alt.YOffset("grupo:N"),
        color=alt.Color(
            "grupo:N",
            title="Segmento",
            scale=alt.Scale(domain=["Con la característica", "Resto de los avisos"], range=[COLORES_COMPARACION["Sobrevalorado"], "#9173aa"]),
        ),
        tooltip=[
            alt.Tooltip("factor:N", title="Factor"),
            alt.Tooltip("grupo:N", title="Grupo"),
            alt.Tooltip("proporcion:Q", title="% Sobrevalorado", format=".1%"),
        ],
    ).properties(height=190, title="Por qué el modelo marca más avisos caros en ciertos segmentos")
    mostrar(chart_factores, width="stretch")
    st.caption(
        "Ojo con la lectura: acá no medimos qué encarece un departamento, sino qué avisos se apartan de la "
        "referencia. Una cochera encarece el alquiler, pero el modelo ya la conoce y por eso no genera sobreprecio; "
        "lo que el aviso cuenta en el texto libre, en cambio, el modelo no lo puede ver."
    )

    que_mueve_el_precio(datos, csv_origen)

    st.markdown("**Cómo cambia la mediana del precio por m² dentro de una misma localidad (40 a 90 m²):**")
    c1, c2 = st.columns(2)
    with c1:
        localidades_opciones = sorted(datos["localidad"].dropna().unique())
        localidad = st.selectbox("Localidad para comparar", localidades_opciones,
                                 index=localidades_opciones.index("Capital") if "Capital" in localidades_opciones else 0)
    with c2:
        variable = st.selectbox("Comparar por", ["Dormitorios", "Cochera", "Portal"])
    nombres = {"Dormitorios": "dormitorios", "Cochera": "cochera", "Portal": "fuente"}
    columna = nombres[variable]
    muestra = datos[datos.localidad.eq(localidad) & datos.superficie_cubierta_m2.between(40, 90)].dropna(subset=[columna]).copy()
    if muestra.empty:
        st.info("No hay avisos para esta comparación.")
    else:
        muestra["grupo"] = muestra[columna].astype(str)
        grupos = muestra.groupby("grupo", as_index=False).agg(avisos=("clave", "size"), mediana=("precio_m2", "median"))
        colores_grupo = escala(COLORES_PORTAL) if variable == "Portal" else alt.Scale(domain=sorted(datos[columna].dropna().astype(str).unique().tolist()), range=COLORES_OTROS)
        barras = alt.Chart(grupos).mark_bar().encode(
            color=alt.Color("grupo:N", scale=colores_grupo, legend=None),
            x=alt.X("mediana:Q", title="Mediana publicada ($/m²)"),
            y=alt.Y("grupo:N", title=variable, sort="-x"),
            tooltip=[alt.Tooltip("grupo:N", title=variable), alt.Tooltip("avisos:Q", title="Avisos"),
                     alt.Tooltip("mediana:Q", title="Mediana ($/m²)", format=",.0f")]
        ).properties(height=max(130, min(260, 30 * len(grupos))))
        mostrar(barras, width="stretch")
        st.caption(f"Avisos de {localidad}, entre 40 y 90 m². Permite ver qué características suben o bajan la mediana en esa zona.")

    grupos_prueba = datos[datos.particion.eq("test")].groupby("etiqueta")
    with st.expander("Cómo lo sabemos · Qué tienen en común los tres grupos en el conjunto de prueba"):
        resumen = grupos_prueba.agg(
            avisos=("clave", "size"), superficie=("superficie_cubierta_m2", "median"),
            dormitorios=("dormitorios", "median"), distancia=("dist_centro_km", "median"),
            precio=("precio_m2", "median"), predicho=("precio_m2_predicho", "median")
        ).reset_index()
        st.dataframe(
            resumen.assign(etiqueta=resumen.etiqueta.map(NOMBRE_COMPARACION)).rename(columns={
                "etiqueta": "Clasificación",
                "avisos": "Avisos (test)",
                "superficie": "Superficie mediana (m²)",
                "dormitorios": "Dormitorios medianos",
                "distancia": "Distancia al centro (km)",
                "precio": "Publicado mediano ($/m²)",
                "predicho": "Modelo mediano ($/m²)",
            }),
            hide_index=True,
            width="stretch",
        )
        sup_opor = resumen.loc[resumen.etiqueta.eq("Oportunidad"), "superficie"]
        sup_sobre = resumen.loc[resumen.etiqueta.eq("Sobrevalorado"), "superficie"]
        if not sup_opor.empty and not sup_sobre.empty:
            st.caption(
                f"En prueba, las oportunidades suelen ser departamentos algo más grandes (mediana de "
                f"{miles(sup_opor.iloc[0])} m² vs. {miles(sup_sobre.iloc[0])} m² de los sobrevalorados) cuyo "
                "precio por m² quedó muy por debajo tanto del modelo como de su zona."
            )

    caro, vecinos_caro = caso_extremo(datos, "Sobrevalorado")
    barato, vecinos_barato = caso_extremo(datos, "Oportunidad")
    hay_casos = caro is not None and barato is not None
    if hay_casos:
        dif_caro = caro.precio_m2 / vecinos_caro.precio_m2.median() - 1
        dif_barato = barato.precio_m2 / vecinos_barato.precio_m2.median() - 1
    st.subheader("Dos casos reales en los extremos: el modelo y los vecinos apuntan en la misma dirección")
    st.markdown(
        "Elegimos dos avisos del conjunto de prueba (que el modelo nunca vio al entrenar), uno de cada extremo, "
        "entre 30 y 130 m² y con al menos diez comparables directos."
        + (f" El caso caro pide un **{porcentaje(dif_caro)}** más que {miles(len(vecinos_caro))} vecinos en "
           f"{nombre(caro.localidad)}; la oportunidad, un **{porcentaje(dif_barato)}** menos que "
           f"{miles(len(vecinos_barato))} en {nombre(barato.localidad)}."
           if hay_casos else "")
    )
    col_sobre, col_opor = st.columns(2)
    columnas_casos = [
        (col_sobre, "Caso sobrevalorado (por encima del corte)", "Sobrevalorado", caro, vecinos_caro),
        (col_opor, "Caso oportunidad (por debajo del corte)", "Oportunidad", barato, vecinos_barato),
    ]
    for contenedor, titulo, lado, aviso, grupo in columnas_casos:
        with contenedor:
            if aviso is None:
                st.caption(f"{titulo}: no encontramos un caso de prueba con suficientes comparables.")
                continue
            mediana = grupo.precio_m2.median()
            direccion = "por debajo" if aviso.precio_m2 < mediana else "por encima"
            dormitorios = "1 dormitorio" if aviso.dormitorios == 1 else f"{miles(aviso.dormitorios)} dormitorios"
            st.markdown(f"#### {nombre(aviso.localidad)} · {miles(aviso.superficie_cubierta_m2)} m² · {dormitorios}")
            diferencia = abs(aviso.precio_m2 / mediana - 1)
            st.markdown(
                f"- **Publicado:** ${miles(aviso.precio_m2)}/m² (${miles(aviso.precio_m2 * aviso.superficie_cubierta_m2)}/mes)\n"
                f"- **Referencia del modelo:** ${miles(aviso.precio_m2_predicho)}/m² ({porcentaje(aviso.residuo_relativo)})\n"
                f"- **Mediana de {miles(len(grupo))} comparables:** ${miles(mediana)}/m² (**{pct(diferencia)} {direccion}**)"
            )
            mostrar(
                grafico_comparacion_aviso(aviso.precio_m2, aviso.precio_m2_predicho, mediana, len(grupo), lado),
                width="stretch",
            )
            with st.expander(f"{titulo} · Ver los {miles(len(grupo))} comparables"):
                st.markdown(f"[Abrir publicación original]({aviso.url})")
                mostrar_comparables(grupo)
    st.caption(
        "Si un aviso pide mucho más que el modelo y que decenas de departamentos del mismo tamaño, moneda y localidad, "
        "conviene verificar en el enlace si ofrece equipamiento extraordinario o si está fuera de precio."
    )


def respuesta(datos, cortes, metrics):
    """El cierre del relato: junta lo que aporto cada paso y responde la pregunta de la portada."""
    st.header("La respuesta")
    paso(6, "juntamos todo lo anterior y respondemos la pregunta.")
    total = len(datos)
    en_linea = datos.etiqueta.eq("En linea").sum()
    sobre = datos.etiqueta.eq("Sobrevalorado").sum()
    opor = datos.etiqueta.eq("Oportunidad").sum()
    test = metrics["test"]
    localidades = (datos.assign(arriba=datos.etiqueta.eq("Sobrevalorado"))
                   .groupby("localidad", as_index=False)
                   .agg(avisos=("clave", "size"), arriba=("arriba", "mean"))
                   .query("avisos >= 50").sort_values("arriba", ascending=False))
    usd, ars = datos[datos.moneda.eq("USD")], datos[datos.moneda.eq("ARS")]
    primera, segunda = localidades.iloc[0], localidades.iloc[1]

    st.markdown(
        "> *¿Cuándo un alquiler en Mendoza está realmente sobrevalorado (o es una oportunidad) y dónde se "
        "concentran esos desfases?*"
    )
    st.subheader("Un alquiler está sobrevalorado cuando se aparta de sus equivalentes más de lo que "
                 "el modelo se equivoca, y sus vecinos lo confirman")
    with st.container(border=True):
        st.markdown(
            f"**Cuándo.** Un aviso está **sobrevalorado** cuando pide más de un **{pct(cortes['alto'], 1)}** "
            f"por encima de lo que piden departamentos equivalentes, y es una **oportunidad** cuando pide más de "
            f"un **{pct(abs(cortes['bajo']), 1)}** por debajo. Además, sus vecinos comparables (misma localidad, "
            "moneda y dormitorios, con superficie parecida) tienen que apuntar en la misma dirección.\n\n"
            f"**Dónde.** En **{nombre(primera.localidad)}** ({pct(primera.arriba, 1)} de sus avisos) y "
            f"**{nombre(segunda.localidad)}** ({pct(segunda.arriba, 1)}), en los avisos **publicados en dólares** "
            f"({pct(usd.etiqueta.eq('Sobrevalorado').mean(), 1)} contra "
            f"{pct(ars.etiqueta.eq('Sobrevalorado').mean(), 1)} en pesos) y en los que destacan en el texto "
            "atributos que la ficha no registra (amoblado, equipado, a estrenar).\n\n"
            f"**En conjunto**, el mercado no está inflado de punta a punta: {pct(en_linea / total, 1)} de los avisos "
            f"tiene precio acorde, y los que se apartan se reparten casi por igual hacia arriba "
            f"({pct(sobre / total, 1)}) y hacia abajo ({pct(opor / total, 1)})."
        )

    st.markdown("#### Cómo llegamos: lo que aportó cada paso")
    st.markdown(
        f"1. **Los datos.** {miles(total)} avisos de Inmoclick e InmoUP, relevados entre el "
        f"{metrics['dataset']['ventana_inicio']} y el {metrics['dataset']['ventana_fin']}, limpios y llevados a "
        "una sola moneda. Esa es la base contra la que se compara cada aviso.\n"
        "2. **Lo que encontramos.** El precio depende del tamaño (el m² de un departamento chico es más caro), de "
        "la moneda y de dónde está (localidad, zona y distancia al centro, que no siguen una línea recta). Por eso "
        "comparar precios crudos no sirve: hay que comparar contra equivalentes.\n"
        f"3. **El modelo.** Estima el precio de cada aviso a partir de lo que publica y erra "
        f"${miles(test['mae_modelo'])}/m² en promedio, un {pct(test['mejora_vs_mejor_piso_pct'])} menos que la "
        "mejor regla simple. Su propio margen de error define los cortes: son los percentiles 10 y 90 del error "
        "**en validación**, sobre avisos que el modelo no vio al entrenar. Dentro de esa banda no afirmamos nada.\n"
        "4. **Los casos.** Los avisos que superan los cortes se contrastan con sus vecinos comparables. Así vimos "
        "dónde se concentran y qué tienen en común.\n"
        "5. **Explorá y probá el modelo.** El mismo criterio queda a tu alcance: podés filtrar el mercado o cargar "
        "un aviso concreto y ver de qué lado de los cortes cae."
    )

    st.markdown("#### El matiz")
    st.markdown(
        "**Parte de lo que el modelo marca como sobreprecio es valor real que el aviso no publica en forma de "
        "dato.** Lo que se cuenta solo en el texto libre (una cocina nueva, muebles, a estrenar) el modelo no lo "
        "puede ver, así que lo cuenta como sobreprecio. Por eso la app no cierra el veredicto con la estimación: lo "
        "contrasta con los comparables y deja el enlace a la publicación original para que cualquiera verifique "
        "si ese sobreprecio corresponde a algo concreto o a una expectativa del propietario."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Precio acorde", pct(en_linea / total, 1))
    c1.caption(f"{miles(en_linea)} de {miles(total)} avisos piden lo mismo que sus equivalentes.")
    c2.metric("Sobrevalorados", pct(sobre / total, 1))
    c2.caption(f"{miles(sobre)} avisos piden más de lo que pide el resto del mercado por lo mismo.")
    c3.metric("Oportunidades", pct(opor / total, 1))
    c3.caption(f"{miles(opor)} avisos piden menos.")
    st.caption(
        f"Todo el recorrido se apoya en {miles(total)} avisos, un modelo HistGradientBoosting con "
        f"{len(metrics['columnas_finales'])} características y una validación agrupada por firma de inmueble "
        "para que ningún departamento se evalúe con su propia copia."
    )
