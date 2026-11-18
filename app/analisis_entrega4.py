"""Datos y gráficos para la historia de la Entrega 4."""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import r2_score
from paleta import COLORES_LOCALIDAD, COLORES_PORTAL, COLORES_OTROS, COLORES_COMPARACION, escala

DATA = Path(__file__).resolve().parent / "data"


def miles(valor):
    return f"{valor:,.0f}".replace(",", ".")


def porcentaje(valor):
    return f"{valor:+.1%}".replace(".", ",")


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
    st.caption("La comparación no controla el estado real ni todos los detalles del inmueble. Un departamento puede aparecer en ambos portales.")
    if len(grupo) < 10:
        st.warning("Hay menos de diez avisos comparables. Usá la mediana con cuidado.")


def ubicacion(datos, escala_etiqueta):
    st.subheader("El precio por m² cambia según la localidad")
    tabla = (datos.groupby("localidad", as_index=False)
             .agg(avisos=("clave", "size"), mediana=("precio_m2", "median"),
                  arriba=("etiqueta", lambda s: s.eq("Sobrevalorado").mean()),
                  abajo=("etiqueta", lambda s: s.eq("Oportunidad").mean()))
             .query("avisos >= 50").sort_values("mediana", ascending=False))
    grafico = alt.Chart(tabla).mark_bar().encode(
        color=alt.Color("localidad:N", scale=escala(COLORES_LOCALIDAD), legend=None),
        x=alt.X("mediana:Q", title="Mediana del precio publicado ($/m²)"),
        y=alt.Y("localidad:N", title=None, sort="-x"),
        tooltip=["localidad:N", alt.Tooltip("avisos:Q", title="Avisos"),
                 alt.Tooltip("mediana:Q", title="Mediana ($/m²)", format=",.0f")]
    ).properties(height=220)
    st.altair_chart(grafico, width="stretch")
    st.caption("Localidades con al menos 50 avisos. Las diferencias incluyen propiedades de distintos tamaños y características.")
    with st.expander("Mapa y zonas"):
        mapa = json.loads((DATA / "departamentos_gran_mendoza.geojson").read_text(encoding="utf-8"))
        geo = datos.dropna(subset=["latitud", "longitud"]).copy()
        zona = geo[geo.longitud.between(-69.1, -68.5) & geo.latitud.between(-33.17, -32.62)]
        limites = alt.Chart(alt.Data(values=mapa["features"])).mark_geoshape(
            fill="#e8e8e4", stroke="#a5aaa5", strokeWidth=0.8).encode(
                tooltip=[alt.Tooltip("properties.departamento:N", title="Departamento")])
        puntos = alt.Chart(zona).mark_circle(size=14, opacity=.6).encode(
            longitude="longitud:Q", latitude="latitud:Q",
            color=alt.Color("etiqueta:N", scale=escala_etiqueta, title="Comparación"),
            tooltip=[alt.Tooltip("localidad:N", title="Localidad"),
                     alt.Tooltip("precio_m2:Q", title="Precio ($/m²)", format=",.0f"),
                     alt.Tooltip("residuo_relativo:Q", title="Diferencia", format="+.1%")])
        st.altair_chart((limites + puntos).project(type="mercator").properties(height=400), width="stretch")
        st.caption(f"Se muestran {miles(len(zona))} avisos con coordenadas en el recorte del Gran Mendoza. Los límites son departamentales; los puntos usan la ubicación publicada.")
        st.markdown("Las seis zonas del modelo agrupan coordenadas cercanas dentro de 30 km del centro. Se crearon para distinguir ubicaciones dentro de una misma localidad; el interior se trata aparte. Son grupos calculados, no barrios oficiales.")
        st.caption("Límites: [departamentos de Argentina](https://github.com/mgaitan/departamentos_argentina), recortados y simplificados para esta vista.")


def ajuste(datos, test_metrics):
    prueba = datos[datos.particion.eq("test")].copy()
    limite = max(prueba.precio_m2.max(), prueba.precio_m2_predicho.max())
    puntos = alt.Chart(prueba).mark_circle(size=22, opacity=.4, color="#2a78d6").encode(
        x=alt.X("precio_m2:Q", title="Precio publicado ($/m²)", scale=alt.Scale(domain=[0, limite])),
        y=alt.Y("precio_m2_predicho:Q", title="Precio estimado ($/m²)", scale=alt.Scale(domain=[0, limite])),
        tooltip=[alt.Tooltip("localidad:N", title="Localidad"),
                 alt.Tooltip("precio_m2:Q", title="Publicado", format=",.0f"),
                 alt.Tooltip("precio_m2_predicho:Q", title="Estimado", format=",.0f"),
                 alt.Tooltip("residuo_relativo:Q", title="Diferencia", format="+.1%")])
    diagonal = alt.Chart(pd.DataFrame({"x": [0, limite], "y": [0, limite]})).mark_line(
        color="#d03b3b", strokeDash=[5, 5]).encode(x="x:Q", y="y:Q")
    st.subheader("Cómo se comparan los precios estimados y publicados")
    st.markdown("Cada punto es un aviso que el modelo no usó para entrenar. Sobre la diagonal, el precio publicado coincide con el estimado.")
    st.altair_chart((puntos + diagonal).properties(height=400).interactive(), width="stretch")
    st.caption(f"{miles(len(prueba))} avisos de prueba. R²: {r2_score(prueba.precio_m2, prueba.precio_m2_predicho):.2f}; error absoluto medio: ${miles(test_metrics['mae_modelo'])}/m². R² resume el ajuste del conjunto, no la proporción de aciertos.")
    tramos = pd.qcut(prueba.precio_m2, 5, labels=False, duplicates="drop")
    error = (prueba.precio_m2 - prueba.precio_m2_predicho).abs()
    error_por_tramo = error.groupby(tramos).mean()
    if len(error_por_tramo) == 5:
        st.markdown(
            f"En el tramo central de los precios **publicados**, el modelo se apartó "
            f"**${miles(error_por_tramo.iloc[2])}/m²** en promedio. En el tramo más caro, "
            f"la diferencia media fue de **${miles(error_por_tramo.iloc[-1])}/m²**. "
            "Si el precio se aleja mucho de la estimación, conviene revisarlo también "
            "frente a otros avisos de características parecidas."
        )
    prueba["tramo_estimado"] = pd.qcut(prueba.precio_m2_predicho, 5, labels=False, duplicates="drop") + 1
    prueba["error_absoluto"] = (prueba.precio_m2 - prueba.precio_m2_predicho).abs()
    tabla = prueba.groupby("tramo_estimado", observed=True).agg(avisos=("clave", "size"), diferencia_mediana=("residuo_relativo", "median"), mae=("error_absoluto", "mean")).reset_index()
    tabla["diferencia_mediana"] = tabla.diferencia_mediana.map(porcentaje)
    tabla["mae"] = tabla.mae.map(miles)
    with st.expander("Cómo se ajusta según el precio estimado"):
        st.dataframe(tabla.rename(columns={"tramo_estimado":"Tramo de referencia", "avisos":"Avisos", "diferencia_mediana":"Diferencia mediana", "mae":"MAE ($/m²)"}), hide_index=True, width="stretch")
        st.caption("Cada tramo reúne aproximadamente una quinta parte de los avisos de prueba. La diferencia mediana conserva el signo; el MAE mide cuánto se aleja, en promedio, la estimación.")


def casos(datos, cortes):
    st.header("Los avisos que se alejan de la referencia")
    st.markdown(
        "Primero miramos cuánto se aparta el precio publicado de la estimación. "
        "Después buscamos avisos de la misma localidad, moneda y cantidad de dormitorios, "
        "con una superficie parecida. Si las dos comparaciones apuntan en la misma "
        "dirección, tenemos más motivos para revisar el precio."
    )
    st.caption(
        f"Marcamos los avisos que se apartan más allá de {porcentaje(cortes['bajo'])} "
        f"o {porcentaje(cortes['alto'])}. Los cortes corresponden a los percentiles "
        "10 y 90 de las diferencias en validación."
    )
    localidades = datos.assign(arriba=datos.etiqueta.eq("Sobrevalorado"), abajo=datos.etiqueta.eq("Oportunidad")).groupby("localidad", as_index=False).agg(
        avisos=("clave", "size"), mediana=("precio_m2", "median"), arriba=("arriba", "mean"), abajo=("abajo", "mean")).query("avisos >= 50").sort_values("arriba", ascending=False)
    st.subheader("Dónde aparecen las diferencias")
    porcentajes = localidades.melt(id_vars=["localidad", "avisos"], value_vars=["arriba", "abajo"], var_name="comparacion", value_name="proporcion")
    porcentajes["comparacion"] = porcentajes["comparacion"].map({"arriba": "Por encima", "abajo": "Por debajo"})
    graf = alt.Chart(porcentajes).mark_bar().encode(
        x=alt.X("proporcion:Q", title="Proporción de avisos", axis=alt.Axis(format="%")),
        y=alt.Y("localidad:N", sort=localidades.localidad.tolist(), title=None),
        yOffset=alt.YOffset("comparacion:N"),
        color=alt.Color("comparacion:N", title="Respecto de la referencia", scale=alt.Scale(domain=["Por encima", "Por debajo"], range=[COLORES_COMPARACION["Sobrevalorado"], COLORES_COMPARACION["Oportunidad"]])),
        tooltip=[alt.Tooltip("localidad:N", title="Localidad"), alt.Tooltip("avisos:Q", title="Avisos"), alt.Tooltip("comparacion:N", title="Grupo"), alt.Tooltip("proporcion:Q", title="Proporción", format=".1%")]).properties(height=240)
    st.altair_chart(graf, width="stretch")
    if not localidades.empty:
        mayor = localidades.iloc[0]
        st.write(f"Entre las localidades con al menos 50 avisos, {mayor.localidad} tiene la mayor proporción por encima del corte: {mayor.arriba:.1%} de {miles(mayor.avisos)} avisos.".replace(".", ",", 1))
    st.caption("Localidades con al menos 50 avisos. Las diferencias pueden reflejar limitaciones del modelo en cada zona.")
    grupos_prueba = datos[datos.particion.eq("test")].groupby("etiqueta")
    perfiles = grupos_prueba[["superficie_cubierta_m2", "dist_centro_km"]].median()
    if {"Sobrevalorado", "Oportunidad"}.issubset(perfiles.index):
        arriba, abajo = perfiles.loc["Sobrevalorado"], perfiles.loc["Oportunidad"]
        st.markdown(
            f"En los avisos de prueba, los que quedaron por encima del corte tenían "
            f"una superficie mediana de {miles(arriba.superficie_cubierta_m2)} m². "
            f"En los que quedaron por debajo era de {miles(abajo.superficie_cubierta_m2)} m². "
            "Los grupos mezclan localidades y características. Para investigar un aviso "
            "concreto, buscamos otros de tamaño y ubicación similares."
        )
    with st.expander("Qué tienen en común los grupos"):
        resumen = grupos_prueba.agg(
            avisos=("clave", "size"), superficie=("superficie_cubierta_m2", "median"),
            dormitorios=("dormitorios", "median"), distancia=("dist_centro_km", "median"),
            precio=("precio_m2", "median")).reset_index()
        st.dataframe(resumen.rename(columns={"etiqueta": "Comparación", "avisos": "Avisos", "superficie": "Superficie mediana (m²)", "dormitorios": "Dormitorios medianos", "distancia": "Distancia mediana (km)", "precio": "Precio mediano ($/m²)"}), hide_index=True, width="stretch")
        st.caption("Son descripciones de grupos que mezclan localidades y tipos de inmueble. El precio forma parte de la clasificación, por eso su diferencia no explica por sí sola las causas.")
    st.subheader("Qué cambia entre los avisos")
    st.markdown("Comparamos el precio por m² de avisos de una misma localidad y de tamaño parecido. Las demás características pueden seguir siendo diferentes.")
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
                     alt.Tooltip("mediana:Q", title="Mediana ($/m²)", format=",.0f")]).properties(height=max(130, min(260, 30 * len(grupos))))
        st.altair_chart(barras, width="stretch")
        st.caption(f"Avisos de {localidad}, entre 40 y 90 m². Las diferencias son descriptivas; no demuestran que esa característica cause el precio.")
    st.subheader("Dos casos de prueba")
    st.markdown(
        "Elegimos un aviso de cada extremo entre los de 30 a 130 m² que tienen "
        "al menos diez comparables. Estos ejemplos muestran cómo hacer la comparación."
    )
    prueba = datos[datos.particion.eq("test") & datos.superficie_cubierta_m2.between(30, 130) & datos.dormitorios.notna()]
    for titulo, ascendente in [("Un precio por encima de la referencia", False), ("Un precio por debajo de la referencia", True)]:
        for _, aviso in prueba.sort_values("residuo_relativo", ascending=ascendente).iterrows():
            grupo = comparables(datos, aviso.localidad, aviso.moneda, aviso.superficie_cubierta_m2, aviso.dormitorios, aviso.clave)
            lado = "Oportunidad" if ascendente else "Sobrevalorado"
            if aviso.etiqueta == lado and len(grupo) >= 10:
                mediana = grupo.precio_m2.median()
                direccion = "por debajo" if aviso.precio_m2 < mediana else "por encima"
                dormitorios = "1 dormitorio" if aviso.dormitorios == 1 else f"{miles(aviso.dormitorios)} dormitorios"
                st.markdown(f"**{aviso.localidad}, {miles(aviso.superficie_cubierta_m2)} m² y {dormitorios}**")
                st.write(
                    f"El aviso pide **${miles(aviso.precio_m2)}/m²**. El modelo estima "
                    f"**${miles(aviso.precio_m2_predicho)}/m²** y la mediana de "
                    f"**{miles(len(grupo))} comparables** es **${miles(mediana)}/m²**."
                )
                if aviso.precio_m2 != mediana:
                    diferencia = abs(aviso.precio_m2 / mediana - 1)
                    st.write(
                        f"Pide {diferencia:.0%} {direccion} de la mediana de sus comparables. "
                        f"Frente al modelo, la diferencia es {porcentaje(aviso.residuo_relativo)}."
                    )
                with st.expander(titulo + " · Ver comparables"):
                    st.markdown(f"[Abrir aviso]({aviso.url})")
                    mostrar_comparables(grupo)
                break
        else:
            st.caption(f"{titulo}: no encontramos un caso de prueba con suficientes comparables.")
    st.markdown(
        "Antes de decidir, revisá si hay diferencias de estado, equipamiento o condiciones "
        "del alquiler que el aviso no explica. El modelo y los comparables usan precios "
        "publicados. El monto pactado puede ser distinto."
    )
