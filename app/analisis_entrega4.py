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


def ubicacion(datos, escala_etiqueta):
    st.subheader("Luján de Cuyo y Godoy Cruz lideran el precio por m², un 29 % por encima de Las Heras")
    st.markdown(
        "El precio por metro cuadrado cambia fuerte según el departamento. Además, dentro de una misma "
        "localidad conviven barrios muy distintos: por eso agrupamos las coordenadas con **KMeans en 6 zonas "
        "geográficas** dentro de los 30 km de Plaza Independencia y separamos el interior provincial."
    )
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
    ).properties(height=280, title="Mediana por localidad (al menos 50 avisos)")

    mapa = json.loads((DATA / "departamentos_gran_mendoza.geojson").read_text(encoding="utf-8"))
    geo = datos.dropna(subset=["latitud", "longitud"]).copy()
    zona = geo[geo.longitud.between(-69.1, -68.5) & geo.latitud.between(-33.17, -32.62)]
    limites = alt.Chart(alt.Data(values=mapa["features"])).mark_geoshape(
        fill="#e8e8e4", stroke="#a5aaa5", strokeWidth=0.8).encode(
            tooltip=[alt.Tooltip("properties.departamento:N", title="Departamento")])
    puntos = alt.Chart(zona).mark_circle(size=16, opacity=.65).encode(
        longitude="longitud:Q", latitude="latitud:Q",
        color=alt.Color("etiqueta:N", scale=escala_etiqueta, title="Comparación"),
        tooltip=[alt.Tooltip("localidad:N", title="Localidad"),
                 alt.Tooltip("precio_m2:Q", title="Publicado ($/m²)", format=",.0f"),
                 alt.Tooltip("precio_m2_predicho:Q", title="Referencia ($/m²)", format=",.0f"),
                 alt.Tooltip("residuo_relativo:Q", title="Diferencia", format="+.1%")])
    mapa_chart = (limites + puntos).project(type="mercator").properties(
        height=320, title=f"Gran Mendoza: {miles(len(zona))} avisos geolocalizados según su diferencia con el modelo"
    )

    c_bar, c_map = st.columns([1, 1.25])
    with c_bar:
        st.altair_chart(grafico, width="stretch")
        st.caption("Localidades con al menos 50 avisos. Luján de Cuyo ($11.764/m²) supera en un 29 % a Las Heras ($9.091/m²).")
    with c_map:
        st.altair_chart(mapa_chart, width="stretch")
        st.caption("Límites: [departamentos de Argentina](https://github.com/mgaitan/departamentos_argentina). Los puntos muestran cómo se distribuyen los avisos en línea, sobrevalorados y oportunidades.")


def ajuste(datos, test_metrics):
    prueba = datos[datos.particion.eq("test")].copy()
    limite = max(prueba.precio_m2.max(), prueba.precio_m2_predicho.max())
    puntos = alt.Chart(prueba).mark_circle(size=24, opacity=.45, color="#2a78d6").encode(
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

    st.subheader("En el tramo central el modelo se ajusta a $1.003/m², tres veces más preciso que en el extremo caro")
    st.markdown(
        "El modelo se entrena para representar a la gran mayoría de los alquileres estándar: sobre la diagonal "
        "roja el precio publicado coincide con la estimación. En los **quintiles centrales (Q2 a Q4)** el error "
        f"absoluto medio baja hasta **${miles(error_por_tramo.loc[error_por_tramo['quintil_real'] == 'Q3 (central)', 'mae'].iloc[0])}/m²**, "
        f"mientras que en el **quintil más caro (Q5)** sube a **${miles(error_por_tramo.loc[error_por_tramo['quintil_real'] == 'Q5 (más caro)', 'mae'].iloc[0])}/m²** "
        "porque allí aparecen departamentos con lujos no estructurados o precios muy por encima de su zona."
    )

    orden_q = ["Q1 (más barato)", "Q2", "Q3 (central)", "Q4", "Q5 (más caro)"]
    colores_q = alt.Scale(domain=orden_q, range=["#9173aa", "#4d9486", "#4f86bd", "#4d9486", "#c85c50"])
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
        st.altair_chart(
            (puntos + diagonal).properties(height=360, title="Precio publicado vs. estimado (676 avisos de prueba)").interactive(),
            width="stretch",
        )
    with col_q:
        st.altair_chart(chart_quintiles, width="stretch")

    st.caption(
        f"{miles(len(prueba))} avisos de prueba. R²: {r2_score(prueba.precio_m2, prueba.precio_m2_predicho):.2f}; "
        f"error absoluto medio global: ${miles(test_metrics['mae_modelo'])}/m². "
        "En el tramo medio el modelo ofrece una referencia sólida; en los extremos señala avisos para auditar con comparables."
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


def casos(datos, cortes, csv_origen=None):
    st.header("Los casos: dónde y por qué 700 avisos se apartan de la referencia")
    st.markdown(
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
    st.subheader("Capital (12,7 %) y Luján de Cuyo (11,4 %) tienen la mayor proporción de avisos sobrevalorados")
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
        tooltip=[
            alt.Tooltip("localidad:N", title="Localidad"),
            alt.Tooltip("avisos:Q", title="Avisos totales"),
            alt.Tooltip("comparacion:N", title="Grupo"),
            alt.Tooltip("proporcion:Q", title="Proporción", format=".1%"),
        ],
    ).properties(height=250)
    st.altair_chart(graf, width="stretch")
    if not localidades.empty:
        mayor = localidades.iloc[0]
        st.markdown(
            f"En **{mayor.localidad}** ({mayor.arriba:.1%} de {miles(mayor.avisos)} avisos por encima del corte) y "
            "**Luján de Cuyo** (11,4 %) conviven departamentos estándar con zonas de alto poder adquisitivo "
            "(Quinta y Sexta Sección, Chacras de Coria, Dalvian). En mercados más homogéneos como **Guaymallén** "
            "(83,9 % en línea) y **Maipú** (82,5 % en línea), los precios se apartan mucho menos.".replace(".", ",", 1)
        )

    st.subheader("Publicar en dólares (+54 %), la cochera y los detalles fuera de la ficha duplican los sobreprecios")
    st.markdown(
        "¿Qué tienen en común los avisos donde el precio supera ampliamente al modelo? Al cruzar las predicciones "
        "con las características del aviso y con el texto libre de la publicación, aparecen dos explicaciones claras: "
        "**el segmento en dólares** (donde casi 4 de cada 10 avisos se salen de los cortes) y **atributos de confort "
        "que la tabla estructurada no captura** (como entregarse *amoblado/equipado* o ser *a estrenar*, que duplican "
        "la tasa de avisos por encima del corte del ~8 % al ~17 %)."
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
    st.altair_chart(chart_factores, width="stretch")

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
        st.altair_chart(barras, width="stretch")
        st.caption(f"Avisos de {localidad}, entre 40 y 90 m². Permite ver qué características suben o bajan la mediana en esa zona.")

    grupos_prueba = datos[datos.particion.eq("test")].groupby("etiqueta")
    with st.expander("Cómo lo sabemos · Qué tienen en común los tres grupos en el conjunto de prueba"):
        resumen = grupos_prueba.agg(
            avisos=("clave", "size"), superficie=("superficie_cubierta_m2", "median"),
            dormitorios=("dormitorios", "median"), distancia=("dist_centro_km", "median"),
            precio=("precio_m2", "median"), predicho=("precio_m2_predicho", "median")
        ).reset_index()
        st.dataframe(
            resumen.rename(columns={
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
        st.caption(
            "En prueba, las oportunidades suelen ser departamentos algo más grandes (mediana de 72 m² vs. 60 m²) "
            "cuyo precio por m² quedó muy por debajo tanto del modelo como de su zona."
        )

    st.subheader("Dos casos reales en los extremos: +63 % frente a 278 vecinos en Capital y -42 % frente a 52 en Las Heras")
    st.markdown(
        "Elegimos dos avisos del conjunto de prueba (que el modelo nunca vio al entrenar), uno de cada extremo, "
        "entre 30 y 130 m² y con al menos diez comparables directos. En ambos casos, **tanto el modelo como la "
        "mediana de los vecinos apuntan en la misma dirección**."
    )
    prueba = datos[datos.particion.eq("test") & datos.superficie_cubierta_m2.between(30, 130) & datos.dormitorios.notna()]
    col_sobre, col_opor = st.columns(2)
    columnas_casos = [
        (col_sobre, "Caso sobrevalorado (por encima del corte)", False),
        (col_opor, "Caso oportunidad (por debajo del corte)", True),
    ]
    for contenedor, titulo, ascendente in columnas_casos:
        with contenedor:
            for _, aviso in prueba.sort_values("residuo_relativo", ascending=ascendente).iterrows():
                grupo = comparables(datos, aviso.localidad, aviso.moneda, aviso.superficie_cubierta_m2, aviso.dormitorios, aviso.clave)
                lado = "Oportunidad" if ascendente else "Sobrevalorado"
                if aviso.etiqueta == lado and len(grupo) >= 10:
                    mediana = grupo.precio_m2.median()
                    direccion = "por debajo" if aviso.precio_m2 < mediana else "por encima"
                    dormitorios = "1 dormitorio" if aviso.dormitorios == 1 else f"{miles(aviso.dormitorios)} dormitorios"
                    st.markdown(f"#### {aviso.localidad} · {miles(aviso.superficie_cubierta_m2)} m² · {dormitorios}")
                    diferencia = abs(aviso.precio_m2 / mediana - 1)
                    st.markdown(
                        f"- **Publicado:** ${miles(aviso.precio_m2)}/m² (${miles(aviso.precio_m2 * aviso.superficie_cubierta_m2)}/mes)\n"
                        f"- **Referencia del modelo:** ${miles(aviso.precio_m2_predicho)}/m² ({porcentaje(aviso.residuo_relativo)})\n"
                        f"- **Mediana de {miles(len(grupo))} comparables:** ${miles(mediana)}/m² (**{diferencia:.0%} {direccion}**)"
                    )
                    st.altair_chart(
                        grafico_comparacion_aviso(aviso.precio_m2, aviso.precio_m2_predicho, mediana, len(grupo), lado),
                        width="stretch",
                    )
                    with st.expander(f"{titulo} · Ver los {miles(len(grupo))} comparables"):
                        st.markdown(f"[Abrir publicación original]({aviso.url})")
                        mostrar_comparables(grupo)
                    break
            else:
                st.caption(f"{titulo}: no encontramos un caso de prueba con suficientes comparables.")
    st.caption(
        "Si un aviso pide mucho más que el modelo y que decenas de departamentos del mismo tamaño, moneda y localidad, "
        "conviene verificar en el enlace si ofrece equipamiento extraordinario o si está fuera de precio."
    )
