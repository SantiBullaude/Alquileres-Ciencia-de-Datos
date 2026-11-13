"""Entrega 4 - genera los artefactos que lee streamlit_app.py.

Toma el CSV mas reciente que produce el pipeline (el mismo que usa
notebooks/entrega3_modelado.ipynb), entrena el pipeline elegido en la
Entrega 3 con la misma semilla, mide todo lo que va en los "Como lo
sabemos", y deja en app/data/:

    modelo_final.joblib        pipeline entrenado (derivar -> imputar -> log -> one-hot -> HGB)
    metrics.json                todos los numeros de las 3 secciones del tallo
    dataset_exploracion.parquet un aviso por fila, con prediccion, residuo y etiqueta
    rangos_columnas.json         minimo/maximo/categorias vistas en el entrenamiento

Se corre de nuevo cada vez que el pipeline suma avisos:

    cd app && python build_app.py
    streamlit run streamlit_app.py

Mismos supuestos que entrega3_modelado.ipynb: dolar MEP del 13/10/2026
($1.540), semilla 42, split y folds por firma de inmueble.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import (
    GroupShuffleSplit, GroupKFold, KFold,
    cross_validate, cross_val_predict, GridSearchCV,
)
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance

from pipeline_comun import (
    SEMILLA, LOCALIDADES_GRANDES, INICIO_VENTANA, COLUMNAS,
    LOGARITMICAS, NUMERICAS, CATEGORICAS,
    derivar_columnas, armar_pipeline, cv_agrupada, medir, MedianaPorGrupo,
)

np.random.seed(SEMILLA)

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "resultados"
SALIDA = Path(__file__).resolve().parent / "data"
SALIDA.mkdir(exist_ok=True)

TC_USD = 1540.0  # dolar MEP al 13/10/2026 (Rava Bursatil), igual que el notebook


def _a_nativo(obj):
    """Convierte recursivamente numpy/pandas scalars a tipos nativos de Python,
    para que json.dump no falle ni los convierta en texto."""
    if isinstance(obj, dict):
        return {k: _a_nativo(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_a_nativo(v) for v in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if pd.isna(obj) if np.isscalar(obj) else False:
        return None
    return obj


def mas_reciente_csv() -> Path:
    carpetas = sorted(p for p in RESULTADOS.iterdir() if p.is_dir())
    for carpeta in reversed(carpetas):
        csv = carpeta / "brujula_inmobiliaria_alquiler_mendoza.csv"
        if csv.exists():
            return csv
    raise FileNotFoundError("No se encontro ningun CSV en resultados/")


def main():
    csv = mas_reciente_csv()
    print(f"[1/9] dataset: {csv.relative_to(RAIZ)}")
    df = pd.read_csv(csv)
    n_total = len(df)

    # --- 0.1 / 1.1 filtro de filas + conversion a pesos ---------------------
    paso1 = df[df["moneda"].notna()]
    paso2 = paso1[paso1["motivo_sospecha"].isna()]
    datos = paso2[paso2["precio_m2"].notna()].copy().reset_index(drop=True)
    datos["precio_m2"] = np.where(datos["moneda"] == "USD",
                                   datos["precio_m2"] * TC_USD, datos["precio_m2"])

    pasos_filtro = [
        {"paso": "Dataset completo", "filas": n_total,
         "descripcion": "Un aviso de alquiler publicado en Inmoclick o InmoUP para Mendoza."},
        {"paso": "Con moneda", "filas": len(paso1),
         "descripcion": "Se descartan los avisos sin precio ni moneda publicados."},
        {"paso": "Sin motivo de sospecha", "filas": len(paso2),
         "descripcion": "Se descartan los que el pipeline marco con superficie imposible o precio de venta cargado como alquiler."},
        {"paso": "Con precio_m2 (entran al modelo)", "filas": len(datos),
         "descripcion": f"Se convierten a pesos los avisos en USD con el dolar MEP del 13/10/2026 (${TC_USD:,.0f})."},
    ]
    print("      " + " -> ".join(f"{p['filas']}" for p in pasos_filtro))

    # --- 1.3 firma de inmueble + split por grupo -----------------------------
    direccion_norm = (datos["direccion"].fillna("").str.lower()
                       .str.replace(r"[^a-z0-9]+", "", regex=True))
    firma = (direccion_norm + "|" + datos["localidad"].fillna("").str.lower()
             + "|" + datos["superficie_cubierta_m2"].astype(str))
    firma = firma.where(direccion_norm != "", datos["clave"])
    grupos = pd.Series(pd.factorize(firma)[0], index=datos.index, name="grupo")

    X = datos[COLUMNAS]
    y = datos["precio_m2"]
    corte = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEMILLA)
    idx_train, idx_test = next(corte.split(X, y, groups=grupos))
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
    g_train, g_test = grupos.iloc[idx_train], grupos.iloc[idx_test]
    print(f"[2/9] train {len(X_train)} filas | test {len(X_test)} filas "
          f"| firmas compartidas: {len(set(g_train) & set(g_test))}")

    # --- 1.4 piso y escalera de pisos -----------------------------------------
    piso_p0 = DummyRegressor(strategy="median")
    piso_p1 = MedianaPorGrupo(grupos=(("tramo_sup",),), n_tramos=10, min_n=20)
    piso_p3 = MedianaPorGrupo(grupos=(("tramo_sup", "moneda"), ("tramo_sup",)), n_tramos=10, min_n=20)

    mae_piso = medir(piso_p0, X_train, y_train, g_train)
    mae_p1 = medir(piso_p1, X_train, y_train, g_train)
    mae_p3 = medir(piso_p3, X_train, y_train, g_train)
    print(f"[3/9] pisos: P0 (global) {mae_piso.mean():,.0f} +/- {mae_piso.std():,.0f} | "
          f"P1 (superficie) {mae_p1.mean():,.0f} +/- {mae_p1.std():,.0f} | "
          f"P3 (superficie x moneda) {mae_p3.mean():,.0f} +/- {mae_p3.std():,.0f}")

    # --- 2.1 comparacion de 4 familias -----------------------------------------
    MODELOS = {
        "P0 (mediana global)": piso_p0,
        "P1 (mediana por superficie)": piso_p1,
        "P3 (mediana sup x moneda)": piso_p3,
        "Lineal": armar_pipeline(LinearRegression()),
        "Lineal log(y)": TransformedTargetRegressor(armar_pipeline(LinearRegression()),
                                                      func=np.log, inverse_func=np.exp),
        "Random Forest": armar_pipeline(RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                                               n_jobs=-1, random_state=SEMILLA), escalar=False),
        "HistGradientBoosting": armar_pipeline(HistGradientBoostingRegressor(loss="absolute_error",
                                                                              random_state=SEMILLA), escalar=False),
    }
    maes_previos = {
        "P0 (mediana global)": mae_piso,
        "P1 (mediana por superficie)": mae_p1,
        "P3 (mediana sup x moneda)": mae_p3,
    }
    maes = {n: (maes_previos[n] if n in maes_previos else medir(m, X_train, y_train, g_train))
            for n, m in MODELOS.items()}
    print("[4/9] comparacion de modelos:")
    for n, v in maes.items():
        print(f"      {n:28s} MAE = {v.mean():,.0f} +/- {v.std():,.0f}")

    mejor = "HistGradientBoosting"
    pares = []
    for otro in ["P0 (mediana global)", "P1 (mediana por superficie)", "P3 (mediana sup x moneda)",
                 "Lineal", "Lineal log(y)", "Random Forest"]:
        dif = maes[otro] - maes[mejor]
        pares.append({"contra": otro, "gana_por": round(float(dif.mean()), 0),
                      "desvio_diferencia": round(float(dif.std()), 0),
                      "folds_gana": int((dif > 0).sum()), "folds_totales": int(len(dif))})

    comparacion_modelos = [
        {"modelo": n, "mae_mean": round(float(v.mean()), 0), "mae_std": round(float(v.std()), 0),
         "mejora_vs_piso": round(1 - v.mean() / mae_piso.mean(), 4),
         "mejora_vs_mejor_piso": round(1 - v.mean() / mae_p3.mean(), 4),
         "folds": [round(float(x), 1) for x in v]}
        for n, v in maes.items()
    ]

    # --- 2.2 ablacion de columnas ----------------------------------------------
    base = maes[mejor]
    modelo_hgb = lambda cols: armar_pipeline(
        HistGradientBoostingRegressor(loss="absolute_error", random_state=SEMILLA),
        columnas=cols, escalar=False)
    ablacion = []
    for col in COLUMNAS:
        sin = medir(modelo_hgb([c for c in COLUMNAS if c != col]), X_train, y_train, g_train)
        dif = sin - base
        ablacion.append({"columna": col, "mae_con": round(float(base.mean()), 0),
                          "mae_sin": round(float(sin.mean()), 0),
                          "delta": round(float(dif.mean()), 0), "desvio_delta": round(float(dif.std()), 0),
                          "folds_ayuda": int((dif > 0).sum())})
    ablacion.sort(key=lambda d: -d["delta"])
    salen = [a["columna"] for a in ablacion if a["delta"] <= 5 or a["folds_ayuda"] < 8]
    FINALES = [c for c in COLUMNAS if c not in salen]
    mae_finales = medir(modelo_hgb(FINALES), X_train, y_train, g_train)
    print(f"[5/9] ablacion -> salen {salen}")
    print(f"      quedan {len(FINALES)} de {len(COLUMNAS)}: {FINALES}")
    print(f"      MAE con todas {base.mean():,.0f} | con finales {mae_finales.mean():,.0f}")

    # --- 2.3 grid search de hiperparametros -------------------------------------
    grilla = GridSearchCV(
        modelo_hgb(FINALES),
        param_grid={
            "modelo__loss": ["absolute_error", "squared_error"],
            "modelo__learning_rate": [0.03, 0.1],
            "modelo__max_iter": [200, 500],
            "modelo__max_leaf_nodes": [15, 31],
            "modelo__min_samples_leaf": [10, 30],
        },
        scoring="neg_mean_absolute_error",
        cv=cv_agrupada(0),
        n_jobs=-1,
    )
    grilla.fit(X_train, y_train, groups=g_train)
    res = pd.DataFrame(grilla.cv_results_)
    cols_param = [c for c in res.columns if c.startswith("param_")]
    tabla_grilla = (res[cols_param + ["mean_test_score", "std_test_score", "rank_test_score"]]
                    .assign(MAE=lambda d: -d["mean_test_score"], desvio=lambda d: d["std_test_score"])
                    .drop(columns=["mean_test_score", "std_test_score"])
                    .rename(columns=lambda c: c.replace("param_modelo__", ""))
                    .sort_values("rank_test_score"))
    grid_tabla = tabla_grilla.round(4).to_dict(orient="records")
    mejor_params = {k.replace("modelo__", ""): v for k, v in grilla.best_params_.items()}
    print(f"[6/9] mejor combinacion: {mejor_params}  MAE={-grilla.best_score_:,.0f}")

    # --- 3.1 examen final sobre test --------------------------------------------
    final = grilla.best_estimator_
    final.fit(X_train, y_train)
    piso_final = DummyRegressor(strategy="median").fit(X_train, y_train)
    piso_p1_final = MedianaPorGrupo(grupos=(("tramo_sup",),), n_tramos=10, min_n=20).fit(X_train, y_train)
    piso_p3_final = MedianaPorGrupo(grupos=(("tramo_sup", "moneda"), ("tramo_sup",)), n_tramos=10, min_n=20).fit(X_train, y_train)

    pred_test = final.predict(X_test)
    pred_piso_test = piso_final.predict(X_test)
    pred_p1_test = piso_p1_final.predict(X_test)
    pred_p3_test = piso_p3_final.predict(X_test)

    mae_test = mean_absolute_error(y_test, pred_test)
    mae_piso_test = mean_absolute_error(y_test, pred_piso_test)
    mae_p1_test = mean_absolute_error(y_test, pred_p1_test)
    mae_p3_test = mean_absolute_error(y_test, pred_p3_test)

    err_rel = float((np.abs(y_test - pred_test) / y_test).median())
    err_rel_piso = float((np.abs(y_test - pred_piso_test) / y_test).median())
    err_rel_p1 = float((np.abs(y_test - pred_p1_test) / y_test).median())
    err_rel_p3 = float((np.abs(y_test - pred_p3_test) / y_test).median())
    print(f"[7/9] TEST modelo MAE={mae_test:,.0f} ({err_rel:.1%}) | "
          f"P0 MAE={mae_piso_test:,.0f} (mejora {1 - mae_test / mae_piso_test:.1%}) | "
          f"P3 MAE={mae_p3_test:,.0f} (mejora {1 - mae_test / mae_p3_test:.1%})")

    escalera_pisos = [
        {"escalon": "P0: Mediana global (0 col)", "regla": "Mediana unica de todo el entrenamiento",
         "columnas": 0, "cv_mae": round(float(mae_piso.mean()), 0), "cv_std": round(float(mae_piso.std()), 0),
         "test_mae": round(float(mae_piso_test), 0), "test_err_rel": round(err_rel_piso, 4)},
        {"escalon": "P1: Mediana por superficie (1 col)", "regla": "Mediana por decil de superficie cubierta",
         "columnas": 1, "cv_mae": round(float(mae_p1.mean()), 0), "cv_std": round(float(mae_p1.std()), 0),
         "test_mae": round(float(mae_p1_test), 0), "test_err_rel": round(err_rel_p1, 4)},
        {"escalon": "P3: Mediana superficie x moneda (2 col)", "regla": "Mediana por decil de superficie y moneda (respaldo jerarquico)",
         "columnas": 2, "cv_mae": round(float(mae_p3.mean()), 0), "cv_std": round(float(mae_p3.std()), 0),
         "test_mae": round(float(mae_p3_test), 0), "test_err_rel": round(err_rel_p3, 4)},
        {"escalon": "Modelo: HistGradientBoosting (13 col)", "regla": "Boosting sobre las 13 columnas seleccionadas",
         "columnas": len(FINALES), "cv_mae": round(float(-grilla.best_score_), 0),
         "cv_std": round(float(tabla_grilla.iloc[0]["desvio"]), 0),
         "test_mae": round(float(mae_test), 0), "test_err_rel": round(err_rel, 4)},
    ]

    resultado = pd.DataFrame({"real": y_test.values, "predicho": pred_test}, index=X_test.index)
    resultado["error"] = resultado["predicho"] - resultado["real"]
    resultado["tramo"] = pd.qcut(resultado["real"], 5,
                                  labels=["1 (mas barato)", "2", "3", "4", "5 (mas caro)"])
    por_tramo = resultado.groupby("tramo", observed=True).agg(
        filas=("real", "size"), real_mediano=("real", "median"),
        MAE=("error", lambda e: e.abs().mean()), sesgo_medio=("error", "mean"),
    ).round(0).reset_index().to_dict(orient="records")

    por_localidad = (resultado.join(X_test["localidad"])
                      .assign(localidad=lambda d: d["localidad"].where(d["localidad"].isin(LOCALIDADES_GRANDES), "Interior"))
                      .groupby("localidad")
                      .agg(filas=("real", "size"), MAE=("error", lambda e: e.abs().mean()),
                           sesgo_medio=("error", "mean"))
                      .sort_values("MAE", ascending=False).round(0).reset_index().to_dict(orient="records"))

    por_moneda = (resultado.join(X_test["moneda"]).groupby("moneda")
                   .agg(filas=("real", "size"), real_mediano=("real", "median"),
                        MAE=("error", lambda e: e.abs().mean()), sesgo_medio=("error", "mean"))
                   .round(0).reset_index().to_dict(orient="records"))

    # --- 3.3 importancia por permutacion ----------------------------------------
    imp = permutation_importance(final, X_test, y_test, scoring="neg_mean_absolute_error",
                                  n_repeats=10, random_state=SEMILLA, n_jobs=-1)
    importancia = (pd.DataFrame({"aumento_mae": imp.importances_mean, "desvio": imp.importances_std},
                                 index=X_test.columns)
                    .loc[FINALES].sort_values("aumento_mae", ascending=False)
                    .round(0).reset_index().rename(columns={"index": "columna"}).to_dict(orient="records"))
    print("[8/9] importancia por permutacion calculada")

    # --- 3.4 chequeo de fuga: superficie -----------------------------------------
    ok = X_train["superficie_cubierta_m2"].notna()
    precio_train = y_train[ok] * X_train.loc[ok, "superficie_cubierta_m2"]
    pendiente = float(np.polyfit(np.log(X_train.loc[ok, "superficie_cubierta_m2"]),
                                  np.log(precio_train), 1)[0])
    sin_sup = [c for c in FINALES if c not in ("superficie_cubierta_m2", "superficie_total_m2")]
    mae_sin_sup = medir(modelo_hgb(sin_sup), X_train, y_train, g_train)
    mae_solo_sup = medir(modelo_hgb(["superficie_cubierta_m2"]), X_train, y_train, g_train)
    hgb_solo_sup = modelo_hgb(["superficie_cubierta_m2"]).fit(X_train, y_train)
    mae_solo_sup_test = mean_absolute_error(y_test, hgb_solo_sup.predict(X_test))

    grid_sup = [25, 35, 45, 55, 70, 90, 120, 160]
    curva_superficie = []
    for s in grid_sup:
        Xs = X_test.copy()
        Xs["superficie_cubierta_m2"] = s
        curva_superficie.append({"superficie": s, "precio_m2_predicho": float(np.median(final.predict(Xs)))})

    fuga_superficie = {
        "pendiente_log_precio_vs_log_superficie": round(pendiente, 2),
        "mae_sin_superficie": round(float(mae_sin_sup.mean()), 0),
        "desvio_sin_superficie": round(float(mae_sin_sup.std()), 0),
        "mae_solo_superficie": round(float(mae_solo_sup.mean()), 0),
        "desvio_solo_superficie": round(float(mae_solo_sup.std()), 0),
        "mae_solo_superficie_test": round(float(mae_solo_sup_test), 0),
        "mae_piso": round(float(mae_piso.mean()), 0),
        "mae_mejor_piso": round(float(mae_p3.mean()), 0),
        "mae_modelo_final_cv": round(float(-grilla.best_score_), 0),
        "curva": curva_superficie,
    }

    # --- 4. cortes de residuo para sobrevalorado / en linea / oportunidad -------
    pred_oof = cross_val_predict(final, X_train, y_train, groups=g_train, cv=cv_agrupada(0))
    resid_oof = (y_train - pred_oof) / pred_oof
    corte_bajo, corte_alto = (float(x) for x in np.quantile(resid_oof, [0.10, 0.90]))

    def etiquetar(real, predicho):
        r = (real - predicho) / predicho
        return np.select([r > corte_alto, r < corte_bajo], ["Sobrevalorado", "Oportunidad"], "En linea")

    resultado["etiqueta"] = etiquetar(resultado["real"], resultado["predicho"])
    etiquetas_test = resultado["etiqueta"].value_counts().to_dict()

    resultado["tramo_pred"] = pd.qcut(resultado["predicho"], 5,
                                       labels=["1 (mas barato)", "2", "3", "4", "5 (mas caro)"])
    cruce_real = (pd.crosstab(resultado["tramo"], resultado["etiqueta"], normalize="index")
                  .reindex(columns=["Oportunidad", "En linea", "Sobrevalorado"], fill_value=0.0)
                  .round(3).reset_index().to_dict(orient="records"))
    cruce_pred = (pd.crosstab(resultado["tramo_pred"], resultado["etiqueta"], normalize="index")
                  .reindex(columns=["Oportunidad", "En linea", "Sobrevalorado"], fill_value=0.0)
                  .round(3).reset_index().rename(columns={"tramo_pred": "tramo_predicho"}).to_dict(orient="records"))
    print(f"[9/9] cortes {corte_bajo:+.1%} / {corte_alto:+.1%} -> {etiquetas_test}")

    # ---------------------------------------------------------------------------
    # dataset de exploracion: TODO el dataset modelado, con prediccion honesta
    # (out-of-fold para train, directa para test) y etiqueta, para la "copa".
    # ---------------------------------------------------------------------------
    pred_oof_full = cross_val_predict(final, X_train, y_train, groups=g_train, cv=cv_agrupada(1))
    explo = datos.loc[X.index].copy()
    explo["particion"] = np.where(explo.index.isin(X_test.index), "test", "train")
    pred_full = pd.Series(index=X.index, dtype=float)
    pred_full.loc[X_train.index] = pred_oof_full
    pred_full.loc[X_test.index] = pred_test
    explo["precio_m2_predicho"] = pred_full.values
    explo["residuo_relativo"] = (explo["precio_m2"] - explo["precio_m2_predicho"]) / explo["precio_m2_predicho"]
    explo["etiqueta"] = etiquetar(explo["precio_m2"], explo["precio_m2_predicho"])
    explo["localidad_agrupada"] = explo["localidad"].where(explo["localidad"].isin(LOCALIDADES_GRANDES), "Interior")

    cols_exploracion = [
        "clave", "fuente", "url", "localidad", "localidad_agrupada", "direccion",
        "superficie_cubierta_m2", "dormitorios", "banos", "antiguedad", "cochera",
        "estado_conservacion", "valor_expensas", "acepta_mascotas", "zona_geo",
        "dist_centro_km", "densidad_1km", "tipo_anunciante", "moneda",
        "precio", "precio_m2", "precio_m2_predicho", "residuo_relativo", "etiqueta",
        "particion", "latitud", "longitud",
    ]
    explo = explo[[c for c in cols_exploracion if c in explo.columns]]

    # --- hallazgos del EDA (Entrega 2), recalculados sobre el dataset actual ---
    limpio_eda = df[df["motivo_sospecha"].isna()].copy()
    ars = limpio_eda.loc[limpio_eda.moneda == "ARS", "precio_m2"].dropna()
    usd = limpio_eda.loc[limpio_eda.moneda == "USD", "precio_m2"].dropna()
    sep_sin_convertir = abs((usd.mean() - ars.mean()) / np.sqrt((ars.var() + usd.var()) / 2))

    sub = limpio_eda[(limpio_eda.moneda == "ARS") & limpio_eda.superficie_cubierta_m2.notna()
                      & (limpio_eda.superficie_cubierta_m2 > 0) & limpio_eda.precio.notna()]
    sp_lineal = sub["superficie_cubierta_m2"].corr(sub["precio"], method="spearman")
    pe_lineal = sub["superficie_cubierta_m2"].corr(sub["precio"], method="pearson")
    sp_log = np.log10(sub["superficie_cubierta_m2"]).corr(np.log10(sub["precio"]), method="spearman")
    pe_log = np.log10(sub["superficie_cubierta_m2"]).corr(np.log10(sub["precio"]), method="pearson")

    h3 = limpio_eda[limpio_eda.dist_centro_km.notna() & (limpio_eda.dist_centro_km <= 30)
                     & (limpio_eda.moneda == "ARS")]
    h3_spearman = h3["dist_centro_km"].corr(h3["precio_m2"], method="spearman")

    hallazgos_eda = {
        "h1_monedas": {
            "n_ars": int(len(ars)), "n_usd": int(len(usd)),
            "mediana_ars": round(float(ars.median()), 0), "mediana_usd": round(float(usd.median()), 2),
            "separacion_sin_convertir": round(float(sep_sin_convertir), 2),
        },
        "h2_superficie_sostenida": {
            "spearman_lineal": round(float(sp_lineal), 3), "pearson_lineal": round(float(pe_lineal), 3),
            "brecha_lineal": round(float(sp_lineal - pe_lineal), 3),
            "spearman_log": round(float(sp_log), 3), "pearson_log": round(float(pe_log), 3),
            "brecha_log": round(float(sp_log - pe_log), 3), "n": int(len(sub)),
        },
        "h3_distancia_refutada": {
            "spearman": round(float(h3_spearman), 3), "n": int(len(h3)),
        },
    }

    # --- evolucion del dataset por corrida ---------------------------------------
    # avisos nuevos = primera vez que se ve esa clave; acumulado = suma corrida a corrida
    primera_vista = df.groupby("clave")["fecha_scraping"].min()
    conteo_por_fecha = primera_vista.value_counts().sort_index()
    evolucion_final = pd.DataFrame({
        "fecha_scraping": conteo_por_fecha.index,
        "avisos_nuevos": conteo_por_fecha.values,
        "avisos_acumulados": conteo_por_fecha.cumsum().values,
    })

    # --- rangos de columnas, para validar el formulario de "Proba el modelo" ----
    # Se miden sobre X_train (lo que vio el entrenamiento), en su forma CRUDA
    # (antes de derivar_columnas): asi el formulario pide lo mismo que publica un aviso.
    CRUDAS_CATEGORICAS = {"cochera", "moneda"}  # numericas/booleanas derivadas, pero el aviso las publica como texto
    rangos = {}
    for col in FINALES:
        serie_cruda = X_train[col].dropna()
        if col in CRUDAS_CATEGORICAS or col in CATEGORICAS:
            # zona_geo llega como float (0.0, 1.0, ...): sin este cast el
            # selectbox muestra "-1.0" y el pipeline no sabe parsear ese texto.
            if pd.api.types.is_float_dtype(serie_cruda):
                cats = sorted(str(int(c)) for c in serie_cruda.unique().tolist())
            else:
                cats = sorted(str(c) for c in serie_cruda.unique().tolist())
            rangos[col] = {"tipo": "categorica", "categorias": cats}
            if col == "zona_geo":
                # El codigo de KMeans no dice nada por si solo para alguien
                # que ve el proyecto por primera vez: se etiqueta con la
                # localidad mas comun de cada cluster (sobre X_train).
                etiquetas = {}
                for cat in cats:
                    sub = X_train.loc[X_train["zona_geo"] == float(cat)]
                    loc_top = sub["localidad"].mode()
                    loc_top = loc_top.iloc[0] if len(loc_top) else "sin dato"
                    dist_mediana = sub["dist_centro_km"].median()
                    if cat == "-1":
                        etiquetas[cat] = f"{cat} - Interior (lejos del Gran Mendoza)"
                    elif pd.notna(dist_mediana):
                        etiquetas[cat] = f"{cat} - {loc_top} (~{dist_mediana:.0f} km del centro)"
                    else:
                        etiquetas[cat] = f"{cat} - {loc_top}"
                rangos[col]["etiquetas"] = etiquetas
        elif col == "fecha_scraping":
            rangos[col] = {"tipo": "fecha", "min": str(serie_cruda.min()), "max": str(serie_cruda.max())}
        elif len(serie_cruda):
            # antiguedad: 20 avisos (de 2.029) cargaron el anio de construccion
            # (2019, 2026) en vez de los anios de antiguedad -- el resto va de
            # 1 a ~70. No se tocan en el entrenamiento (son filas reales, el
            # modelo las sigue viendo tal cual), pero el rango del formulario
            # no tiene sentido yendo hasta "2026 anios": se calcula sobre la
            # parte limpia, con un tope de 100 de cualquier forma.
            serie_rango = serie_cruda[serie_cruda <= 100] if col == "antiguedad" else serie_cruda
            # dormitorios/banos/antiguedad/densidad_1km son conteos enteros
            # (float64 solo por los NaN). Con cantidad par de observaciones la
            # mediana puede caer en .5 (ej 122.5 vecinos): si la serie entera
            # es siempre entera, se redondea para que el valor por defecto del
            # slider tenga sentido (ver streamlit_app.py: min/max/mediana
            # enteros -> slider de enteros).
            es_entero = bool((serie_rango == serie_rango.round()).all())
            mediana = round(serie_rango.median()) if es_entero else serie_rango.median()
            rangos[col] = {"tipo": "numerica", "min": float(serie_rango.min()), "max": float(serie_rango.max()),
                            "mediana": float(mediana)}

    metrics = {
        "dataset": {"csv": str(csv.relative_to(RAIZ)), "filas_totales": n_total,
                    "filas_modeladas": len(datos), "avisos_usd_convertidos": int((datos["moneda"] == "USD").sum()),
                    "tc_usd": TC_USD, "ventana_inicio": "2026-09-07", "ventana_fin": "2026-10-13"},
        "pasos_filtro": pasos_filtro,
        "columnas_descartadas_fuga": [
            {"columna": "precio", "motivo": "Es el objetivo (precio_m2) multiplicado por la superficie: define la respuesta exacta."},
            {"columna": "dias_publicado", "motivo": "Censurada por la ventana de observacion: para un aviso nuevo siempre vale 0."},
            {"columna": "veces_visto", "motivo": "Censurada por la ventana de observacion: para un aviso nuevo siempre vale 1."},
            {"columna": "clave, url, usr_id, prp_id, id_publicado", "motivo": "Identificadores: no son atributos del inmueble."},
        ],
        "split": {"train": len(X_train), "test": len(X_test),
                   "firmas_compartidas_train_test": len(set(g_train) & set(g_test)),
                   "pct_avisos_con_firma_compartida": round(float((grupos.map(grupos.value_counts()) > 1).mean()), 3)},
        "piso": {"mae_mean": round(float(mae_piso.mean()), 0), "mae_std": round(float(mae_piso.std()), 0)},
        "piso_superficie": {"mae_mean": round(float(mae_p1.mean()), 0), "mae_std": round(float(mae_p1.std()), 0)},
        "mejor_piso": {"nombre": "P3 (mediana superficie x moneda)",
                       "mae_mean": round(float(mae_p3.mean()), 0), "mae_std": round(float(mae_p3.std()), 0)},
        "escalera_pisos": escalera_pisos,
        "comparacion_modelos": comparacion_modelos,
        "pares_contra_mejor": pares,
        "ablacion": ablacion,
        "columnas_finales": FINALES,
        "columnas_salen": salen,
        "mae_con_todas": round(float(base.mean()), 0),
        "mae_con_finales": round(float(mae_finales.mean()), 0),
        "grid_mejor_params": mejor_params,
        "grid_mejor_mae": round(float(-grilla.best_score_), 0),
        "grid_tabla": grid_tabla,
        "mae_sin_ajustar_finales": round(float(mae_finales.mean()), 0),
        "test": {
            "mae_modelo": round(float(mae_test), 0), "mae_piso": round(float(mae_piso_test), 0),
            "mae_piso_superficie": round(float(mae_p1_test), 0),
            "mae_mejor_piso": round(float(mae_p3_test), 0),
            "mejora_pct": round(1 - mae_test / mae_piso_test, 4),
            "mejora_vs_mejor_piso_pct": round(1 - mae_test / mae_p3_test, 4),
            "error_relativo_mediano_modelo": round(err_rel, 4),
            "error_relativo_mediano_piso": round(err_rel_piso, 4),
            "error_relativo_mediano_mejor_piso": round(err_rel_p3, 4),
            "mae_cv_elegido": round(float(-grilla.best_score_), 0),
        },
        "por_tramo": por_tramo,
        "por_localidad": por_localidad,
        "por_moneda": por_moneda,
        "importancia_permutacion": importancia,
        "fuga_superficie": fuga_superficie,
        "cortes_residuo": {"bajo": round(corte_bajo, 4), "alto": round(corte_alto, 4)},
        "etiquetas_test": {k: int(v) for k, v in etiquetas_test.items()},
        "cruce_veredicto_tramo_real": cruce_real,
        "cruce_veredicto_tramo_predicho": cruce_pred,
        "hallazgos_eda": hallazgos_eda,
        "evolucion_dataset": evolucion_final.to_dict(orient="records"),
        "semilla": SEMILLA,
    }

    joblib.dump(final, SALIDA / "modelo_final.joblib")
    explo.to_parquet(SALIDA / "dataset_exploracion.parquet", index=False)
    with open(SALIDA / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(_a_nativo(metrics), f, ensure_ascii=False, indent=2)
    with open(SALIDA / "rangos_columnas.json", "w", encoding="utf-8") as f:
        json.dump(_a_nativo(rangos), f, ensure_ascii=False, indent=2)

    print("\nListo. Artefactos en app/data/:")
    for p in sorted(SALIDA.iterdir()):
        print("  ", p.name, f"({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
