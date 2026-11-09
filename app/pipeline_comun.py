"""Pipeline compartido entre build_app.py y streamlit_app.py.

Tiene que vivir en un modulo importable (no en el script que se ejecuta
como __main__): joblib guarda en el pickle la ruta del modulo de cada
funcion, y streamlit_app.py necesita poder resolver esa ruta al cargar
modelo_final.joblib.
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_validate

SEMILLA = 42

LOCALIDADES_GRANDES = ["Capital", "Godoy Cruz", "Guaymallen", "Lujan de Cuyo", "Las Heras", "Maipu"]
INICIO_VENTANA = pd.Timestamp("2026-09-07")

COLUMNAS = [
    "superficie_cubierta_m2", "superficie_total_m2", "dormitorios", "banos",
    "antiguedad", "cochera", "estado_conservacion", "valor_expensas",
    "tiene_expensas", "acepta_mascotas", "localidad", "zona_geo",
    "dist_centro_km", "densidad_1km", "tipo_anunciante", "fuente",
    "fecha_scraping", "moneda",
]
LOGARITMICAS = ["superficie_cubierta_m2", "superficie_total_m2", "valor_expensas"]
NUMERICAS = ["dormitorios", "banos", "antiguedad", "cochera", "dist_centro_km",
             "densidad_1km", "fecha_scraping", "moneda"]
CATEGORICAS = ["estado_conservacion", "tiene_expensas", "acepta_mascotas",
               "localidad", "zona_geo", "tipo_anunciante", "fuente"]


def derivar_columnas(X_):
    X_ = X_.copy()
    if "cochera" in X_:
        cochera = X_["cochera"].astype(object)
        X_["cochera"] = np.where(cochera.isna() | (cochera == "Indistinto"), np.nan,
                                  np.where(cochera == "Sin Cochera", 0.0, 1.0)).astype(float)
    if "estado_conservacion" in X_:
        X_["estado_conservacion"] = (X_["estado_conservacion"]
            .replace({"---": np.nan, "Recicled": "Bueno", "Regular": "Bueno", "Semi new": "Muy Bueno"})
            .fillna("sin_dato"))
    for col in ["tiene_expensas", "acepta_mascotas"]:
        if col in X_:
            X_[col] = X_[col].astype(str).str.lower().map({"true": "si", "false": "no"}).fillna("sin_dato")
    if "localidad" in X_:
        X_["localidad"] = X_["localidad"].where(X_["localidad"].isin(LOCALIDADES_GRANDES), "Interior")
    if "zona_geo" in X_:
        # pd.to_numeric antes del int(): el formulario de la app puede mandar
        # el valor como texto ("-1.0") en vez de float.
        X_["zona_geo"] = pd.to_numeric(X_["zona_geo"], errors="coerce").fillna(-2).astype(int).astype(str)
    if "moneda" in X_:
        X_["moneda"] = (X_["moneda"] == "USD").astype(float)
    if "fecha_scraping" in X_:
        X_["fecha_scraping"] = (pd.to_datetime(X_["fecha_scraping"]) - INICIO_VENTANA).dt.days
    return X_


def armar_pipeline(estimador, columnas=COLUMNAS, escalar=True):
    esc = [StandardScaler()] if escalar else []
    log_ = [c for c in LOGARITMICAS if c in columnas]
    num_ = [c for c in NUMERICAS if c in columnas]
    cat_ = [c for c in CATEGORICAS if c in columnas]
    partes = []
    if log_:
        partes.append(("log", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                              FunctionTransformer(np.log1p), *esc), log_))
    if num_:
        partes.append(("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), *esc), num_))
    if cat_:
        partes.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10), cat_))
    # Sin paso de seleccion explicito: ColumnTransformer ya elige sus columnas
    # por nombre, asi que alcanza con que el DataFrame de entrada las tenga.
    return Pipeline([
        ("derivar", FunctionTransformer(derivar_columnas)),
        ("preparar", ColumnTransformer(partes)),
        ("modelo", estimador),
    ])


def cv_agrupada(rep):
    return GroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA + rep)


def medir(modelo, X_, y_, g_, repeticiones=3):
    maes = []
    for rep in range(repeticiones):
        r = cross_validate(modelo, X_, y_, groups=g_, cv=cv_agrupada(rep),
                            scoring="neg_mean_absolute_error")
        maes.extend(-r["test_score"])
    return np.array(maes)
