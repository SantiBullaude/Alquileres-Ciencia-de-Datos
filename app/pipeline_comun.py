"""Pipeline compartido entre build_app.py y streamlit_app.py.

Tiene que vivir en un modulo importable (no en el script que se ejecuta
como __main__): joblib guarda en el pickle la ruta del modulo de cada
funcion, y streamlit_app.py necesita poder resolver esa ruta al cargar
modelo_final.joblib.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
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


class MedianaPorGrupo(BaseEstimator, RegressorMixin):
    """Predice la mediana de y del grupo del aviso, con respaldo jerarquico.

    grupos: lista de niveles, del mas fino al mas grueso. Cada nivel es una tupla
    de columnas. 'tramo_sup' es la superficie cubierta partida en `n_tramos`
    cuantiles calculados SOLO con el fold de entrenamiento. Si el grupo tiene
    menos de `min_n` avisos se cae al nivel siguiente y, al final, a la global.
    """

    def __init__(self, grupos=(("tramo_sup",),), n_tramos=10, min_n=20):
        self.grupos = grupos
        self.n_tramos = n_tramos
        self.min_n = min_n

    def _preparar(self, X):
        Z = pd.DataFrame(index=X.index)
        sup = X["superficie_cubierta_m2"]
        Z["tramo_sup"] = np.searchsorted(self.cortes_, sup.fillna(self.sup_med_), side="right")
        Z["localidad"] = X["localidad"].where(X["localidad"].isin(LOCALIDADES_GRANDES), "Interior").fillna("Interior")
        Z["moneda"] = X["moneda"].fillna("ARS")
        Z["dormitorios"] = X["dormitorios"].fillna(-1).clip(upper=4)
        return Z

    def fit(self, X, y):
        sup = X["superficie_cubierta_m2"]
        self.sup_med_ = float(sup.median())
        qs = np.linspace(0, 1, self.n_tramos + 1)[1:-1]
        self.cortes_ = np.unique(np.quantile(sup.dropna(), qs))
        Z = self._preparar(X)
        y = pd.Series(np.asarray(y), index=X.index)
        self.global_ = float(y.median())
        self.tablas_ = []
        for nivel in self.grupos:
            g = y.groupby([Z[c] for c in nivel])
            agg = g.agg(["median", "size"])
            self.tablas_.append(agg[agg["size"] >= self.min_n]["median"].to_dict())
        return self

    def predict(self, X):
        Z = self._preparar(X)
        pred = np.full(len(Z), np.nan)
        for nivel, tabla in zip(self.grupos, self.tablas_):
            claves = list(zip(*[Z[c] for c in nivel])) if len(nivel) > 1 else list(Z[nivel[0]])
            vals = np.array([tabla.get(k, np.nan) for k in claves], dtype=float)
            pred = np.where(np.isnan(pred), vals, pred)
        return np.where(np.isnan(pred), self.global_, pred)

