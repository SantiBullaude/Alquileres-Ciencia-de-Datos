# Brujula Inmobiliaria

Proyecto Integrador - Ciencia de Datos 2026 (UTN FRM).

## La pregunta

Un aviso de alquiler en Mendoza, ¿esta sobrevalorado, en linea con el
mercado, o es una oportunidad? El modelo es supervisado (regresion sobre
precio_m2 + analisis del residuo); el detalle de la propuesta completa esta
en `Documentacion/Propuestas_Proyecto_Integrador_Inmobiliario.docx`.

## Fuentes y unidad de analisis

- **Segmento:** departamentos en alquiler (no temporario), Mendoza.
- **Unidad de analisis:** un aviso publicado. "Una fila es un departamento
  en alquiler publicado en Inmoclick o InmoUP para la provincia de Mendoza,
  en el momento del scraping."
- **Columna objetivo:** `precio_m2` (precio / superficie cubierta).
- **Clave primaria:** `clave` = `<fuente>:<usr_id>-<prp_id>`.

| Portal | Rol | Filas | robots.txt |
|---|---|---|---|
| Inmoclick | principal | 1.111 | sin reglas de exclusion |
| InmoUP | principal | 1.393 | permite sitemaps; prohibe `/json/` |
| ZonaProp | descartado por ahora | 1.330 disponibles | solo paginas 2-5 de busqueda |
| Argenprop | descartado por ahora | 127 disponibles | solo paginas 1-3 de busqueda |

ZonaProp y Argenprop quedaron afuera de esta entrega porque su `robots.txt`
no permite paginar los resultados de busqueda: para bajarlos completos hay
que armar el pipeline desde sus sitemaps, que es un camino distinto. Con
los dos portales que si estan, el dataset llega a **2.504 avisos** (objetivo
de la catedra: mas de 1.000).

### Cosas que aparecieron al mirar el HTML real

1. **El ID del aviso no es unico por si solo.** En
   `inmoclick.com/4335-llopart-inmobiliaria/inmuebles/3059/ficha/...` el
   `3059` es el numero del aviso *dentro de esa inmobiliaria*: hay decenas
   de avisos distintos con `prp_id=1`, uno por cada dueño directo. La ficha
   lo confirma mostrando `ID# 4335-3059`. Y como los dos portales usan el
   mismo esquema, la clave tambien lleva el nombre de la fuente.
2. **Hay dos monedas mezcladas.** La mayoria publica en pesos, unos pocos
   en dolares. `precio_m2` queda expresado en la moneda del aviso, y el
   dataset lleva la columna `moneda` para que el modelo segmente o
   convierta antes de comparar.
3. **La descripcion de la ficha de Inmoclick viene ofuscada.** La parten en
   spans e intercalan textos tipo "SUSTRAIDO DE INMOCLICK.COM.AR" para que
   no se copie. La misma descripcion aparece limpia en la tarjeta del
   listado, asi que se toma de ahi.
4. **InmoUP publica JSON-LD.** Cada ficha trae un bloque schema.org con los
   atributos en `additionalProperty`, usando las mismas etiquetas que la
   tabla de Inmoclick. Se lee de ahi: es mas estable que scrapear HTML y no
   depende de nombres de clases CSS. Ademas es el unico de los dos que
   publica la fecha del aviso (`fecha_publicacion`).
5. **InmoUP no se puede paginar.** La grilla se completa por JavaScript,
   `?page=N` devuelve siempre la primera tanda, y los endpoints JSON que
   usa el sitio estan prohibidos en su `robots.txt`. Lo que ese mismo
   `robots.txt` si permite son los sitemaps, y ahi estan listadas las
   busquedas ya recortadas por barrio, superficie y dormitorios. El
   scraper recorre esas ~670 busquedas de Mendoza y une los resultados.

## Arquitectura del pipeline

Cinco tareas de Airflow, con la capa bronce (`data/raw/`, HTML crudo tal
como llega) separada de la capa plata (el CSV final en `resultados/`):

    extract_listings -> parse_raw -> transform_clean -> quality_check -> export_csv

| Archivo | Que hace |
|---|---|
| `dags/brujula_pipeline.py` | El DAG: solo el orden de las tareas |
| `dags/brujula/fuentes/inmoclick.py` | Bajar y parsear Inmoclick |
| `dags/brujula/fuentes/inmoup.py` | Bajar y parsear InmoUP |
| `dags/brujula/fuentes/comun.py` | Vocabulario compartido entre portales |
| `dags/brujula/extract.py` | Recorre las fuentes -> `data/raw/AAAAMMDD/<fuente>/` |
| `dags/brujula/parse.py` | Junta los registros de todas las fuentes |
| `dags/brujula/transform.py` | Tipado, clave primaria, `precio_m2`, marcas |
| `dags/brujula/quality.py` | Los seis chequeos -> `quality_report.txt` |
| `dags/brujula/export.py` | El CSV final |
| `dags/brujula/pipeline.py` | Corre las cinco seguidas, sin Airflow |

Agregar un portal nuevo es agregar un modulo en `fuentes/` con
`descargar()` y `parsear()`, y registrarlo: el resto del pipeline no se
toca.

La logica vive en el paquete `dags/brujula/` y no adentro del DAG, para
poder probarla desde una terminal sin levantar Airflow. Esta dentro de
`dags/` porque Airflow monta esa carpeta y la agrega al `sys.path`.

## Donde queda cada cosa

    data/raw/AAAAMMDD/<fuente>/   HTML crudo, capa bronce      (no se versiona, ~1,7 GB)
    data/processed/               intermedios de trabajo        (no se versiona)
    resultados/AAAAMMDD/          CSV final + reporte de calidad (SI se versiona)
    notebooks/                    analisis exploratorio (Entrega 2)

`resultados/` es la excepcion a la regla de "los datos no van al repo": son
los dos archivos que se entregan y se defienden, pesan un par de MB, y asi
el resto del equipo y la catedra los pueden abrir sin levantar Airflow ni
volver a scrapear. Una carpeta por fecha de scrapeo, para poder comparar
como se movio el mercado entre corridas sin pisar la anterior.

## Si recien clonaste el repo

Antes que nada, **que queres hacer**, porque no siempre hace falta levantar
Airflow:

| Si vas a... | Necesitas |
|---|---|
| Analizar o modelar el dataset | Nada mas: el CSV ya esta en `resultados/` |
| Reprocesar la capa bronce | Airflow + la carpeta `data/raw/`, que **no se versiona** (pesa ~2 GB): te la tiene que pasar alguien del grupo |
| Scrapear de cero | Airflow y unas 3 horas |
| Regenerar el notebook | Airflow + las librerias del final de este README |

Tres cosas del proyecto **no estan en el repo a proposito** y hay que
reponerlas a mano: el `docker-compose.yaml` oficial (se baja), el `.env`
(se copia del ejemplo) y `data/raw/` (se scrapea o se copia).

## Como levantar el entorno

```bash
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml'
cp .env.example .env
docker compose up airflow-init
docker compose up -d
# UI en http://localhost:8080 (usuario y contraseña: airflow)
```

> El `docker-compose.yaml` oficial no se versiona (ver `.gitignore`): cada
> integrante lo baja una vez con el comando de arriba. Lo que si esta
> versionado es `docker-compose.override.yml`, que Compose junta solo con
> el oficial y agrega tres cosas: montar `data/` y `resultados/` adentro
> del contenedor (sin eso el HTML crudo y el CSV quedarian encerrados ahi),
> montar `notebooks/`, y apagar los DAGs de ejemplo. Las librerias se
> instalan via `_PIP_ADDITIONAL_REQUIREMENTS` en el `.env`.

Despues, en la UI: activar el DAG `brujula_inmobiliaria_pipeline`. **Desde
la Entrega 2 corre solo, todos los dias a las 06:00** (`schedule="0 9 * * *"`,
en UTC), asi que alcanza con dejarlo activado y los contenedores
levantados: el scheduler vive adentro de Docker, no en el sistema
operativo, asi que con la maquina apagada no corre y, por `catchup=False`,
los dias perdidos no se recuperan.

Igual se puede disparar a mano, y al hacerlo se le pueden pasar cuatro
parametros:

| Parametro | Para que |
|---|---|
| `max_paginas` | Prueba corta: recorta cuantas busquedas se piden por portal |
| `fuentes` | Un solo portal, ej. `["inmoup"]` |
| `fecha` | `AAAAMMDD` de la capa bronce a usar |
| `acumular_historico` | `True` (por defecto) arma el dataset con **todas** las fechas de `data/raw/`, no solo con la de esta corrida |

> **Ojo con `fecha`.** El contenedor corre en UTC, no en hora argentina:
> despues de las 21 hs locales el "hoy" de Airflow ya es el dia siguiente,
> y una corrida sin `fecha` se arma una capa bronce nueva y vuelve a
> scrapear los ~2.400 avisos desde cero. Para reprocesar lo que ya esta en
> disco (por ejemplo despues de corregir algo en la transformacion), hay
> que pasar la fecha de esa carpeta.

### Sin Docker, para desarrollar

```bash
pip install -r requirements.txt
cd dags
python -m brujula.pipeline --max-paginas 1    # prueba rapida
python -m brujula.pipeline --fuentes inmoup   # un solo portal
python -m brujula.pipeline                    # corrida completa (~1 h la primera vez)
```

## Lo que hay que saber explicar en la defensa

**¿Que es una fila?** Un aviso de departamento en alquiler publicado en
Inmoclick o InmoUP para Mendoza, en el momento del scraping. No una
propiedad: si el mismo departamento se publica en los dos portales, son dos
avisos y por lo tanto dos filas.

**¿Cual es la columna objetivo?** `precio_m2`. El modelo estima el esperado
y el residuo (real menos esperado) da el veredicto sobrevalorado / de
mercado / oportunidad.

**¿Cual es la clave primaria?** `clave` = `fuente:usr_id-prp_id`. Ver el
punto 1 de arriba: el numero del aviso solo no alcanza.

**¿Y si la misma propiedad esta en los dos portales?** Pasa: hay 185 avisos
marcados en `posible_duplicado_cruzado`. No se borran, porque la unidad de
analisis es el aviso publicado y son dos avisos reales. Pero se marcan para
que la etapa de modelado los pueda filtrar: si no, esas propiedades pesarian
el doble al entrenar. La marca sale de una huella armada con direccion
normalizada, localidad y superficie cubierta.

**¿Por que hay nulos?** Porque los portales dejan campos opcionales y el
que publica los completa o no. `dormitorios` y `banos` casi siempre estan;
`antiguedad`, `piscina` o `amoblado` faltan seguido, sobre todo en avisos
de dueño directo. Hay ademas nulos que dependen de la fuente:
`tipo_construccion` y `id_publicado` solo los publica Inmoclick, y
`fecha_publicacion` solo InmoUP; por eso rondan el 42-58% de nulos, que es
justo la proporcion de filas del otro portal. No se imputa nada en esta
entrega: el `quality_report.txt` muestra la proporcion real por columna.
Las columnas que quedan 100% vacias se descartan en `transform_clean` y
queda registrado en el log cual fue.

**¿Hay valores raros?** Si, y estan marcados en `motivo_sospecha` (40
avisos). Son errores de carga del que publica: departamentos de 150.000 m2
(un tipeo) y alquileres de US$ 850.000 por mes (el precio de venta cargado
en un aviso de alquiler). Se marcan con reglas de dominio explicitas, no
con estadistica, y no se borran: son avisos publicados de verdad.

**¿Que pasa si se corre de nuevo?** La capa bronce es reanudable: no vuelve
a pedir el HTML que ya esta en disco, asi que una corrida cortada se retoma
sin castigar al servidor, y reprocesar la misma fecha tarda minutos en vez
de una hora. La clave primaria se deduplica en `transform_clean`, asi que
reprocesar no duplica filas. Si se corre otro dia se crea otra carpeta
`data/raw/AAAAMMDD/` y la anterior queda intacta — eso permite comparar el
mercado entre fechas, pero hay que pasar el parametro `fecha` si lo que se
quiere es reprocesar una capa bronce vieja (ver la nota de arriba sobre
UTC).

**¿El scraping da lo mismo cada vez?** No, y esta bien que no: entre la
corrida local y la de Airflow, dos avisos de Inmoclick se dieron de baja
(706 -> 704 filas). El mercado se mueve, y por eso cada corrida guarda su
propia capa bronce fechada en vez de pisar la anterior.

**¿Donde queda el dato crudo?** En `data/raw/AAAAMMDD/<fuente>/`, el HTML
tal como llego, sin tocar. Separado a proposito: cuando aparecio un bug en
la transformacion (los precios se estaban multiplicando por diez), se
arreglo y se reproceso sin volver a scrapear nada.

**¿Como pasan los datos de una tarea a otra?** Por archivo, no por XCom.
`parse_raw` escribe los registros en `data/processed/_registros.json` y
devuelve la ruta. Son miles de registros con descripciones largas: pasarlos
por XCom meteria varios MB adentro de la base de metadatos de Airflow, que
no esta para eso.

**¿El dataset es tidy?** Casi. Una fila = un aviso, una columna = una
variable, un valor por celda. La excepcion conocida es `descripcion`: texto
libre donde el que publica mete amenities, condiciones y expensas todo
junto. Es el mismo caso del ejemplo de FIFA de la catedra; en la Unidad 2
se puede derivar de ahi variables binarias (tiene cochera, acepta mascotas)
para complementar los campos estructurados.

**¿Es legal scrapear esto?** Se respeta el `robots.txt` de cada portal: por
eso InmoUP se recorre por sitemaps y no por sus endpoints JSON, y por eso
ZonaProp y Argenprop quedaron afuera. El pipeline se identifica con un
User-Agent propio y espera 1 segundo entre pedidos.

## Entrega 2: que cambio

Las dos correcciones que marco la catedra sobre la Entrega 1, y el analisis
exploratorio. El plan completo esta en
`Documentacion/Entrega_2_Plan.md`; el analisis, en
`notebooks/entrega2_eda.ipynb`.

**1. Scheduling.** El DAG paso de `schedule=None` a `"0 9 * * *"` (06:00 en
Argentina), con `catchup=False`, `max_active_runs=1` y dos reintentos. Sin
catchup a proposito: un backfill no traeria los avisos de esa fecha, porque
los portales muestran solo lo que esta publicado hoy. Correr hacia atras
scrapearia N veces el inventario actual bajo N fechas distintas.

**2. Latitud y longitud usadas de verdad.** Estaban en el CSV pero crudas.
Ahora `transform.py` las convierte en cuatro features: `dist_centro_km`
(haversine a Plaza Independencia), `zona_geo` (KMeans sobre el aglomerado),
`densidad_1km` (vecinos en 1 km) y `tiene_geo`. Antes de derivarlas, anula
las coordenadas imposibles: habia avisos de Mendoza ubicados en Salta y en
Peru.

### Regenerar el notebook

El `.ipynb` se genera con `notebooks/build_nb.py` y no se edita a mano: es
el entregable, y tiene que poder recalcularse contra un CSV nuevo cada vez
que el pipeline suma avisos.

Una sola vez por contenedor, para instalar las herramientas (no estan en el
`.env` porque solo hacen falta en uno):

```bash
docker exec alquileres-ciencia-de-datos-airflow-triggerer-1 \
    python -m pip install matplotlib nbformat nbconvert ipykernel
```

Despues, cada vez que se quiera actualizar el analisis:

```bash
docker exec alquileres-ciencia-de-datos-airflow-triggerer-1 \
    python /opt/airflow/notebooks/build_nb.py \
    /opt/airflow/resultados/20260922/brujula_inmobiliaria_alquiler_mendoza.csv

docker exec alquileres-ciencia-de-datos-airflow-triggerer-1 \
    python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=900 /opt/airflow/notebooks/entrega2_eda.ipynb
```

El primero arma las celdas; el segundo las ejecuta y guarda las salidas
adentro del archivo, que es lo que pide la consigna. El notebook se abre con
VS Code sin necesidad de kernel, porque las salidas ya estan guardadas.

**3. El dataset acumula fechas.** Cada scrapeo es una foto del inventario
publicado ese dia, no un incremento. `parse_raw` ahora lee todas las fechas
de `data/raw/` y `transform_clean` deduplica por `clave` quedandose con la
observacion mas reciente. Entre corridas consecutivas aparecen entre 17 y 35
avisos nuevos por dia, asi que la union suma filas reales.

## Entrega 4: la aplicacion

La version final del proyecto: una app de Streamlit, publicada en
**[completar URL de Streamlit Community Cloud]**, que cuenta los hallazgos
de las tres entregas anteriores (el tallo: *El proyecto*, *Los datos*, *Lo
que encontramos*, *El modelo*) y deja usarlos (la copa: *Explora* y *Proba
el modelo*). El codigo vive en `app/`.

| Archivo | Que hace |
|---|---|
| `app/build_app.py` | Toma el CSV mas reciente de `resultados/`, entrena el mismo pipeline de `notebooks/entrega3_modelado.ipynb` (misma semilla, mismos folds por firma de inmueble) y deja en `app/data/` el modelo entrenado, el dataset con predicciones, y todos los numeros de los "Como lo sabemos" |
| `app/streamlit_app.py` | La app: lee `app/data/` y no reentrena nada |
| `app/requirements.txt` | Dependencias para Streamlit Community Cloud |

### Como se regenera

Si el pipeline suma avisos y hay un CSV mas nuevo en `resultados/`, correr:

```bash
cd app
python build_app.py        # tarda unos 15-20 min: compara 4 modelos, hace
                            # ablacion de 18 columnas y ajusta la grilla de
                            # hiperparametros, todo con validacion cruzada
```

Esto sobrescribe `app/data/modelo_final.joblib`, `app/data/metrics.json`,
`app/data/dataset_exploracion.parquet` y `app/data/rangos_columnas.json`.
Esos cuatro archivos se versionan en el repo (pesan unos pocos MB) para que
Streamlit Community Cloud no tenga que reentrenar en cada arranque.

### Como se corre localmente

```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Como se publica

En [share.streamlit.io](https://share.streamlit.io), conectar este
repositorio de GitHub y apuntar el "main file path" a `app/streamlit_app.py`.
Streamlit Cloud instala `app/requirements.txt` solo. Hay que mantenerla
publicada: si se borra o se duerme la app, conviene volver a abrirla antes
de la exposicion.

### Numeros clave de la app (salen de `app/data/metrics.json`)

| Que | Numero |
|---|---|
| Avisos modelados | 3.364 de 3.539 (269 en USD convertidos a pesos) |
| Comparacion de modelos (MAE, validacion cruzada) | Piso 2.667 · Lineal 1.949 · Lineal log(y) 1.822 · Random Forest 1.820 · **HistGradientBoosting 1.750** |
| Columnas finales | 13 de 18 (salen `tiene_expensas`, `acepta_mascotas`, `tipo_anunciante`, `fecha_scraping`, `superficie_total_m2`) |
| Mejor combinacion de hiperparametros | tasa 0,1 · 500 arboles · hojas de 15 nodos · minimo 10 filas por hoja → MAE 1.705 |
| Test (una sola medicion) | Modelo 1.722 $/m² (11,4 % de error relativo) contra 2.778 $/m² del piso (19,1 %) → **38 % menos error** |
| Cortes de residuo para el veredicto | -20,2 % / +25,8 % (percentiles 10 y 90 del error en validacion) |

## Estado

- [x] Fase 0 - estructura del repo y DAG esqueleto
- [x] Fase 1 - extraccion real (extract_listings, parse_raw)
- [x] Fase 2 - transformacion y calidad (transform_clean, quality_check)
- [x] Fase 3 - correr el DAG completo en Airflow y dejarlo en verde
- [x] Fase 4 - Entrega 2: scheduling, features geograficas, EDA
- [x] Fase 5 - Entrega 3: modelado (HistGradientBoosting)
- [x] Fase 6 - Entrega 4: aplicacion Streamlit publicada
- [ ] Ensayo cronometrado de la exposicion final (25/11)
