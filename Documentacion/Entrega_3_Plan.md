# Entrega 3 - Guion de exposicion

25 minutos con el docente. Nos toca exponer **16 minutos**; despues vienen
6 de preguntas dirigidas y 3 de devolucion. La nota es **individual** y las
preguntas pueden caerle a cualquiera sobre cualquier parte: cada uno expone
su bloque, pero **todos leen el guion entero**.

**Antes de entrar:** el notebook `entrega3_modelado.ipynb` abierto en VS Code,
scrolleado arriba de todo. **No ejecutarlo**: ya tiene todas las salidas
guardadas y coinciden con los textos.

**Como usar este guion:** cada bloque dice que seccion del notebook mostrar,
que grafico señalar, que decir (en palabras simples) y con que conclusion
cerrar. No hace falta leerlo textual: hay que contar la idea con los numeros.

---

## Reparto

| Min | Bloque | Seccion del notebook | Graficos | Quien |
|---|---|---|---|---|
| 0-3 | 1. Que nos corrigieron | 0, 0.1, 0.2 | 1 y 2 | Integrante 1 |
| 3-7 | 2. El planteo | 1.1 a 1.6 | - | Integrante 2 |
| 7-10 | 3. La comparacion de modelos | 2.1 | 3 | Integrante 3 |
| 10-13 | 4. Columnas, ajuste y examen final | 2.2, 2.3, 3.1 | 4 | Integrante 4 |
| 13-15 | 5. Donde se equivoca y en que se apoya | 3.2, 3.3 | 5 y 6 | Integrante 5 |
| 15-16 | 6. La superficie no es fuga + lo que no anduvo | 3.4, 4 | 7 | Integrante 1 |
| 16-22 | Preguntas dirigidas | - | - | Todos |
| 22-25 | Devolucion | - | - | Anotar todo |

> Completar la columna "Quien" con los nombres. El integrante 1 habla dos
> veces porque su primer bloque es corto.

---

## Bloque 1 - Que nos corrigieron (min 0-3)

**Mostrar:** secciones 0, 0.1 y 0.2. Graficos 1 y 2.

**La frase de apertura:**
> "En la Entrega 2 nos marcaron dos hipotesis, H1 y H3. Las dos estan
> resueltas y las dos cambiaron el modelo."

### H1 - Pasar todo a pesos (grafico 1, las dos cajas)

- **Antes:** dejamos afuera los avisos en dolares porque decíamos que el tipo
  de cambio era un supuesto que no se podia sostener.
- **Por que estaba mal:** en toda la ventana (7/9 al 13/10) el dolar se movio
  **menos de 2 %**. Si se puede sostener.
- **Que hicimos:** convertimos con el **dolar MEP del 13/10, $1.540**. Entran
  **269 avisos mas**.
- **Señalar en el grafico:** "con los dos en pesos las cajas casi se tocan;
  ya no son dos mundos. Pero la de dolares esta mas a la derecha: piden un
  **54 % mas por m2**."
- **Conclusion:** los avisos en dolares son un segmento mas caro. Por eso el
  modelo recibe la columna `moneda`, que avisa "este se publico en dolares".

### H3 - La distancia, separando por metros (grafico 2, las 5 lineas)

- **Antes:** la correlacion entre distancia al centro y precio por m2 daba
  casi cero (-0,087). La sacamos del modelo.
- **Que hicimos:** separamos los deptos en 5 sectores de superficie y medimos
  la distancia dentro de cada uno.
- **Señalar en el grafico:**
  - "La linea azul son los deptos chicos, hasta 40 m2: baja al alejarse del
    centro. Ahi la correlacion da **-0,27, amarilla**."
  - "Las demas tienen forma de **U**: bajan hasta los 4-7 km y vuelven a
    subir, que es donde esta Chacras, Dalvian, Palmares."
- **Conclusion:** la correlacion global daba cero porque una U sube y baja a
  la vez. La distancia si importa, y **vuelve a entrar al modelo**.

---

## Bloque 2 - El planteo (min 3-7)

**Mostrar:** secciones 1.1 a 1.6, sin graficos. Ir señalando las salidas
impresas.

**Que predecimos (1.1):**
> "Predecimos el precio por m2 en pesos de un aviso. Entran **3.364 avisos**,
> despues de sacar los que no tienen precio y los que el pipeline marco como
> sospechosos."

**La metrica (1.2): MAE.**
- Es "por cuantos pesos por m2 le erramos en promedio". Se dice en pesos.
- No usamos RMSE porque eleva los errores al cuadrado: un par de avisos mal
  cargados (hay uno de 99.000 $/m2 con mediana de 10.500) pesarian demasiado.

**La escalera de pisos (1.4): de 2.667 a 2.285 $/m2.**
> "Armamos una escalera de pisos: la mediana global (P0: **2.667**), la mediana
> por tramo de superficie (P1: **2.398**) y la mediana por superficie y moneda
> (P3: **2.285**, el mejor piso). Si el modelo no le gana con holgura al mejor
> piso, no justifica su complejidad."

**Como partimos los datos (1.3 y 1.6) - lo mas importante del bloque:**
- El mismo depto puede estar publicado en los dos portales. Si una copia cae
  en entrenamiento y la otra en test, el modelo "se copia".
- Por eso armamos una **firma** (direccion + localidad + metros) y la misma
  firma **nunca queda de los dos lados**. El 24 % de los avisos comparte
  firma con otro.
- **Señalar la salida de 1.6:** "lo medimos: al azar el error da **1.651**,
  por firma da **1.750**. Esos **99 pesos** eran el modelo copiandose."

**Pipeline (1.5), en una oracion:**
> "Todo lo que se le hace a los datos (completar vacios, logaritmos, pasar
> categorias a numeros) esta adentro del pipeline, asi se calcula solo con
> el entrenamiento y no ve el test."

---

## Bloque 3 - La comparacion de modelos (min 7-10)

**Mostrar:** seccion 2.1. Tabla de modelos, grafico 3 (las cajas) y tabla
"pedazo contra pedazo".

**Que decir:**
> "Probamos la escalera de pisos y cuatro familias de modelos, todos con las
> mismas columnas y los mismos 15 pedazos de validacion: una lineal, una
> lineal sobre el logaritmo del precio, un Random Forest y un
> HistGradientBoosting."

**Señalar en el grafico 3:**
- Cada caja son los 15 errores de un modelo. **Mas a la izquierda es mejor.**
- "La mediana global esta a la derecha en 2.667; la mediana por superficie baja
  a 2.398 y por superficie × moneda a 2.285."
- "El boosting es el de mas a la izquierda: **1.750**."

**Señalar la tabla de abajo (pedazo contra pedazo):**
- Contra el mejor piso (P3) gana en **15 de 15** pedazos, por **535** pesos (y
  el modelo ajustado final le gana por **575** pesos, desvio 80).
- Contra la lineal gana en **15 de 15** pedazos, por **199** pesos, con un
  desvio de 84.
- Contra la lineal en log y el bosque gana por **~70** pesos, en **14 de 15**.

**Conclusion:**
> "La consigna dice que si el complejo no le gana al simple por mas que el
> desvio, se queda el simple. Aca le gana siempre y por el doble del desvio:
> **elegimos HistGradientBoosting**."

---

## Bloque 4 - Columnas, ajuste y examen final (min 10-13)

**Mostrar:** secciones 2.2, 2.3 y 3.1. Grafico 4.

**Columnas (2.2):**
> "Sacamos cada columna de a una y medimos si el modelo empeoraba. Quedaron
> **13 de 18**, y con esas 13 el modelo baja de **1.750 a 1.730 $/m2**."

- Las que mas aportan: **superficie (+252)** y **moneda (+156)**, que es la
  de la correccion de H1.
- Las 5 que salieron: `tiene_expensas`, `acepta_mascotas`, `tipo_anunciante`,
  `fecha_scraping` y `superficie_total_m2`. Sacarlas mejora el error.

**Ajuste (2.3), grafico 4:**
- Cada punto es una combinacion de "perillas" del modelo; probamos 32, de
  mejor (izquierda) a peor (derecha).
- **Señalar:** "todas las azules van antes que las naranjas. Las azules
  castigan el error comun y las naranjas el error al cuadrado: confirma que
  MAE era la metrica correcta."
- **Señalar la linea punteada:** "es el modelo sin ajustar (1.730). El mejor
  llega a **1.705**: el ajuste mejoro **25 pesos**, menos que el desvio. Los
  valores por defecto ya estaban bien."

**Examen final (3.1):**
> "Recien ahora miramos el test, una sola vez. El modelo erra **1.722** contra
> **2.258** del mejor piso P3 (**24 % menos error**) y contra **2.778** de la
> mediana global (**38 % menos error**). Y es casi igual a lo que daba en
> validacion (**1.705**): la validacion no nos estaba mintiendo."

**Es mucho o poco:**
> "En un depto de 50 m2 son unos $86.000 por mes de error (11,4 % de error
> relativo mediano). La app no puede marcar como caro un aviso que pide 5 %
> mas: solo los que se alejan ~20 % o mas."

---

## Bloque 5 - Donde se equivoca y en que se apoya (min 13-15)

**Mostrar:** secciones 3.2 y 3.3. Graficos 5 y 6.

**Grafico 5, izquierda (nube de puntos):**
- Cada punto es un aviso del test: abajo lo que pide, a la izquierda lo que
  predijo. Si fuera perfecto, todos caerian sobre la linea roja.
- "La nube sigue la linea pero mas acostada: el modelo tira todo hacia el
  medio."

**Grafico 5, derecha (barras):**
- "Partimos los avisos en 5 grupos de mas barato a mas caro. Con los baratos
  predice **de mas** (+1.442) y con los caros **de menos** (-3.063)."
- **Por que, desde el dominio:** lo que hace caro un depto en Mendoza (el
  edificio, las terminaciones, si esta amoblado, la cuadra exacta) esta en la
  descripcion y las fotos, que el modelo no ve.

**Tablas de localidad y moneda (sin grafico), una frase cada una:**
- Erra mas en Maipu, Lujan y Capital, que mezclan zonas caras con comunes.
- En los avisos en dolares erra el doble (3.480 contra 1.608), pero sumarlos
  no empeoro los de pesos.

**Grafico 6 (barras de importancia):**
- "Desordenamos cada columna y medimos cuanto empeora el error. La
  **superficie** es lejos la mas importante, despues **moneda**, despues
  distancia al centro, cochera y expensas."

---

## Bloque 6 - La superficie no es fuga + lo que no anduvo (min 15-16)

**Mostrar:** secciones 3.4 y 4. Grafico 7.

**La sospecha:**
> "La consigna dice que si la columna que mas pesa sorprende, hay que
> sospechar de una fuga. La superficie esta en la formula del objetivo, asi
> que la revisamos."

**Cuatro chequeos, rapido:**
1. Existe al predecir: la publica el aviso.
2. No contiene la respuesta: si un depto tiene el doble de metros, el
   alquiler no sube el doble, sube ~64 %. Por eso el m2 de los grandes es mas
   barato.
3. Sin ella el modelo sigue ganandole al piso (2.186 contra 2.667).
4. Con una sola columna (solo superficie) el boosting da **2.390** en CV
   (**2.406** en test), empatado con la mediana por superficie (2.398): las
   otras 12 columnas bajan el error en **685 $/m2** hasta **1.705**.

**Grafico 7 (la curva):**
> "Le cambiamos solo la superficie a todos los avisos. La curva baja suave,
> como el mercado real. Si fuera trampa veriamos saltos raros."

**Lo que no anduvo (seccion 4), cierre:**
> "Lo que probamos y no sirvio: partir al azar (era fuga), 5 columnas que no
> aportaron, castigar el error al cuadrado, y el ajuste de la grilla, que solo
> mejoro 25 $/m2. Y para el veredicto mantuvimos un unico par de cortes
> (-20,2 % / +25,8 %): al cruzar por tramo de precio predicho (lo que ve la app)
> el reparto es parejo (~73–85 % en linea). Lo que falta para errar menos en los
> caros es la descripcion del aviso."

---

## Preguntas probables (min 16-22)

Cualquiera puede recibir cualquiera. Respuestas cortas:

**"Por que no usaron el precio?"**
Porque con la superficie da el objetivo exacto, y porque es lo que queremos
evaluar: si ya lo tenes, no hay nada que predecir.

**"Por que una sola cotizacion del dolar?"**
Porque se movio menos de 2 % en toda la ventana. Al lado de un error de 11,4 %
no cambia nada.

**"`moneda` no es trampa?"**
No. Dice en que moneda se publico, no cuanto pide. El aviso la trae.

**"Por que partieron por firma y no al azar?"**
Porque hay deptos publicados dos veces. Al azar el error daba 99 pesos mas
bajo, y era falso.

**"Por que no por fecha?"**
La fecha de cada aviso es la ultima vez que lo vimos, y el 58 % tiene la
misma (13/10). No hay un orden limpio para cortar.

**"Por que mediana en el piso y no promedio?"**
Porque con MAE el numero fijo que menos erra es la mediana. Y ademas armamos
una escalera de medianas por grupo (superficie y superficie × moneda).

**"Por que MAE y no RMSE?"**
RMSE se deja arrastrar por los avisos mal cargados. Y la grilla lo confirma:
las 16 mejores combinaciones usan error absoluto.

**"La distancia pesa en el grafico 6 (~170) pero sacarla cambiaba poco
(+6). Como es?"**
Sacarla pregunta si la informacion esta en otro lado (si: `localidad` tambien
dice donde queda). Desordenarla pregunta si este modelo la usa (si, y le
duele). La ubicacion importa, pero ninguna columna sola es imprescindible.

**"Por que no miraron el test antes?"**
Porque si miras el test, cambias algo y volves a mirar, deja de ser un
examen. Todo se eligio con validacion cruzada.

**"Hay clases desparejas?"**
Es regresion, no hay clases. Lo equivalente es la cola de precios muy altos,
y por eso elegimos MAE.

**Si no saben algo:** "No lo probamos; lo hariamos asi..." suma mas que
inventar.

---

## Numeros para saber de memoria

| Que | Numero |
|---|---|
| Avisos en el modelo | 3.364 (269 en USD convertidos) |
| Cotizacion | MEP 13/10: $1.540 (se movio < 2 % en la ventana) |
| USD vs ARS | 54 % mas caros por m2 |
| H3, deptos de hasta 40 m2 | -0,27 (amarilla) |
| Escalera de pisos (CV / test) | P0 global: 2.667 / 2.778 · P1 superficie: 2.398 / 2.420 · **P3 superficie × moneda: 2.285 / 2.258** |
| Modelo elegido | **1.705** (validacion) / **1.722 (test)** |
| Mejora sobre los pisos (test) | **24 % menos error que el mejor piso P3** (y **38 % menos que P0 global**) |
| Error relativo mediano (test) | **11,4 %** (mejor piso P3: 14,4 % · piso global P0: 19,1 %) |
| Fuga al partir al azar | 99 (1.651 vs 1.750) |
| Boosting vs lineal | gana por **199**, en 15 de 15 (desvio 84); vs lineal log y RF por **~70** en 14 de 15 |
| Columnas finales | **13 de 18** (salen `tiene_expensas`, `acepta_mascotas`, `tipo_anunciante`, `fecha_scraping`, `superficie_total_m2`) |
| Mejora de la grilla | **25 pesos** (de 1.730 a 1.705) |
| Chequeo una sola columna | HGB solo superficie: **2.390** en CV (2.406 en test) vs **1.705** con las 13 columnas; sin superficie: **2.186** |
| Superficie: doble de metros | el alquiler sube ~64 % (pendiente 0,71) |
