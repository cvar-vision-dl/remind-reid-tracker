# Metrics Summary For TFM

Este documento resume, en formato breve y mas academico, las metricas y salidas generadas por el pipeline de evaluacion de tracking e identidad en `REMIND/testing`. Su objetivo es servir como base reutilizable para la redaccion del TFM y para tablas o secciones de resultados.

## 1. Objetivo de la evaluacion

La evaluacion no se limita a medir si una deteccion coincide geometricamente con un objeto GT en un frame aislado, sino que analiza de forma conjunta:

- la correcta deteccion de novedad, es decir, distinguir objetos nuevos frente a objetos ya existentes
- la correcta asociacion de identidad a lo largo del tiempo
- la estabilidad temporal de los tracks
- la calidad de las decisiones inciertas o provisionales
- el efecto de modulos internos de resolucion, como distancia o contexto estructural

Por este motivo, se guardan salidas a varios niveles: caso, objeto, frame, clase, escena y batch global.

## 2. Niveles de analisis

### 2.1. Caso

La unidad basica de evaluacion es el caso, definido como una observacion GT visible en un frame concreto. A este nivel se registra:

- el estado real del objeto, nuevo o existente
- la decision final del sistema
- la decision colapsada usada para evaluar
- la IoU observada
- el contexto local de dificultad, como tamano relativo del objeto o numero de distractores
- la telemetria de modulos internos, por ejemplo distancia, neighbor sets y contexto

Este nivel es el mas importante para analisis posteriores, ya que permite reconstruir casi cualquier agregado offline.

### 2.2. Objeto

El nivel objeto resume el comportamiento temporal de cada instancia GT a lo largo de todos los frames en los que aparece. Permite medir:

- estabilidad de identidad
- numero de ids utilizados
- fragmentacion
- recuperacion de identidad
- tasa de seguimiento correcto del objeto

### 2.3. Frame

El nivel frame permite estudiar la dificultad instantanea de la escena y la evolucion temporal del sistema. Aqui se guardan:

- numero de objetos visibles
- numero de clases visibles
- exactitud del frame
- distribucion de decisiones firmes e inciertas
- uso de modulos internos
- tiempos de procesamiento por frame

En la implementacion actual, `n_objects` a nivel frame representa los GT visibles en ese frame, no solo los objetos que recibieron asignacion base.

### 2.4. Clase y escena

El nivel clase permite estudiar diferencias sistematicas entre categorias semanticas. El nivel escena permite analizar variabilidad entre secuencias y relacionar rendimiento con complejidad visual, densidad o composicion.

### 2.5. Batch global

El nivel batch agrega los resultados de todas las escenas completadas y proporciona un resumen compacto del experimento completo.

## 3. Familias de metricas

## 3.1. Metricas de decision colapsada

Estas metricas evalian si el sistema toma la decision correcta una vez convertida la salida final a una forma comparable, principalmente `existing` frente a `new`.

- `accuracy_global_collapsed`: exactitud global de la decision final colapsada
- `accuracy_existing_vs_new_collapsed`: exactitud al distinguir entre objeto existente y objeto nuevo
- `accuracy_parent_collapsed`: exactitud en la asignacion del parent correcto cuando el objeto ya existia
- `new_detection_accuracy_collapsed`: exactitud en la deteccion de objetos realmente nuevos
- `set_accuracy_ambiguous`: capacidad de los conjuntos ambiguos para contener la respuesta correcta

Estas metricas son especialmente utiles cuando la salida del sistema no siempre es estrictamente determinista, ya que permiten evaluar de forma justa decisiones ambiguas o provisionales.

## 3.2. Metricas de identidad y tracking

El segundo bloque mide continuidad temporal e identidad:

- `IDP`, `IDR`, `IDF1`
- `IDSW`
- `Frag`
- `tracking_recall`
- `mean_tracking_iou`
- `DetA`
- `AssA`
- `HOTA`

En este contexto:

- `IDF1` resume el equilibrio entre precision y recall de identidad dentro de la representacion colapsada usada por este evaluador
- `IDSW` cuantifica los cambios de identidad dentro de esa misma representacion interna
- `Frag` mide interrupciones o fragmentaciones del seguimiento en esa secuencia interna
- `tracking_recall` indica la proporcion de observaciones GT visibles correctamente seguidas
- `mean_tracking_iou` incorpora un componente espacial medio
- `DetA`, `AssA` y `HOTA` ofrecen una lectura mas cercana a literatura de tracking, separando deteccion y asociacion

Ademas, el evaluador generico guarda una segunda familia de metricas con sufijo `_iou40`. Estas metricas adicionales no reemplazan a la version oficial actual, sino que exigen `IoU >= 0.40` para contar un caso como correcto o como match valido dentro de esa variante mas estricta.

En esta implementacion, `IDP`, `IDR`, `IDF1`, `IDSW` y `Frag` deben interpretarse como metricas internas consistentes calculadas sobre la salida colapsada del pipeline, no como una exportacion literal de un toolkit oficial de benchmark MOT.

En esta implementacion, `HOTA` se calcula como una aproximacion derivada de `DetA` y `AssA`, por lo que debe interpretarse como una medida interna consistente del sistema y no como un reemplazo directo de implementaciones oficiales externas.

## 3.3. Metricas de incertidumbre

Dado que el pipeline puede producir salidas ambiguas o provisionales, se mide tambien la calidad de esa incertidumbre:

- cobertura de decisiones firmes
- exactitud de las decisiones firmes
- tasa de ambiguedad
- tasa de provisionalidad
- recuperacion de la respuesta correcta dentro de conjuntos ambiguos o provisionales

Estas metricas son utiles cuando interesa evaluar no solo si el sistema acierta, sino tambien si sabe cuando no puede decidir con suficiente confianza.

En particular, la recuperacion incierta incluye tres situaciones complementarias:

- acierto dentro de un conjunto ambiguo
- acierto dentro del conjunto de parents provisional
- deteccion correcta de novedad cuando la salida es `PROVISIONAL_NEW`

## 3.4. Metricas de estabilidad estructural

Se incluyen ademas metricas orientadas a estabilidad y salud de los tracks:

- inflacion del numero de tracks respecto al numero de objetos GT
- reaperturas de objetos existentes como si fueran nuevos
- objetos fragmentados
- uso de identidades ajenas
- eventos de tipo `swap` y `theft`

Estas medidas ayudan a interpretar fallos tipicos del sistema que no siempre quedan reflejados en una sola metrica agregada.

## 4. Metricas por objeto

Cada objeto GT dispone de un resumen temporal que incluye:

- `tracking_recall_object`
- `mean_tracking_iou_object`
- `idsw_object`
- `frag_object`
- `mt_pt_ml_label`

Las metricas `strict_accuracy` y `permissive_accuracy` de objeto se calculan sobre todos los frames visibles del GT. Por tanto, un frame visible sin asignacion firme tambien penaliza.

La clasificacion `MT/PT/ML` sigue el criterio habitual:

- `MT`: objeto mayoritariamente seguido
- `PT`: objeto parcialmente seguido
- `ML`: objeto mayoritariamente perdido

Esto permite distinguir si el rendimiento global se debe a unos pocos objetos muy bien seguidos o a un comportamiento estable sobre la mayoria de instancias.

## 5. Informacion de dificultad

Con el fin de relacionar rendimiento y dificultad, se guardan variables contextuales por caso:

- tamano relativo del objeto en la imagen
- numero de objetos visibles en el frame
- numero de objetos de la misma clase
- numero total de distractores
- numero de distractores de la misma clase
- antiguedad temporal del objeto

Estas variables no son metricas finales por si mismas, pero son esenciales para explicar errores y para realizar analisis posteriores por complejidad.

## 6. Telemetria de modulos internos

Una aportacion importante de esta evaluacion es que no solo guarda el resultado final, sino tambien el comportamiento de modulos intermedios de asociacion. En particular, se registra:

- si la distancia fue utilizada
- si la distancia resolvio efectivamente el caso
- si esa resolucion fue correcta
- si habia neighbor sets disponibles
- si el contexto modifico la mejor opcion
- si esa intervencion contextual fue correcta
- si se aplicaron rescates o vetos contextuales
- scores y margenes de los mejores candidatos

Esta informacion resulta especialmente valiosa para diagnosticar por que falla o mejora el sistema en escenas concretas.

## 7. Ficheros principales

Las salidas mas relevantes del sistema son:

- `per_case.csv`: base principal para analisis detallado y agregaciones offline
- `per_object.csv`: estabilidad temporal e identidad por instancia GT
- `per_class.csv`: comparativas por categoria semantica dentro de cada escena
- `per_scene.csv`: resumen por escena
- `summary_global.csv`: resumen agregado del batch completo
- `per_case_modules.csv`: diagnostico especifico de distancia, contexto y neighbor sets
- `per_event.csv`: eventos explicativos como swaps, robos de identidad y reaperturas
- `manifest.csv`: trazabilidad y estado de cada escena en el batch
- `run_config.csv`: configuracion efectiva del experimento

## 8. Justificacion del diseno

El diseno de las salidas sigue un criterio conservador: guardar de forma inline todo aquello que no puede reconstruirse de manera fiable tras la evaluacion, pero evitar almacenar metricas derivadas redundantes que se puedan recomputar offline.

Por ello, se guarda inline:

- la decision final por caso
- la informacion temporal por objeto
- la telemetria de modulos internos
- la trazabilidad por escena y por batch
- los eventos estructurales y los tiempos

Y se deja para analisis offline:

- comparativas macro frente a micro
- estudios por tipo manual de escena
- analisis de sesgo por clase
- tablas finales para memoria o paper

Este enfoque minimiza el riesgo de tener que repetir una evaluacion costosa por falta de informacion, manteniendo al mismo tiempo un volumen de salida razonable y util.

## 9. Recomendacion para presentar resultados

Para una presentacion clara en el TFM, las metricas mas representativas a nivel global suelen ser:

- `IDF1`
- `IDP`
- `IDR`
- `IDSW`
- `Frag`
- `tracking_recall`
- `mean_tracking_iou`
- `DetA`
- `AssA`
- `HOTA`
- `MT`, `PT`, `ML`
- `accuracy_global_collapsed`
- `new_detection_accuracy_collapsed`
- `reopen_rate_existing`
- `pred_track_inflation_factor`

Como apoyo interpretativo, conviene acompanarlas con:

- analisis por clase
- ejemplos de escenas representativas
- eventos explicativos
- variables de dificultad
- telemetria de distancia y contexto en escenas problematicas

## 10. Nota final

La combinacion de metricas de decision, identidad, estabilidad temporal, incertidumbre y telemetria interna permite una evaluacion mas completa que un simple score agregado. Esto es especialmente relevante en un sistema donde el reto no es solo detectar objetos, sino mantener identidades coherentes, manejar novedad y resolver ambiguedad en escenas complejas.
