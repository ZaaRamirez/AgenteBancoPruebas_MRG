# Benchmark de modelos por capa del agente

Este documento explica **qué se mide** en cada capa, **bajo qué parámetros**, **cómo se ejecuta paso a paso** y **cómo se ordenan los resultados** para elegir el modelo más adecuado para el agente.

Solo se evalúa la **ejecución (inferencia)** de modelos ya entrenados. Las pruebas son exhaustivas, para que la comparación se haga sobre una población grande.

---

## 0. Principios del diseño

1. **Cada capa se evalúa aislada y contra una respuesta conocida.**
   - Al LLM le llega una percepción perfecta (una imagen clara de la escena) y una ejecución simulada: si la tarea falla, el error es del razonamiento.
   - El VLM responde sobre imágenes etiquetadas.
   - El VLA se compara contra demostraciones grabadas y, después, se comprueba su desempeño en el robot fisico o su respectiva simulación.
2. **Se evalúa exactamente lo que usa el agente.** Los prompts, preguntas, esquemas JSON y validaciones salen del mismo código y de los mismos perfiles de tarea que el agente (`agente/capas/*`, `config/tareas/*`). Lo que se mide es lo que el modelo verá en operación.
3. **Diseño de pares.** Todos los modelos reciben los mismos casos: misma semilla, mismas imágenes, mismos episodios, mismas escenas iniciales y los mismos fallos inyectados. Así las diferencias se deben al modelo, no a la muestra, y se comparan caso por caso.
4. **Condiciones iguales para todos.** `parametros_comunes` fija los mismos parámetros de inferencia para todos los modelos durante el benchmark: temperatura, `max_tokens`, contexto y semilla. No se modifica `config/modelos.yaml`. Antes de medir se hace una llamada de calentamiento, para que el tiempo de carga del modelo en la GPU no cuente como latencia.
5. **Estadística explícita.**
   - Las proporciones llevan intervalo de confianza (IC) al 95 % de Wilson; las medias, IC bootstrap.
   - Para comparar dos modelos se usa una **prueba de signos pareada (solo los casos en los que los dos modelos discrepan)**, que equivale a calcular la probabilidad exacta usando la distribución binomial. cuando el resultado es acierto/fallo.
   - Los modelos sin diferencia significativa comparten **grupo** (A, B, …).
6. **Se guarda todo.** Cada respuesta cruda, su calificación, latencia y tokens van a `resultados/benchmark/<tipo>/<fecha>/`. Las métricas se calculan después, así que se pueden recalcular o agregar métricas sin repetir las pruebas.
7. **Las corridas largas se pueden reanudar.** Si se interrumpen, se relanza el mismo comando con `--corrida <carpeta>` y se saltan las respuestas ya guardadas.
8. **Hay referencias que validan el banco.**
   - `guion` (LLM) responde siempre la respuesta correcta: si no obtiene 100 %, el banco está mal.
   - `postura_prueba` (VLA) es un piso trivial que todo VLA debe superar.

---

## 1. Preparación (una vez)

```bash
source ~/Projects/BancoPruebasAgente/entorno.sh
agente verificar                         # Ollama, rutas, librerías
```

1. **Mismos parámetros para todos.** Revisar `parametros_comunes` en `config/benchmark/{llm,vlm}.yaml`.
2. **Modelos a comparar.** Agrégarlos al catálogo `config/modelos.yaml` y ponerlos en la lista `modelos:` de cada archivo de `config/benchmark/`, o pásarlos con `--modelos a,b,c`. Por ejemplo, los VLM que tiene Ollama:
   ```yaml
   # en la sección vlm: de config/modelos.yaml
   qwen3-vl:    {backend: ollama, model: "qwen3-vl:32b", url: "${OLLAMA_URL}", temperature: 0.0}
   internvl3.5: {backend: ollama, model: "blaifa/InternVL3_5:8b", url: "${OLLAMA_URL}", temperature: 0.0}
   ```
   y en la sección `llm:` `gpt-oss: {backend: ollama, model: "gpt-oss:latest", url: "${OLLAMA_URL}"}`. También sirve `--modelos ollama:gpt-oss:latest` sin tocar el catálogo.
3. **Pedidos reales.** `config/benchmark/comandas.yaml` sale de `comanda.xlsx` (6 pedidos, 6 medicamentos). Si se cambia la hoja: `python3 -m benchmark comandas comanda.xlsx`.
4. **Prueba rápida.** Antes de cada ejecución prolongada, se tiene que correr la versión `--rapido` (pocos casos) para comprobar que todo funciona.

Todos los comandos también se pueden escribir como `agente benchmark <subcomando> ...`.

---

## 2. Capa de razonamiento (LLM)

### 2.1 Prueba A — casos de decisión (`llm-casos`)

**Qué es.** Un caso congela un momento de una tarea: qué se pidió, qué ya se hizo, qué falló y qué ve la percepción. Con eso se le pide al LLM **la siguiente acción**. La respuesta correcta se conoce de antemano. Hay 12 categorías:

| Categoría | Perfil | Situación | Respuesta correcta |
|---|---|---|---|
| `med_inicio` | medicamentos | nada hecho | cualquier medicamento del pedido |
| `med_intermedio` | medicamentos | avance parcial | cualquier medicamento **pendiente** |
| `med_ultimo` | medicamentos | falta una unidad | ese medicamento |
| `med_tras_fallo` | medicamentos | la última subtarea falló | un pendiente (incluido el que falló) |
| `med_distractores` | medicamentos | en el estante hay medicamentos **no pedidos** | un pendiente, nunca un distractor |
| `med_grande` | medicamentos | 5–6 medicamentos, muchas unidades | un pendiente |
| `med_real` | medicamentos | pedidos reales de `comanda.xlsx` | un pendiente |
| `gen_secuencial` | instrucción libre | 1–4 pasos en orden ("…, then …") | **solo** el paso que sigue |
| `gen_conjunto` | instrucción libre | varios objetos al mismo destino | cualquier objeto pendiente |
| `gen_completo` | instrucción libre | ya se hizo todo | `done` |
| `gen_tras_fallo` | instrucción libre | el último paso falló | reintentar ese paso |
| `gen_ausente` | instrucción libre | la percepción dice que el objeto no está | `abort` u `observe` |

**Métricas, por modelo y condición:**

| Métrica | Definición | Interpretación |
|---|---|---|
| **Exactitud de decisión** (principal) | casos con acción correcta / casos | qué tan seguido decide bien |
| Formato válido | respuestas con JSON y campo `action` | si se puede usar sin reintentar |
| Errores de estado | respuestas que dejarían mal la tarea: repetir lo hecho, elemento no pedido o inventado, fin prematuro, ignorar a la percepción, orden o destino equivocado | errores graves de razonamiento |
| Errores por tipo | `respuesta_vacia`, `formato`, `accion_inexistente`, `argumentos_incompletos`, `repite_lo_hecho`, `elemento_no_pedido`, `elemento_inventado`, `orden_incorrecto`, `destino_incorrecto`, `fin_prematuro`, `no_termina`, `aborto_injustificado`, `ignora_percepcion`, `observa_sin_necesidad` | dónde falla |
| Exactitud por categoría | exactitud en cada una de las 12 categorías | qué situaciones le cuestan |
| Estabilidad | con varias repeticiones: misma decisión en todas; correcto en todas; correcto en alguna | qué tan predecible es |
| Latencia media / p95 | segundos por consulta | costo en tiempo |
| Tokens y truncadas | tokens de la respuesta; respuestas cortadas por `max_tokens` | si `max_tokens` lo pone en desventaja |
| Memoria | RAM y VRAM del modelo (Ollama `/api/ps`) | costo en hardware |

**Parámetros (`config/benchmark/llm.yaml`):**

| Parámetro | Valor por omisión | Para qué |
|---|---|---|
| `casos.por_categoria` | 60, es decir 720 casos * Anexo justificación| tamaño de la población |
| `casos.semilla` | 2026 | mismos casos para todos los modelos |
| `parametros_comunes` | `max_tokens` 4096, `num_ctx` 8192, `seed` 2026 | igualdad de condiciones |
| condición `json_t0` | JSON forzado, temperatura 0 | como lo usará el agente |
| condición `libre_t0` | sin forzar formato, temperatura 0 | si sigue el formato del prompt por sí solo |
| condición `json_t07` | JSON, temperatura 0.7, 3 repeticiones | estabilidad bajo muestreo |

**Ejecución:**

```bash
python3 -m benchmark llm-casos --rapido                            # prueba (60 casos)
python3 -m benchmark llm-casos                                     # completa: 720 casos × 5 llamadas × modelo
python3 -m benchmark llm-casos --modelos guion,qwen3.8,deepseek-r1,gemma4,gpt-oss
python3 -m benchmark llm-casos --corrida resultados/benchmark/llm_casos/<fecha>   # reanudar
```

### 2.2 Prueba B — tareas completas (`llm-episodios`)

**Qué es.** El LLM resuelve tareas enteras **dentro del agente real**. Las tareas son:
- los 6 pedidos de `comanda.xlsx`;
- 24 pedidos generados;
- 24 instrucciones libres con su plan de referencia.

La percepción es un **oráculo de escena** (`vlm: escena`): sabe qué hay en el estante y en la caja y responde la verdad, incluidas las preguntas libres (`observe`) del LLM. La ejecución es simulada.

Se corre en dos condiciones:
- **`nominal`**: todo sale bien.
- **`fallos_30`**: el 30 % de las subtareas fallan, con los mismos fallos para todos los modelos.

**Métricas:**

| Métrica | Definición |
|---|---|
| **Plan correcto** (principal) | hizo exactamente lo pedido: todas las unidades o pasos, ninguno de más, en orden cuando la instrucción lo exige |
| Éxito | el agente terminó la tarea |
| Progreso | fracción del pedido completada |
| Llamadas por subtarea exitosa | eficiencia del razonamiento (ideal ≈ 1) |
| Decisiones inválidas por tarea | respuestas que el agente tuvo que rechazar |
| Recuperación tras fallo | éxito en las tareas donde se inyectó al menos un fallo |
| Tiempo de razonamiento por tarea | suma de latencias del LLM |
| Motivos de rechazo | conteo por tipo |

**Ejecución:**

```bash
python3 -m benchmark llm-episodios --rapido
python3 -m benchmark llm-episodios                  # 54 tareas × 2 condiciones × 3 repeticiones por modelo
```

---

## 3. Capa de percepción (VLM)

### 3.1 Construir el conjunto etiquetado (una vez)

El VLM se evalúa sobre **imágenes cuya respuesta se conoce**. Los conjuntos viven en `datos/vlm/<nombre>/`, con las imágenes y un `etiquetas.jsonl`. Hay tres formas de construirlos, y se pueden combinar:

**a) Desde los episodios grabados, sin etiquetar a mano** (verificación de subtareas). En cada episodio:
- el fotograma inicial es un caso "el objeto **no** está en el destino";
- el final, "**sí** está";
- además se agrega un negativo cruzado: el fotograma final, preguntando por **otro** medicamento que no se movió.

```bash
python3 -m benchmark vlm-datos desde-lerobot --fuente lerobot:arrg-unam/omx_pick_hidroferol \
    --salida omx_hidroferol --destino "brown box" --negativos-cruzados 2
# conjunto local:  --fuente lerobot:arrg-unam/omx_pick_hidroferol@/ruta/al/conjunto
```

**b) Capturando de la cámara del robot** (MuJoCo o físico) y tecleando la etiqueta. Es lo indicado para el **inventario**: se acomodan N cajas de cada medicamento, y se capturan e indican los conteos.

```bash
python3 -m benchmark vlm-datos capturar --robot omx --tipo inventario --salida inventario_estante
python3 -m benchmark vlm-datos capturar --robot omx --tipo verificacion --salida verificacion_mujoco
```

**c) Con fotos y un CSV** con las columnas `imagen, tipo, objeto, destino, esperado, conteos`:

```bash
python3 -m benchmark vlm-datos desde-carpeta --carpeta fotos/ --etiquetas fotos/etiquetas.csv --salida fotos_estacion
```

**Revisar siempre las etiquetas.** Este comando genera un mosaico de imágenes con su etiqueta y una tira con las perturbaciones:

```bash
python3 -m benchmark vlm-datos revisar omx_hidroferol
```

**Población recomendada:**
- Verificación: al menos 150 imágenes, con positivos y negativos.
- Inventario: al menos 60 escenas que cubran los 6 medicamentos con conteos de 0 a 4.

Cada imagen se consulta con cada redacción y cada perturbación. Por ejemplo, 150 imágenes × 3 redacciones × 9 perturbaciones = **4050 consultas por modelo**.

### 3.2 Medir (`vlm`)

**Qué es.** Cada modelo responde, sobre cada imagen etiquetada, **la misma pregunta que hace el agente** (perfil `empaquetado_medicamentos`), con el mismo esquema JSON. Además:
- responde dos **paráfrasis** de la pregunta;
- responde sobre 8 **perturbaciones** de la imagen: brillo bajo y alto, contraste bajo, desenfoque, ruido, JPEG de baja calidad, recorte y media resolución.

**Métricas de verificación** ("¿quedó el objeto en el destino?"):

| Métrica | Definición | Por qué importa |
|---|---|---|
| **Exactitud balanceada** (principal) | (sensibilidad + especificidad) / 2 | no se incrementa si hay más casos de una clase |
| Sensibilidad | "sí" cuando sí se cumplió | si es baja, hay reintentos innecesarios |
| Especificidad | "no" cuando falló | — |
| **Falsos positivos** | dice "sí" cuando falló | **el error más costoso**: el agente avanza sin reintentar |
| F1, MCC, exactitud | métricas estándar de clasificación | — |
| Sin respuesta | respuesta sin JSON válido; se cuenta como "no", igual que en el agente | — |
| Matriz de confusión | VP, FN, FP, VN | — |

**Métricas de inventario** ("¿cuántas cajas de cada medicamento hay?"):

| Métrica | Definición |
|---|---|
| **Conteo exacto por imagen** (principal) | todas las cantidades de la imagen correctas |
| Conteo exacto por elemento, MAE | aciertos y error medio en número de cajas |
| Presencia (precisión, sensibilidad, F1) | si detecta que el medicamento está |
| **Alucinaciones** | reporta cajas de un medicamento ausente |
| Omisiones | reporta 0 de uno presente |
| Nombres no pedidos | menciona medicamentos por los que no se preguntó |

**Robustez y consistencia:**
- exactitud por perturbación;
- **invariancia**: misma respuesta que con la imagen original;
- **consistencia entre paráfrasis**: misma respuesta con las tres redacciones;
- exactitud por redacción (q0 es la del agente).

**Costo:** latencia mediana y p95; memoria.

**Parámetros (`config/benchmark/vlm.yaml`):**
- `modelos`;
- `parametros_comunes` (temperatura 0, `max_tokens` 1024);
- `conjuntos`;
- `perfil_tarea`, de donde salen las preguntas del agente;
- `preguntas` (paráfrasis);
- `perturbaciones`;
- `salida_estructurada`;
- `repeticiones`.

**Ejecución:**

```bash
python3 -m benchmark vlm --rapido
python3 -m benchmark vlm --set conjuntos=[omx_hidroferol,inventario_estante] --modelos gemma4,qwen2.5vl,qwen3-vl,internvl3.5
python3 -m benchmark vlm --modelos molmoact2_vlm          # backbone de MolmoAct2 (transformers, en su entorno)
```

---

## 4. Capa de ejecución (VLA)

### 4.1 Prueba A — fuera de línea (`vla-offline`)

**Qué es.** El VLA recibe, fotograma por fotograma, las **observaciones grabadas** de demostraciones (imagen, estado e instrucción). Su acción se compara con la que ejecutó el operador. No necesita robot ni simulador, así que puede ser exhaustiva: todos los fotogramas de todos los episodios de prueba.

> Usar episodios **que no se usaron para entrenar**. El valor por omisión `episodios: {ultimos: 0.2}` toma el último 20 %; hay que ajustarlo a la partición real.

**Métricas** (en unidades del conjunto: grados en las articulaciones, 0–80 en la pinza del OMX):

| Métrica | Definición |
|---|---|
| **MAE articular** (principal, menor es mejor) | error absoluto medio por episodio |
| RMSE, error máximo, error final | penalizan errores grandes y el final de la tarea |
| Error por articulación | qué articulación aprende peor |
| Dentro de tolerancia | fotogramas con **todas** las articulaciones a menos de 5° |
| Acierto de pinza | mismo estado abierta/cerrada que el operador |
| Suavidad | jerk de la acción predicha entre el del operador (1 = igual de suave) |
| Latencia media, p95 y máxima; Hz máx.; memoria GPU pico | si alcanza la frecuencia de control del robot (10 Hz en el OMX) |
| Sensibilidad al lenguaje | cuánto cambian la acción y el error al cambiar **solo** la redacción de la instrucción |

**Ejecución:**

```bash
python3 -m benchmark vla-offline --rapido
python3 -m benchmark vla-offline --modelos molmoact2_omx,postura_prueba
python3 -m benchmark vla-offline --set offline.episodios=40-49      # episodios de prueba explícitos
```

Para comparar varios checkpoints, por ejemplo distintos pasos de entrenamiento, hay que agregarlos como entradas del catálogo (`molmoact2_omx_500`, `molmoact2_omx_1000`, …) y pasarlos en `--modelos`.

### 4.2 Prueba B — lazo cerrado (`vla-lazo`)

**Qué es.** El VLA **mueve el robot**, el gemelo MuJoCo o el físico, con el mismo entorno ROS 2 que usa el agente. Un **juez** decide si cumplió.

Las **escenas iniciales** de `lazo.escenas` se recorren en el mismo orden para cada modelo, así el ensayo *k* de todos los modelos parte del mismo arreglo de objetos (diseño pareado).

**Jueces:**

| Juez | Cómo decide | ¿Verdad de referencia? |
|---|---|---|
| `manual` | el operador califica: éxito / agarró sin colocar / no agarró / inválido | sí |
| `pose` | el simulador publica `geometry_msgs/PoseStamped` del objeto; éxito si queda dentro de `region_destino` | sí, y automático |
| `vlm` | un VLM juzga | no; usarlo solo para explorar |
| `detector` | solo comprueba que el robot se detuvo | no; sirve para probar el banco |

**Reinicio entre ensayos:**
- `manual`: el programa indica qué escena preparar y espera Enter;
- `servicio`: llama un servicio ROS 2 `std_srvs/Trigger` o `Empty` de la simulación;
- `ninguno`.

**Métricas:**

| Métrica | Definición |
|---|---|
| **Tasa de éxito** (principal) | IC de Wilson |
| Etapa alcanzada | agarró sin colocar / no agarró ni colocó |
| Éxito por escena y por redacción | dónde y con qué instrucción falla |
| Terminó por estabilidad | frente a tiempo agotado |
| Duración, pasos | — |
| Ciclos de pinza | reagarres |
| Recorrido articular, jerk | qué tan directo y suave es el movimiento |
| Latencia | por paso de control |

**Población recomendada:** 5 escenas × 4 ensayos × 2 redacciones = **40 ensayos por modelo**, como mínimo. Con 40 ensayos, el IC de una tasa del 70 % es de ±14 %. Para diferencias pequeñas, hay que subir `ensayos_por_escena`.

**Ejecución:**

```bash
# 1) lanza la simulación MuJoCo del OMX
python3 -m benchmark vla-lazo --rapido
python3 -m benchmark vla-lazo --modelos molmoact2_omx
python3 -m benchmark vla-lazo --set lazo.juez.modo=pose --set lazo.reinicio.modo=servicio   # si la sim lo publica
```

---

## 5. Resultados ordenados: el reporte

```bash
python3 -m benchmark reporte                     # última corrida de cada tipo
python3 -m benchmark reporte resultados/benchmark/llm_casos/<fecha> resultados/benchmark/vlm/<fecha> ...
```

Genera `resultados/benchmark/reporte_<fecha>/`, que contiene `REPORTE.md`, `resumen.csv`, un CSV por tabla y una gráfica por ranking (métrica principal con su IC).

`REPORTE.md` sigue siempre el mismo orden:
1. **Equipo** donde se midió: GPU, RAM y versiones.
2. **Resumen**: el mejor modelo **real** de cada capa y prueba, su valor con IC y qué modelos no difieren significativamente de él. Los modelos de referencia se excluyen de la recomendación.
3. Por cada prueba, tres apartados:
   - **Qué se mide**;
   - **Bajo qué parámetros**: modelos, parámetros comunes, condiciones, tamaño de la población, semillas, fuentes de datos;
   - **Resultados**: ranking con grupos de empate, desgloses por categoría, escena, perturbación o redacción, errores por tipo, matriz de confusión, robustez, costo.

**Cómo elegir el modelo de cada capa:**
1. Tomar los modelos del **grupo A** en la métrica principal de la prueba más cercana al uso real: `json_t0` y `fallos_30` para el LLM, verificación para el VLM, lazo cerrado para el VLA.
2. Si varios comparten el grupo A, desempatar por las métricas de riesgo:
   - LLM: errores de estado;
   - VLM: falsos positivos y alucinaciones;
   - VLA: acierto de pinza y tiempo agotado.
3. Si sigue el empate, desempatar por costo: latencia p95 y memoria. Considerar también si los modelos de las tres capas caben juntos en la GPU.
4. Si el grupo A sigue siendo amplio, **aumentar la población** (`casos.por_categoria`, imágenes, ensayos) y volver a correr. Las corridas se reanudan y se acumulan.

---

## 6. Agregar métricas o modelos

- **Modelo nuevo:** entrada en `config/modelos.yaml`, nombre en `modelos:` y volver a correr. Con `--corrida` solo se calcula lo que falta.
- **Métrica nueva:** las respuestas crudas están en `respuestas.jsonl`, `episodios.jsonl`, `fotogramas.jsonl` y `ensayos.jsonl`. Agrega la función en `benchmark/<capa>/metricas.py` y regenera el reporte; no hace falta repetir nada.
- **Categoría de caso nueva (LLM):** agrégarla en `benchmark/llm/casos.py` con su respuesta correcta y su calificación en `benchmark/llm/calificar.py`.

---

## 7. Anexos

- **720 para número de casos**: 12 categorías × 60 casos. El 60 se determina como número redondo para que cada categoría tuviera una precisión razonable sin que la corrida completa fuera interminable. Aun así se puede justificar con la fórmula del tamaño de muestra para una proporción, que es la que usa el benchmark en tamano_muestra() dentro de benchmark/comun/estadistica.py:

n = z² · p(1 − p) / E²

donde z = 1.96 (confianza del 95 %), p es la exactitud esperada y E es el margen de error que se acepta.

- **Por categoría (n = 60)**: Si un modelo acierta cerca del 80 %, el margen queda en unos ±10 puntos (con p = 0.8 y E = 0.10 la fórmula da 62). En el peor caso, p = 0.5, el margen es de unos ±12.6 puntos. Alcanza para ver en qué situaciones falla cada modelo, no para distinguir diferencias finas dentro de una sola categoría.
En total (n = 720): el margen es de ±2.2 puntos con p = 0.9 y de ±3.7 en el peor caso.

Este razonamiento solo cubre la precisión de cada modelo por separado. Para separar dos modelos importa otra cosa: en cuántos casos uno acierta y el otro falla, porque la prueba pareada usa solo esos casos. Si dos modelos difieren en pocos casos, 720 puede no bastar, y por eso el reporte marca a esos modelos con el mismo grupo.

-**semilla 2026**: Es arbitraria, representa el año actual. Cualquier entero fijo funciona igual. No afecta la dificultad ni el resultado. Solo garantiza dos cosas:

**Reproducibilidad**: con la misma semilla se generan exactamente los mismos casos hoy y dentro de un mes.
**Comparación justa**: todos los modelos reciben los mismos casos, que es lo que permite compararlos caso por caso.
