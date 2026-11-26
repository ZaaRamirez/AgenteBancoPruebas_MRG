# Banco de pruebas del agente fundacional

Esta carpeta contiene **solo la arquitectura del agente**. Los modelos se quedan donde esten instalados (Ollama, LeRobot, MolmoAct2), y `config/rutas.env` indica dónde encontrarlos.

El agente es un programa de Python independiente de la tarea y del modelo. ROS 2 se usa únicamente en la **conexión con el robot**, con mensajes estándar, de modo que el mismo agente funciona sin cambios en la simulación MuJoCo y en la plataforma física.

```
                         ┌──────────────────── AGENTE (Python, sin ROS) ────────────────────┐
  tarea ──► INTERACCION ─┼─► RAZONAMIENTO (LLM) ◄──── estado de la tarea (lo lleva el agente)  │
  archivo / terminal /   │        │   ▲                                                       │
  topico /agente/tarea   │        ▼   │ observaciones, verificacion                           │
                         │   EJECUCION (VLA)     PERCEPCION (VLM) = "ojos"                    │
                         │        │                    ▲                                      │
                         └────────┼────────────────────┼──────────────────────────────────────┘
                                  ▼                    │
                         ENTORNO (unica conexion con el mundo)
                           simulado: en memoria (pruebas)
                           ros2:  JointTrajectory / GripperCommand ─►  robot  ─► Image / JointState
                                  (los mismos mensajes para el gemelo MuJoCo y para el robot fisico)
```

| Pieza                                        | Dónde                                                                       | Descripción                                          |
| -------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------- |
| Ciclo del agente: tarea → LLM → VLA → VLM | `agente/agente.py`                                                         | es el mismo para toda tarea y todo modelo             |
| Interfaces de cada capa                      | `agente/interfaces.py`                                                     | se implementan para modelos o robots nuevos           |
| Modelos por capa                             | `config/modelos.yaml`                                                      | agregar un modelo es agregar una entrada              |
| Tareas                                       | `config/tareas/*.yaml` (perfiles) y `config/ensayos/*.yaml` (instancias) | una tarea nueva es un YAML nuevo                      |
| Robots                                       | `config/robots/*.yaml`                                                     | cámaras, articulaciones, unidades, tópicos          |
| Experimentos                                 | `config/experimentos/*.yaml`                                               | qué modelo usa cada capa, qué entorno y qué modo   |
| Indicadores                                  | `evaluacion/indicadores/*.py`                                              | funciones que se calculan sobre corridas ya guardadas |

```
BancoPruebasAgente/
├── entorno.sh                 # source en cada terminal: ROS 2 + entorno de Python + comando "agente"
├── config/
│   ├── rutas.env              # DÓNDE están Ollama, LeRobot y MolmoAct2 (agregar rutas de los modelos utilizados)
│   ├── modelos.yaml           # catálogo de modelos por capa
│   ├── experimentos/          # sim_completo, llm_real, vlm_imagen, omx_cableado, omx_mujoco, omx_mujoco_e2e, so101_fisico
│   ├── tareas/                # perfiles: empaquetado_medicamentos, manipulacion_general
│   ├── ensayos/               # tareas concretas: medicamentos_basico, medicamentos_seis_tipos, omx_hidroferol, general_ejemplo
│   └── robots/                # omx (gemelo MuJoCo), so101_bimanual (física; plantilla)
├── agente/                    # NÚCLEO
│   ├── agente.py              # ciclo ReAct (modular y extremo a extremo)
│   ├── interfaces.py          # ModeloLenguaje, ModeloVisionLenguaje, PoliticaVLA, Entorno, FuenteTareas
│   ├── capas/                 # razonamiento.py, percepcion.py, ejecucion.py
│   ├── modelos/               # llm.py (ollama, openai, guion), vlm.py (ollama, openai, transformers, simulado),
│   │                          # vla.py (lerobot, postura, simulado), catalogo.py
│   ├── tareas/                # perfiles.py, seguimiento.py (el estado lo lleva el agente)
│   ├── entornos/              # simulado.py, ros2.py (ÚNICO contacto con el robot), robot.py
│   ├── fuentes/               # archivo.py, terminal.py, ros2.py (tarea por tópico)
│   ├── registro.py            # guarda todo en resultados/
│   └── cli.py                 # comando "agente"
├── benchmark/                 # BENCHMARK por capa (LLM, VLM, VLA): casos, metricas, reporte  -> BENCHMARK.md
│   ├── llm/  vlm/  vla/       # generacion de casos, ejecucion reanudable y metricas de cada capa
│   └── reporte.py             # REPORTE.md ordenado con ranking, IC 95 % y grupos de empate
├── config/benchmark/          # llm.yaml, vlm.yaml, vla.yaml, objetos.yaml, comandas.yaml (de comanda.xlsx)
├── datos/vlm/                 # conjuntos de imagenes etiquetadas para el VLM
├── evaluacion/                # evaluar.py + indicadores/
├── herramientas/robot_ficticio.py   # nodo ROS que imita al robot (pruebas sin MuJoCo)
├── resultados/                # una carpeta por corrida
└── tests/
```

---

## Paso 1. Decirle dónde están los modelos (`config/rutas.env`)

Editar estas líneas:

```bash
OLLAMA_URL="http://localhost:11434"             # servidor de Ollama (no importa dónde guarde los modelos)
PY_ENV_ACTIVATE="source $HOME/Projects/MolmoAct/LeRobotMolmo/LeRobotMolmo/bin/activate"   # el entorno donde YA corre molmo_ros_bridge.py
LEROBOT_DIR="$HOME/Projects/MolmoAct/LeRobotMolmo"
MOLMO_VLM_PATH="$LEROBOT_DIR/models/MolmoAct2"
MOLMO_OMX_CKPT="$LEROBOT_DIR/outputs/<entrenamiento>/checkpoints/<paso>/pretrained_model"
```

El agente necesita  `numpy`, `pyyaml`, `requests` y `pillow`. Si falta alguno: `pip install pyyaml requests pillow`.

## Paso 3. Cargar el entorno y verificar

En cada terminal nueva:

```bash
source ~/Projects/BancoPruebasAgente/entorno.sh
agente verificar -x omx_mujoco
```

`verificar` muestra:

- si cada ruta de `rutas.env` existe;
- si Ollama responde y qué modelos tiene;
- qué librerías encuentra el Python activo: rclpy, control_msgs, torch, transformers, lerobot;
- qué modelo usará cada capa y si está disponible;
- qué tópicos espera del robot.

Si Ollama no responde, ejecuta `ollama serve`. Si falta algún modelo, `ollama pull <modelo>`.

## Paso 4. Probar la arquitectura por etapas

En cada etapa una capa simulada se sustituye por la real. Así, si algo falla, se puede indentificar en qué capa está. Cada corrida se guarda en `resultados/<experimento>_<fecha>/`.

**A. Todo simulado** (sin GPU, sin ROS, sin robot). Prueba el ciclo completo del agente con al tarea de medicamentos:

```bash
agente ejecutar -x sim_completo -e medicamentos_basico
agente ejecutar -x sim_completo -e general_ejemplo              # otra tarea, mismo agente
agente ejecutar -x sim_completo -e general_ejemplo -r 5 --vlm simulado_70 --vla simulado_70   # fallos y reintentos
agente ejecutar -x sim_completo -e medicamentos_basico --vlm simulado_faltante                 # tarea no realizable
```

**B. LLM real** (razonamiento con Ollama; percepción y ejecución simuladas):

```bash
agente ejecutar -x llm_real -e medicamentos_seis_tipos --llm qwen3.8
agente comparar -x llm_real -e medicamentos_seis_tipos -r 5 --llm qwen3.8,deepseek-r1,gemma4
```

**C. VLM real con una imagen fija.** Guarda como PNG un cuadro del conjunto `arrg-unam/omx_pick_hidroferol` y ejecuta:

```bash
agente ejecutar -x vlm_imagen -e omx_hidroferol --set entorno.imagen=$HOME/cuadro_hidroferol.png
agente ejecutar -x vlm_imagen -e omx_hidroferol --vlm molmoact2_vlm --set entorno.imagen=$HOME/cuadro.png  # backbone de MolmoAct2
```

**D. Conexión con el robot sin modelos.**

1. Lanza la simulación MuJoCo del OMX.
2. Comprueba los tópicos: `ros2 topic list | grep -E "front_camera|joint_states|main_controller|gripper"`.
3. Ejecuta:

   ```bash
   agente ejecutar -x omx_cableado -e omx_hidroferol
   ```

   El brazo debe ir a la postura de `postura_prueba` (en `config/modelos.yaml`), detenerse y reportar `subtarea_exitosa`.

Si el brazo no se mueve:

- cambia a `modo_comando: accion` en `config/robots/omx.yaml`;
- compara los nombres de las articulaciones con `ros2 topic echo /joint_states --once`.

Para probar sin MuJoCo: `python3 herramientas/robot_ficticio.py omx` en otra terminal.

**E. Arquitectura completa sobre el gemelo OMX:**

```bash
agente ejecutar -x omx_mujoco -e omx_hidroferol          # una subtarea, como en el conjunto de datos
agente ejecutar -x omx_mujoco -e medicamentos_basico -r 3
```

**F. Referencia de extremo a extremo** (MolmoAct2 solo, verificado por el mismo VLM):

```bash
agente ejecutar -x omx_mujoco_e2e -e omx_hidroferol
```

**G. Agente en espera de tareas.** Este es el modo de uso "como agente". Las tareas llegan por un tópico, o por la terminal con `--fuente terminal`:

```bash
agente servir -x omx_mujoco --fuente ros2
# en otra terminal:
ros2 topic echo /agente/estado
ros2 topic pub --once /agente/tarea std_msgs/msg/String \
  "{data: '{\"perfil\": \"empaquetado_medicamentos\", \"parametros\": {\"items\": {\"Hidroferol\": 1}, \"destination\": \"brown box\"}}'}"
ros2 topic pub --once /agente/tarea std_msgs/msg/String "{data: 'Put the banana in the bowl'}"
ros2 topic pub --once /agente/tarea std_msgs/msg/String "{data: '{\"cambiar\": {\"llm\": \"deepseek-r1\"}}'}"
```

En `--fuente terminal` se escriben las tareas a mano. Además:

- `:llm deepseek-r1` cambia el modelo de una capa en caliente;
- `@medicamentos_basico` corre un ensayo completo.

## Paso 5. Cambiar de modelo

| Cómo                 | Ejemplo                                                                                |
| --------------------- | -------------------------------------------------------------------------------------- |
| Al ejecutar           | `--llm deepseek-r1 --vlm qwen2.5vl --vla molmoact2_omx`                              |
| Sin catálogo         | `--llm ollama:llama3.2:3b`                                                           |
| Durante la ejecución | `:llm gemma4` (terminal) o `{"cambiar": {"vlm": "gemma4"}}` (tópico)              |
| Comparar varios       | `agente comparar -x llm_real -e medicamentos_seis_tipos -r 10 --llm a,b,c --vlm x,y` |
| Cualquier opción     | `--set razonamiento.salida_estructurada=schema --set ejecucion.max_duracion_s=120`   |

Para **agregar un modelo**:

- **Si usa un backend existente:** agrega una entrada en `config/modelos.yaml`.
- **Si es otro tipo de modelo:** escribe una clase que cumpla su interfaz en `agente/interfaces.py`:

  - LLM: `generar(prompt, esquema, contexto)`.
  - VLM: `consultar(prompt, imagenes, esquema, contexto)`.
  - VLA: `reiniciar()` y `predecir(obs, instruccion)`.

  Luego se regístra en el catálogo con `backend: mi_paquete.mi_modulo:MiClase`. El ciclo del agente no se toca.

## Paso 6. Evaluar

```bash
agente evaluar                                        # todas las corridas
agente evaluar resultados/llm_real__* --por-tarea -o resumen.csv
```

Cada corrida guarda todo lo necesario para evaluarla después:

- `corrida.yaml`: la configuración.
- `eventos.jsonl`: cada prompt, respuesta, latencia, validez, subtarea y estado.
- `tareas.jsonl`: el resultado de cada tarea.
- `imagenes/`: lo que vio el VLM, si `registro.guardar_imagenes: true`.

Los indicadores **no** se calculan durante la corrida. Son funciones en `evaluacion/indicadores/`, así que cuando se definan los indicadores definitivos se calculan sobre corridas ya hechas, sin repetir ensayos.

`generales.py` trae un conjunto inicial que no depende de la tarea:

| Capa         | Indicadores                                                                     |
| ------------ | ------------------------------------------------------------------------------- |
| Razonamiento | formato válido, decisión válida, errores de estado, latencia                 |
| Percepción  | formato válido, latencia por propósito, veredictos positivos                  |
| Ejecución   | fin detectado, duración, pasos, latencia de inferencia                         |
| Agente       | éxito, no realizable, tiempo, reintentos, subtareas, llamadas al LLM, progreso |

`_plantilla.py` muestra cómo escribir uno nuevo con referencia, por ejemplo la exactitud del inventario.

## Paso 7. Benchmark de modelos (LLM, VLM, VLA)

Para elegir el modelo más adecuado de cada capa hay un benchmark exhaustivo. Cada capa se mide aislada y contra respuestas conocidas, con los mismos casos para todos los modelos, intervalos de confianza y pruebas de significancia. El procedimiento completo, las métricas, los parámetros y cómo leer los resultados están en **[BENCHMARK.md](BENCHMARK.md)**. En resumen:

| Paso                        | Comando                                                                           | Qué produce                                                                                        |
| --------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 1. LLM, decisiones aisladas | `python3 -m benchmark llm-casos`                                                | exactitud de decisión en 720 casos con respuesta conocida, errores por tipo, estabilidad, latencia |
| 2. LLM, tareas completas    | `python3 -m benchmark llm-episodios`                                            | plan correcto, eficiencia y recuperación tras fallos con el agente real                            |
| 3. VLM, conjunto etiquetado | `python3 -m benchmark vlm-datos desde-lerobot ...` / `capturar` / `revisar` | imágenes con su respuesta correcta                                                                 |
| 4. VLM, medición           | `python3 -m benchmark vlm`                                                      | exactitud balanceada, falsos positivos, conteo, alucinaciones, robustez                             |
| 5. VLA, fuera de línea     | `python3 -m benchmark vla-offline`                                              | error de acción contra las demostraciones, pinza, latencia, sensibilidad al lenguaje               |
| 6. VLA, lazo cerrado        | `python3 -m benchmark vla-lazo`                                                 | tasa de éxito en MuJoCo o en el robot, con juez                                                    |
| 7. Reporte                  | `python3 -m benchmark reporte`                                                  | `REPORTE.md` ordenado (qué se mide, parámetros, resultados), CSV y gráficas                    |

Antes de cada corrida larga conviene la versión `--rapido`. Las corridas se reanudan con `--corrida <carpeta>`.

## Paso 8. Otra tarea u otro robot (sin tocar el código)

- **Tarea nueva:**  se copia `config/tareas/manipulacion_general.yaml`. Y se define:

  - el rol del LLM;
  - las habilidades y la instrucción que recibe el VLA;
  - la pregunta de verificación al VLM;
  - el seguimiento del estado: `none` o `counts`.

  Después se crea un ensayo en `config/ensayos/`.
- **Robot nuevo o SO-101 físico:** se ajusta `config/robots/so101_bimanual.yaml` a los tópicos de su driver y usa `-x so101_fisico`. El robot físico debe publicar y aceptar los mismos mensajes que el gemelo:

  - `Image` y `JointState` como entrada;
  - `JointTrajectory` y `GripperCommand` como salida.

  Con eso el agente es idéntico en simulación y en físico.

## Problemas comunes

| Síntoma                                                | Qué revisar                                                                                                                |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `Sin datos del robot tras 15 s`                       | ¿Corre la simulación? `ros2 topic hz /front_camera/color`; mismo `ROS_DOMAIN_ID`                                      |
| `error_llm: Connection refused`                       | `ollama serve`; `OLLAMA_URL` en `rutas.env`                                                                           |
| `ModuleNotFoundError: lerobot / torch`                | `PY_ENV_ACTIVATE` en `rutas.env` y volver a hacer `source entorno.sh`                                                 |
| `ModuleNotFoundError: rclpy` en el entorno de LeRobot | ese entorno debe usar Python 3.12 (el de Jazzy); haz `source /opt/ros/jazzy/setup.bash` antes de activarlo                |
| Formato inválido frecuente en el LLM                   | `--set razonamiento.salida_estructurada=schema`; temperatura más baja en `modelos.yaml`                                |
| La subtarea nunca termina                               | `ejecucion.umbral_estable`, `ciclos_estables` y `max_duracion_s` en el experimento                                    |
| `respuesta_vacia` frecuente en el LLM (p. ej. qwen3)  | el modelo agota `max_tokens` razonando: sube `max_tokens` o prueba `think: false` (ver `config/benchmark/llm.yaml`) |
| El benchmark se interrumpió                            | relanza el mismo comando con `--corrida resultados/benchmark/<tipo>/<fecha>`: continúa donde iba                         |
