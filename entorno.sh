# Carga el entorno del banco de pruebas. En CADA terminal nueva:
#     source ~/Projects/BancoPruebasAgente/entorno.sh
# Hace tres cosas: ROS 2 (si no esta cargado) -> entorno de Python de los modelos
# (PY_ENV_ACTIVATE de config/rutas.env, si se definio) -> comando "agente".
# No instala nada ni copia modelos.

BANCO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; source "$BANCO_DIR/config/rutas.env"; set +a

if [ -z "$ROS_DISTRO" ] && [ -f "$ROS_SETUP" ]; then
  source "$ROS_SETUP"
fi
if [ -n "$PY_ENV_ACTIVATE" ]; then
  eval "$PY_ENV_ACTIVATE"
fi

export BANCO_DIR
export PYTHONPATH="$BANCO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY="localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"
agente() { python3 -m agente "$@"; }
export -f agente 2>/dev/null

echo "Banco de pruebas: $BANCO_DIR"
echo "  ROS 2: ${ROS_DISTRO:-NO CARGADO}   Python: $(command -v python3)"
echo "  Usa:  agente verificar   |   agente ejecutar -x sim_completo -e medicamentos_basico"
