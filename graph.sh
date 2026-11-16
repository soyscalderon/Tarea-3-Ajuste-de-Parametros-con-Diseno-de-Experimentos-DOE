#!/usr/bin/bash
# Usa el entorno virtual de la tarea si existe; si no, el Python del sistema.
PYTHON="./.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

for file in ./src/*.py; do
  clear
  "$PYTHON" "$file" --graph
done
clear
