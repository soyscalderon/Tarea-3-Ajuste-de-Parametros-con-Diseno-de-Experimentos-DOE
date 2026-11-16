# Tarea 3 — Ajuste de parámetros con Diseño de Experimentos (DOE)

Ajuste de la **temperatura inicial del Recocido Simulado con 2-opt** para el TSP
de la Tarea 2. Se usa un **diseño factorial completo 3 × 3** (Opción A):

- **Factor:** temperatura inicial `T0` (niveles bajo, medio y alto).
- **Bloque:** tamaño del TSP `n ∈ {5, 10, 15}` (una instancia por tamaño).
- **Réplicas:** 10 corridas por cada punto de diseño.
- **Total:** 3 × 3 × 10 = **90 corridas**.

El algoritmo no se reimplementa: `src/6.doe_temperatura.py` importa
`tour_length`, `linear_cooling_schedule` y `simulated_annealing` desde
`src/4.simulated_annealing.py`, el mismo código de la Tarea 2. Todas las
corridas usan el mismo presupuesto de 3,000 pasos de enfriamiento lineal para
que la temperatura inicial sea el único factor que cambia.

## Contenido

```
├── src/                 # scripts de la Tarea 2 + 6.doe_temperatura.py
├── graphs/              # figuras generadas con --graph
├── resultados/          # corridas, ANOVA, bootstrap y resumen en CSV
├── docs/                # reporte en Markdown y PDF
├── instructions.md      # enunciado de la tarea
├── requirements.txt     # dependencias de Python
├── exec.sh              # ejecuta todos los scripts
└── graph.sh             # ejecuta todos los scripts con --graph
```

| Script | Experimento | Figuras |
| ------ | ----------- | ------- |
| `src/1.estimate_pi.py` | Aceleración paralela de `estimate_pi` | `1.parallel_speedup.png` |
| `src/2.knapsack.py` | Mochila voraz: menor peso vs. valor/peso | `2.knapsack_greedy.png` |
| `src/3.TSP.py` | Búsqueda local 2-opt para el TSP | `3.tsp_2opt_initialization.png` |
| `src/4.simulated_annealing.py` | Recocido Simulado con enfriamiento lineal y exponencial | `4.sa_convergence.png`, `4.sa_time_vs_quality.png` |
| `src/5.MST.py` | Razón de aproximación del MST | `5.mst_approximation_ratio.png`, `5.mst_method_comparison.png` |
| `src/6.doe_temperatura.py` | **DOE 3 × 3 de la temperatura inicial** | `6.doe_efectos_principales.png`, `6.doe_interaccion.png`, `6.doe_distribuciones.png`, `6.doe_diagnostico_anova.png` |

## Niveles de temperatura de este estudiante

El valor `r ~ Uniforme(0,1)` se genera **una sola vez** al inicio con la
matrícula como semilla:

```python
SEMILLA_MATRICULA = 2_087_472
r = random.Random(SEMILLA_MATRICULA).random()   # r = 0.6643924310
```

Con `[T_min, T_max] = [0.1, 120.0]` y las fracciones de la Opción A:

| Nivel | Fracción | Temperatura real |
| ----- | -------- | ---------------- |
| bajo  | 0.199318 | 23.998196 |
| medio | 0.499318 | 59.968196 |
| alto  | 0.799318 | 95.938196 |

## Ejecución

```bash
python3 -m venv --system-site-packages .venv
./.venv/bin/pip install -r requirements.txt

./exec.sh      # corridas y salida por consola
./graph.sh     # además genera las figuras en graphs/
```

El script del DOE también se puede ejecutar solo:

```bash
./.venv/bin/python src/6.doe_temperatura.py --graph
```

Opciones adicionales: `--replicas N`, `--pasos N` y `--bootstrap N`.

## Resultados generados

- `resultados/doe_temperatura_corridas.csv`: las 90 corridas individuales.
- `resultados/doe_temperatura_anova.csv`: tabla ANOVA de dos factores.
- `resultados/doe_temperatura_posthoc.txt`: Kruskal-Wallis y Tukey HSD por tamaño.
- `resultados/doe_temperatura_bootstrap.csv`: mediana e IC 95 % por punto.
- `resultados/doe_temperatura_resumen.csv`: mejor temperatura por tamaño.
- `resultados/doe_temperatura_metadatos.json`: parámetros y semillas del experimento.

## Reporte

El reporte completo, con la tabla resumen, las gráficas y el análisis, está en
[`docs/reporte_tarea3.pdf`](docs/reporte_tarea3.pdf).

**Repositorio:** `PENDIENTE: agregar aquí la URL pública del repositorio.`

## Autor

Sebastián Calderón — Análisis y Optimización de Algoritmos, 5.º semestre.
