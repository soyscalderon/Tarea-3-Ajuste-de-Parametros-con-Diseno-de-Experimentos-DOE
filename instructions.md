# Tarea 3 – Ajuste de Parámetros con Diseño de Experimentos (DOE)

A continuación se presenta el ejercicio a realizar. Implementa el algoritmo, genera las gráficas requeridas para respaldar tus experimentos y responde las preguntas analíticas en tu reporte final.

## Requisitos de Entrega (Repositorio y PDF)

1. **Repositorio (GitHub / GitLab):**
   - Incluye en tu reporte la URL pública de tu repositorio.
   - Organiza el código y recursos con la misma estructura utilizada en tareas anteriores.

2. **Reporte en PDF:**
   - Incluye todas las gráficas solicitadas (efectos principales, interacción, o curva ajustada según la opción elegida) y una tabla con la temperatura seleccionada como óptima para cada tamaño (n = 5, 10, 15), junto con su mediana bootstrap e intervalo de confianza.
   - El PDF debe dejar claro, para cada tamaño, cuál fue el mejor parámetro encontrado y cómo se llegó a esa conclusión (ANOVA, bootstrap).

## Ejercicio 1: Ajuste del Parámetro de Temperatura (Recocido Simulado) mediante DOE

- **Contexto:** Retomando el Recocido Simulado con 2-opt sobre TSP de la Tarea 2 (Ejercicio 4), ajusta el **parámetro de temperatura** (ej. temperatura inicial T₀) de tu algoritmo usando **uno** de los dos enfoques siguientes.

### Diseño experimental (aplica a ambas opciones)

- **Factor:** Temperatura del Recocido Simulado.
- **Bloque:** Tamaño del problema TSP, n ∈ {5, 10, 15} (una instancia por tamaño).
- **Réplicas:** 10 corridas por cada punto de diseño (temperatura × tamaño), con semillas distintas para capturar la aleatoriedad del algoritmo.

### Generación aleatoria de tus niveles de temperatura (cada estudiante usa valores distintos)

- Antes de empezar tu experimento, genera **un solo** número aleatorio r ~ Uniforme(0, 1) — por ejemplo, usando tu matrícula como semilla: `random.seed(matricula); r = random.random()`.
- Usa ese mismo valor de r durante **todo** tu experimento. No lo vuelvas a generar en cada corrida ni entre niveles — se fija una vez al inicio.
- Define tú mismo tu rango de temperatura [T_mín, T_máx] para el problema. Con tu valor de r, calcula tus niveles como una **fracción** de ese rango:
  - **Opción A (3 niveles):** bajo = 0.3·r, medio = 0.3·r + 0.3, alto = 0.3·r + 0.6
  - **Opción B (5 niveles):** axial bajo = 0.2·r, bajo = 0.2·r + 0.2, centro = 0.2·r + 0.4, alto = 0.2·r + 0.6, axial alto = 0.2·r + 0.8
- Convierte cada fracción a una temperatura real con: T_nivel = T_mín + fracción · (T_máx − T_mín).

## Opción A — Diseño Factorial Completo (3 niveles de temperatura)

- Define 3 niveles de temperatura (bajo, medio, alto) → diseño 3 × 3 = 9 puntos de diseño, 10 réplicas cada uno (90 corridas en total).
- Realiza un **ANOVA de dos factores** (temperatura, tamaño, e interacción) con nivel de significancia α = 0.05 (IC 95%).

### Cómo funciona el ANOVA de dos factores (no visto en clase)

- La idea es separar la variación total observada en tus 90 corridas en cuatro partes: la debida al factor **temperatura**, la debida al factor **tamaño**, la debida a la **interacción** (¿el efecto de la temperatura cambia según el tamaño?), y la debida al **error** (la variación entre tus 10 réplicas dentro de cada punto de diseño — es tu estimación del "ruido" propio del algoritmo).
- Para cada una de las tres primeras fuentes se calcula un estadístico **F = (variación explicada por esa fuente ÷ sus grados de libertad) / (variación del error ÷ sus grados de libertad)**. Un F grande (con p < 0.05) significa que esa fuente explica más variación de la que el puro azar explicaría.
- No necesitas calcularlo a mano. En Python, con `statsmodels`:

  ```python
  import statsmodels.api as sm
  from statsmodels.formula.api import ols

  modelo = ols('costo ~ C(temperatura) * C(tamano)', data=df).fit()
  tabla_anova = sm.stats.anova_lm(modelo, typ=2)
  print(tabla_anova)
  ```

- La columna `PR(>F)` de la tabla resultante te da el p-value de cada efecto (temperatura, tamaño, interacción). Si es menor a 0.05, ese efecto es estadísticamente significativo.
- Reporta la **mediana vía bootstrap** (con su intervalo de confianza del 95%) solo para los puntos de diseño cuyo efecto correspondiente resultó significativo.

### Visualización

- Gráfica de efectos principales (temperatura, tamaño).
- Gráfica de interacción temperatura × tamaño.

## Opción B — Metodología de Superficie de Respuesta (RSM)

Como en clase solo vimos el concepto de RSM, aquí va la adaptación a un solo factor continuo (versión simplificada de un **Diseño Central Compuesto**, ya que con un solo parámetro no aplica Box-Behnken):

- Para cada tamaño n, evalúa la temperatura en **5 niveles**: dos puntos "axiales" más allá del rango normal (muy bajo, muy alto), los puntos bajo/alto del rango factorial, y el centro.
- **Réplicas:** 5 por cada uno de los 4 puntos externos (axial bajo, bajo, alto, axial alto) y **10 en el punto central** — el centro se replica más porque es la única fuente de **error puro** para la prueba de falta de ajuste. Esto da 30 corridas por tamaño (90 en total), frente a las 90 de la Opción A.
- Ajusta un **modelo de regresión cuadrático**: Costo = β₀ + β₁·T + β₂·T² (uno por tamaño).

### Cómo ajustar el modelo cuadrático (no visto en clase)

- Junta, para cada tamaño, tus 50 observaciones (5 niveles de temperatura × 10 réplicas) en un vector de costos y su temperatura correspondiente — usa las observaciones individuales, no solo los promedios, para conservar la información de variabilidad.
- Ajusta por mínimos cuadrados con, por ejemplo:

  ```python
  import numpy as np
  beta2, beta1, beta0 = np.polyfit(temperatura, costo, 2)
  ```

  o, si también quieres los p-values de cada coeficiente, con `statsmodels`:

  ```python
  modelo = ols('costo ~ temperatura + I(temperatura**2)', data=df).fit()
  print(modelo.summary())
  ```

- El óptimo de la parábola ajustada es T* = −β₁ / (2β₂). Verifica que β₂ > 0 (parábola hacia arriba) para que T* sea un **mínimo** de costo, no un máximo.

### ANOVA de regresión (no visto en clase)

- Separa la variación total en la **explicada por el modelo** (regresión) y la **no explicada** (residual).
- Dentro de la variación residual, como tienes réplicas, puedes separar el **error puro** (la variación entre tus 10 réplicas dentro de cada nivel de temperatura — no depende del modelo) de la **falta de ajuste** (lo que el modelo cuadrático no logra capturar de las diferencias entre niveles).
- Si la falta de ajuste **no** es significativa (p ≥ 0.05), el modelo cuadrático describe bien los datos. Si **sí** es significativa, un término cuadrático no es suficiente para esa curva.
- En la práctica, basta con revisar en `modelo.summary()` el p-value del coeficiente β₂ (el término `I(temperatura**2)`): si es menor a 0.05, la curvatura es significativa y puedes confiar en el T* calculado.
- Si el ANOVA es significativo, reporta la **mediana vía bootstrap** (IC 95%) del costo en cada punto de diseño.

### Visualización

- Curva ajustada (Costo vs. Temperatura) con los puntos observados y su intervalo de confianza, una por cada tamaño.

## Entregable final (ambas opciones)

- **Tabla resumen:** para cada tamaño (n = 5, 10, 15), el valor de temperatura con mejor desempeño, su mediana bootstrap y su IC.
- **Análisis:** ¿la temperatura óptima cambia con el tamaño del problema? ¿hay una tendencia clara entre n y T*?
