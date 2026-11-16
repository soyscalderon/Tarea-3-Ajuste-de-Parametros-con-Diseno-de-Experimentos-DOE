"""Ejercicio 1 de la Tarea 3: ajuste de la temperatura inicial del Recocido
Simulado mediante un diseño factorial completo 3 x 3 (Opción A).

El algoritmo de Recocido Simulado con 2-opt no se vuelve a implementar: se
importa desde ``src/4.simulated_annealing.py`` de la Tarea 2, junto con la
función de costo y el esquema de enfriamiento lineal.  Este script solamente
agrega el diseño de experimentos, el ANOVA de dos factores, el bootstrap y las
gráficas solicitadas.

Diseño:
    Factor:     temperatura inicial T0 (3 niveles: bajo, medio, alto).
    Bloque:     tamaño del TSP n (5, 10 y 15 ciudades, una instancia por n).
    Réplicas:   10 corridas por cada punto de diseño (90 corridas en total).

Los niveles de temperatura se derivan de un único valor r ~ Uniforme(0, 1),
generado una sola vez con la matrícula como semilla.  Para que la comparación
sea justa, todas las corridas usan el mismo presupuesto de enfriamiento
(``PASOS_ENFRIAMIENTO``) y el mismo conjunto de semillas de corrida.
"""

import argparse
import importlib.util
import json
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin ventana, para guardar las figuras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ---------------------------------------------------------------------------
# Parámetros del experimento
# ---------------------------------------------------------------------------

# Matrícula usada como semilla para generar r.  Cambiarla por la matrícula
# propia y volver a ejecutar el script reproduce exactamente otro experimento.
SEMILLA_MATRICULA = 2_087_472

# Rango de temperatura definido para este problema (unidades arbitrarias).
TEMPERATURA_MINIMA = 0.1
TEMPERATURA_MAXIMA = 120.0

# Tamaños de TSP que se usan como bloque experimental.
TAMANOS = (5, 10, 15)

# Número de réplicas por punto de diseño.
REPLICAS = 10

# Presupuesto de enfriamiento: todos los niveles usan exactamente los mismos
# pasos para que la temperatura inicial sea el único factor que cambia.
PASOS_ENFRIAMIENTO = 3_000

# Semillas independientes para generar las instancias y para las corridas.
SEMILLA_INSTANCIA_BASE = 10_000
SEMILLA_CORRIDA_BASE = 20_000
SEMILLA_BOOTSTRAP_BASE = 30_000

# Número de remuestreos bootstrap y nivel de confianza.
NUMERO_BOOTSTRAP = 10_000
NIVEL_CONFIANZA = 0.95
ALFA = 0.05

# Nombres y orden de los tres niveles de temperatura de la Opción A.
NOMBRES_NIVELES = ("bajo", "medio", "alto")

DIRECTORIO_RAIZ = Path(__file__).resolve().parents[1]
DIRECTORIO_GRAFICAS = DIRECTORIO_RAIZ / "graphs"
DIRECTORIO_RESULTADOS = DIRECTORIO_RAIZ / "resultados"
RUTA_RECOCIDO = Path(__file__).resolve().parent / "4.simulated_annealing.py"


# ---------------------------------------------------------------------------
# Carga del código de la Tarea 2
# ---------------------------------------------------------------------------

def cargar_modulo(ruta_archivo, nombre_modulo):
    """Carga un archivo .py como módulo sin importar que su nombre tenga puntos."""
    especificacion = importlib.util.spec_from_file_location(nombre_modulo, ruta_archivo)
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"No se pudo cargar el módulo: {ruta_archivo}")
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


modulo_recocido = cargar_modulo(RUTA_RECOCIDO, "recocido_simulado_tarea2")

# Funciones reutilizadas de la Tarea 2.
calcular_longitud_tour = modulo_recocido.tour_length
programa_enfriamiento_lineal = modulo_recocido.linear_cooling_schedule
recocido_simulado = modulo_recocido.simulated_annealing


# ---------------------------------------------------------------------------
# Generación del valor r y de los niveles de temperatura
# ---------------------------------------------------------------------------

def generar_valor_r(semilla):
    """Genera un único r ~ Uniforme(0,1) con la semilla indicada.

    Se usa un generador local para no alterar el estado global de ``random``
    que después consumen las corridas del Recocido Simulado.
    """
    generador = random.Random(semilla)
    return generador.random()


def calcular_niveles_temperatura(valor_r, temperatura_minima, temperatura_maxima):
    """Convierte las fracciones de la Opción A a temperaturas reales.

    Opción A:
        bajo  = 0.3 * r
        medio = 0.3 * r + 0.3
        alto  = 0.3 * r + 0.6

    Luego: T_nivel = T_min + fraccion * (T_max - T_min).
    """
    fracciones = {
        "bajo": 0.3 * valor_r,
        "medio": 0.3 * valor_r + 0.3,
        "alto": 0.3 * valor_r + 0.6,
    }
    rango = temperatura_maxima - temperatura_minima
    return {
        nombre: temperatura_minima + fraccion * rango
        for nombre, fraccion in fracciones.items()
    }, fracciones


# ---------------------------------------------------------------------------
# Instancias y corridas
# ---------------------------------------------------------------------------

def generar_instancia(tamano, semilla):
    """Genera una instancia aleatoria del TSP en un cuadrado de 10 x 10."""
    generador = random.Random(semilla)
    return [(generador.uniform(0, 10), generador.uniform(0, 10))
            for _ in range(tamano)]


def ejecutar_experimento(niveles_temperatura, replicas, pasos_enfriamiento):
    """Ejecuta las 90 corridas del diseño factorial 3 x 3.

    Para cada tamaño se usa una sola instancia (el bloque).  Las semillas de
    corrida son las mismas en los tres niveles de temperatura; así las
    diferencias entre niveles no se confunden con la aleatoriedad de la
    solución inicial ni con la secuencia de números aleatorios del algoritmo.
    """
    registros = []
    semillas_corrida = [SEMILLA_CORRIDA_BASE + indice for indice in range(replicas)]

    for tamano in TAMANOS:
        semilla_instancia = SEMILLA_INSTANCIA_BASE + tamano
        coordenadas = generar_instancia(tamano, semilla_instancia)

        print(f"\nTamaño n = {tamano} (semilla de instancia = {semilla_instancia})")
        for nombre_nivel in NOMBRES_NIVELES:
            temperatura = niveles_temperatura[nombre_nivel]
            costos_nivel = []

            for numero_corrida, semilla in enumerate(semillas_corrida, start=1):
                # La misma semilla produce la misma solución inicial en los
                # tres niveles de temperatura (números aleatorios comunes).
                random.seed(semilla)
                tour_inicial = list(range(tamano))
                random.shuffle(tour_inicial)
                costo_inicial = calcular_longitud_tour(coordenadas, tour_inicial)

                programa = programa_enfriamiento_lineal(
                    initial_temperature=temperatura,
                    total_steps=pasos_enfriamiento,
                )

                inicio = time.perf_counter()
                tour_final = recocido_simulado(tour_inicial, coordenadas, programa)
                duracion = time.perf_counter() - inicio
                costo_final = calcular_longitud_tour(coordenadas, tour_final)
                costos_nivel.append(costo_final)

                registros.append({
                    "tamano": tamano,
                    "temperatura_nivel": nombre_nivel,
                    "temperatura": temperatura,
                    "corrida": numero_corrida,
                    "semilla": semilla,
                    "costo_inicial": costo_inicial,
                    "costo": costo_final,
                    "tiempo_segundos": duracion,
                })

            promedio = sum(costos_nivel) / len(costos_nivel)
            print(f"  {nombre_nivel:5s} (T0 = {temperatura:6.2f}): "
                  f"costo medio = {promedio:7.3f}")

    return pd.DataFrame(registros)


# ---------------------------------------------------------------------------
# ANOVA de dos factores
# ---------------------------------------------------------------------------

def realizar_anova(datos):
    """Ajusta el modelo de dos factores con interacción y devuelve la tabla."""
    modelo = ols("costo ~ C(temperatura_nivel) * C(tamano)", data=datos).fit()
    tabla = sm.stats.anova_lm(modelo, typ=2)

    # Traducción de los nombres de las fuentes de variación al español.
    traduccion = {
        "C(temperatura_nivel)": "Temperatura",
        "C(tamano)": "Tamaño",
        "C(temperatura_nivel):C(tamano)": "Temperatura × Tamaño",
        "Residual": "Error",
    }
    tabla = tabla.rename(index=traduccion)

    # Algunas versiones de statsmodels ya no incluyen la columna mean_sq;
    # se calcula a partir de la suma de cuadrados y los grados de libertad.
    if "mean_sq" not in tabla.columns:
        tabla["mean_sq"] = tabla["sum_sq"] / tabla["df"]

    # Tamaño del efecto: proporción de la variación total explicada.
    suma_total = tabla["sum_sq"].sum()
    tabla["eta_cuadrada"] = tabla["sum_sq"] / suma_total
    return modelo, tabla


def mostrar_anova(tabla):
    """Imprime la tabla ANOVA en español y resume la significancia."""
    tabla_impresa = tabla.rename(columns={
        "sum_sq": "Suma de cuadrados",
        "df": "gl",
        "mean_sq": "Cuadrado medio",
        "F": "F",
        "PR(>F)": "p-valor",
        "eta_cuadrada": "η²",
    })
    print("\n" + "=" * 78)
    print("ANOVA DE DOS FACTORES (temperatura, tamaño e interacción)")
    print("=" * 78)
    print(tabla_impresa.to_string(
        float_format=lambda valor: f"{valor:.6g}",
        na_rep="—",
    ))

    print("\nConclusión con α = 0.05:")
    for efecto in ("Temperatura", "Tamaño", "Temperatura × Tamaño"):
        p_valor = float(tabla.loc[efecto, "PR(>F)"])
        decision = "SIGNIFICATIVO" if p_valor < ALFA else "no significativo"
        print(f"  {efecto:22s}: p = {p_valor:.4g}  ->  {decision}")


def realizar_pruebas_posthoc(datos):
    """Pruebas de respaldo: Kruskal-Wallis y Tukey HSD por cada tamaño.

    El ANOVA supone varianzas homogéneas; en n = 5 varias corridas llegan
    exactamente al mismo óptimo (varianza cero).  Estas pruebas no paramétricas
    y de comparaciones múltiples ayudan a confirmar que las conclusiones no
    dependen solamente de ese supuesto.
    """
    lineas = []
    for tamano in TAMANOS:
        subconjunto = datos[datos["tamano"] == tamano]
        grupos = [
            subconjunto.loc[
                subconjunto["temperatura_nivel"] == nombre, "costo"
            ].to_numpy()
            for nombre in NOMBRES_NIVELES
        ]

        lineas.append(f"\nTamaño n = {tamano}")

        # Si todas las corridas de todos los niveles llegan exactamente al
        # mismo costo, las pruebas no paramétricas no están definidas.
        todos_iguales = all(np.ptp(grupo) == 0.0 for grupo in grupos)
        if todos_iguales:
            lineas.append(
                "Kruskal-Wallis: no aplica (las 30 corridas alcanzaron "
                "exactamente el mismo costo; no hay variación que comparar)."
            )
            lineas.append(
                "Tukey HSD: no aplica (varianza cero en todos los niveles)."
            )
            continue

        estadistico_kruskal, p_valor_kruskal = stats.kruskal(*grupos)
        lineas.append(
            f"Kruskal-Wallis: H = {estadistico_kruskal:.4f}, "
            f"p = {p_valor_kruskal:.6g} "
            f"({'significativo' if p_valor_kruskal < ALFA else 'no significativo'})"
        )
        comparacion_tukey = pairwise_tukeyhsd(
            endog=subconjunto["costo"],
            groups=subconjunto["temperatura_nivel"],
            alpha=ALFA,
        )
        lineas.append("Tukey HSD:")
        lineas.append(str(comparacion_tukey.summary()))

    texto = "\n".join(lineas)
    print("\n" + "=" * 78)
    print("PRUEBAS POST-HOC DE RESPALDO (por tamaño)")
    print("=" * 78)
    print(texto)
    return texto


# ---------------------------------------------------------------------------
# Bootstrap de la mediana por punto de diseño
# ---------------------------------------------------------------------------

def bootstrap_mediana(valores, numero_remuestreos, nivel_confianza, semilla):
    """Calcula la mediana y su intervalo de confianza por percentiles."""
    arreglo = np.asarray(valores, dtype=float)
    generador = np.random.default_rng(semilla)
    remuestras = generador.choice(
        arreglo,
        size=(numero_remuestreos, arreglo.size),
        replace=True,
    )
    medianas_remuestreadas = np.median(remuestras, axis=1)
    cola = (1.0 - nivel_confianza) / 2.0
    limite_inferior, limite_superior = np.percentile(
        medianas_remuestreadas, [100.0 * cola, 100.0 * (1.0 - cola)]
    )
    return float(np.median(arreglo)), float(limite_inferior), float(limite_superior)


def calcular_bootstrap_por_punto(datos, numero_remuestreos):
    """Aplica el bootstrap a cada combinación temperatura × tamaño."""
    filas = []
    indice_bootstrap = 0

    for tamano in TAMANOS:
        for nombre_nivel in NOMBRES_NIVELES:
            subconjunto = datos[
                (datos["tamano"] == tamano)
                & (datos["temperatura_nivel"] == nombre_nivel)
            ]
            valores = subconjunto["costo"].to_numpy()
            mediana, limite_inferior, limite_superior = bootstrap_mediana(
                valores,
                numero_remuestreos=numero_remuestreos,
                nivel_confianza=NIVEL_CONFIANZA,
                semilla=SEMILLA_BOOTSTRAP_BASE + indice_bootstrap,
            )
            indice_bootstrap += 1

            filas.append({
                "tamano": tamano,
                "temperatura_nivel": nombre_nivel,
                "temperatura": float(subconjunto["temperatura"].iloc[0]),
                "replicas": len(valores),
                "media": float(np.mean(valores)),
                "desviacion_estandar": float(np.std(valores, ddof=1)),
                "mediana": mediana,
                "ic_inferior": limite_inferior,
                "ic_superior": limite_superior,
            })

    return pd.DataFrame(filas)


def construir_resumen_por_tamano(tabla_bootstrap):
    """Selecciona la temperatura con menor mediana bootstrap para cada tamaño."""
    filas = []
    for tamano in TAMANOS:
        subconjunto = tabla_bootstrap[tabla_bootstrap["tamano"] == tamano].copy()
        # En caso de empate exacto se prefiere la temperatura más baja.
        subconjunto = subconjunto.sort_values(
            by=["mediana", "temperatura"], ascending=[True, True]
        )
        mejor = subconjunto.iloc[0]
        filas.append({
            "tamano": tamano,
            "mejor_nivel": mejor["temperatura_nivel"],
            "mejor_temperatura": mejor["temperatura"],
            "mediana_bootstrap": mejor["mediana"],
            "ic_inferior": mejor["ic_inferior"],
            "ic_superior": mejor["ic_superior"],
            "media": mejor["media"],
            "desviacion_estandar": mejor["desviacion_estandar"],
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def _intervalo_confianza_media(cuadrado_medio_error, grados_libertad_error, numero_observaciones):
    """Semi-ancho del IC 95 % para una media marginal, usando el error del ANOVA."""
    valor_critico = stats.t.ppf(1.0 - ALFA / 2.0, grados_libertad_error)
    error_estandar = math.sqrt(cuadrado_medio_error / numero_observaciones)
    return valor_critico * error_estandar


def graficar_efectos_principales(datos, tabla_anova, ruta_salida):
    """Gráfica de efectos principales para temperatura y tamaño."""
    cuadrado_medio_error = float(tabla_anova.loc["Error", "mean_sq"])
    grados_libertad_error = int(tabla_anova.loc["Error", "df"])

    promedios_temperatura = (
        datos.groupby("temperatura_nivel", sort=False)["costo"].mean()
        .reindex(NOMBRES_NIVELES)
    )
    promedios_tamano = datos.groupby("tamano", sort=True)["costo"].mean()

    # Número real de réplicas por celda (normalmente 10, pero se calcula para
    # que las gráficas sigan siendo correctas si se usa --replicas).
    replicas_por_celda = int(
        datos.groupby(["temperatura_nivel", "tamano"]).size().min()
    )

    # Medias marginales: cada nivel de temperatura promedia 3 tamaños × las
    # réplicas de cada celda; cada tamaño promedia 3 temperaturas × réplicas.
    observaciones_por_temperatura = len(TAMANOS) * replicas_por_celda
    observaciones_por_tamano = len(NOMBRES_NIVELES) * replicas_por_celda

    error_temperatura = _intervalo_confianza_media(
        cuadrado_medio_error, grados_libertad_error, observaciones_por_temperatura
    )
    error_tamano = _intervalo_confianza_media(
        cuadrado_medio_error, grados_libertad_error, observaciones_por_tamano
    )

    figura, (eje_temperatura, eje_tamano) = plt.subplots(1, 2, figsize=(12, 5))

    posiciones_temperatura = np.arange(len(NOMBRES_NIVELES))
    eje_temperatura.errorbar(
        posiciones_temperatura,
        promedios_temperatura.to_numpy(),
        yerr=error_temperatura,
        marker="o",
        markersize=8,
        linewidth=2,
        capsize=5,
        color="#1f77b4",
    )
    eje_temperatura.set_xticks(posiciones_temperatura)
    eje_temperatura.set_xticklabels([
        f"{nombre}\n{datos.loc[datos['temperatura_nivel'] == nombre, 'temperatura'].iloc[0]:.2f}"
        for nombre in NOMBRES_NIVELES
    ])
    eje_temperatura.set_xlabel("Nivel de temperatura inicial T₀")
    eje_temperatura.set_ylabel("Costo final medio del tour")
    eje_temperatura.set_title("Efecto principal de la temperatura")
    eje_temperatura.grid(alpha=0.3)

    posiciones_tamano = np.arange(len(TAMANOS))
    eje_tamano.errorbar(
        posiciones_tamano,
        promedios_tamano.to_numpy(),
        yerr=error_tamano,
        marker="s",
        markersize=8,
        linewidth=2,
        capsize=5,
        color="#d62728",
    )
    eje_tamano.set_xticks(posiciones_tamano)
    eje_tamano.set_xticklabels([str(tamano) for tamano in TAMANOS])
    eje_tamano.set_xlabel("Tamaño del problema (número de ciudades, n)")
    eje_tamano.set_ylabel("Costo final medio del tour")
    eje_tamano.set_title("Efecto principal del tamaño")
    eje_tamano.grid(alpha=0.3)

    figura.suptitle(
        "Efectos principales del diseño factorial 3 × 3 (barras: IC 95 %)",
        fontsize=13,
    )
    figura.tight_layout()
    figura.savefig(ruta_salida, dpi=150)
    plt.close(figura)


def graficar_interaccion(datos, tabla_anova, ruta_salida):
    """Gráfica de interacción temperatura × tamaño."""
    cuadrado_medio_error = float(tabla_anova.loc["Error", "mean_sq"])
    grados_libertad_error = int(tabla_anova.loc["Error", "df"])
    replicas_por_celda = int(
        datos.groupby(["temperatura_nivel", "tamano"]).size().min()
    )
    error_celda = _intervalo_confianza_media(
        cuadrado_medio_error, grados_libertad_error, replicas_por_celda
    )

    figura, eje = plt.subplots(figsize=(9, 6))
    colores = {5: "#2ca02c", 10: "#ff7f0e", 15: "#9467bd"}
    marcadores = {5: "o", 10: "s", 15: "^"}

    posiciones = np.arange(len(NOMBRES_NIVELES))
    for tamano in TAMANOS:
        subconjunto = datos[datos["tamano"] == tamano]
        promedios = (
            subconjunto.groupby("temperatura_nivel", sort=False)["costo"].mean()
            .reindex(NOMBRES_NIVELES)
        )
        eje.errorbar(
            posiciones,
            promedios.to_numpy(),
            yerr=error_celda,
            marker=marcadores[tamano],
            markersize=8,
            linewidth=2,
            capsize=4,
            color=colores[tamano],
            label=f"n = {tamano}",
        )

    eje.set_xticks(posiciones)
    eje.set_xticklabels([
        f"{nombre}\n{datos.loc[datos['temperatura_nivel'] == nombre, 'temperatura'].iloc[0]:.2f}"
        for nombre in NOMBRES_NIVELES
    ])
    eje.set_xlabel("Nivel de temperatura inicial T₀")
    eje.set_ylabel("Costo final medio del tour")
    eje.set_title("Interacción temperatura × tamaño (barras: IC 95 %)")
    eje.grid(alpha=0.3)
    eje.legend(title="Tamaño")
    figura.tight_layout()
    figura.savefig(ruta_salida, dpi=150)
    plt.close(figura)


def graficar_distribuciones(datos, ruta_salida):
    """Diagramas de caja por tamaño, con los puntos observados encima."""
    figura, ejes = plt.subplots(1, len(TAMANOS), figsize=(15, 5), sharey=False)
    colores_nivel = {"bajo": "#4c72b0", "medio": "#dd8452", "alto": "#55a868"}

    for eje, tamano in zip(ejes, TAMANOS):
        subconjunto = datos[datos["tamano"] == tamano]
        valores_por_nivel = [
            subconjunto.loc[
                subconjunto["temperatura_nivel"] == nombre, "costo"
            ].to_numpy()
            for nombre in NOMBRES_NIVELES
        ]
        cajas = eje.boxplot(
            valores_por_nivel,
            tick_labels=[
                f"{nombre}\n{subconjunto.loc[subconjunto['temperatura_nivel'] == nombre, 'temperatura'].iloc[0]:.1f}"
                for nombre in NOMBRES_NIVELES
            ],
            patch_artist=True,
            widths=0.55,
        )
        for caja, nombre in zip(cajas["boxes"], NOMBRES_NIVELES):
            caja.set_facecolor(colores_nivel[nombre])
            caja.set_alpha(0.55)

        # Puntos individuales con un pequeño desplazamiento horizontal.
        for indice, (nombre, valores) in enumerate(zip(NOMBRES_NIVELES, valores_por_nivel)):
            desplazamiento = np.random.default_rng(100 + indice).uniform(
                -0.10, 0.10, size=len(valores)
            )
            eje.scatter(
                np.full(len(valores), indice + 1) + desplazamiento,
                valores,
                s=28,
                color="black",
                alpha=0.65,
                zorder=3,
            )

        eje.set_title(f"n = {tamano}")
        eje.set_xlabel("Nivel de temperatura")
        eje.set_ylabel("Costo final del tour")
        eje.grid(alpha=0.3, axis="y")

    figura.suptitle("Distribución del costo final por punto de diseño", fontsize=13)
    figura.tight_layout()
    figura.savefig(ruta_salida, dpi=150)
    plt.close(figura)


def graficar_diagnostico(modelo_anova, ruta_salida):
    """Gráficas de diagnóstico del ANOVA: residuales vs ajuste y Q-Q normal."""
    valores_ajustados = modelo_anova.fittedvalues
    residuales = modelo_anova.resid
    residuales_estandarizados = residuales / np.std(residuales, ddof=1)

    figura, (eje_residuales, eje_qq) = plt.subplots(1, 2, figsize=(12, 5))

    eje_residuales.scatter(valores_ajustados, residuales_estandarizados, alpha=0.75)
    eje_residuales.axhline(0, color="black", linewidth=1, linestyle="--")
    eje_residuales.set_xlabel("Valor ajustado por el modelo")
    eje_residuales.set_ylabel("Residual estandarizado")
    eje_residuales.set_title("Residuales vs valores ajustados")
    eje_residuales.grid(alpha=0.3)

    stats.probplot(residuales_estandarizados, dist="norm", plot=eje_qq)
    eje_qq.set_xlabel("Cuantiles teóricos de la normal")
    eje_qq.set_ylabel("Residuales ordenados")
    eje_qq.set_title("Gráfico Q-Q normal de los residuales")
    eje_qq.grid(alpha=0.3)

    figura.suptitle("Diagnóstico del ANOVA de dos factores", fontsize=13)
    figura.tight_layout()
    figura.savefig(ruta_salida, dpi=150)
    plt.close(figura)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():
    analizador = argparse.ArgumentParser(
        description="Ajuste de la temperatura inicial del Recocido Simulado "
                    "mediante un diseño factorial 3 x 3."
    )
    analizador.add_argument(
        "--graph",
        action="store_true",
        help="Genera las figuras en graphs/.",
    )
    analizador.add_argument(
        "--replicas",
        type=int,
        default=REPLICAS,
        help=f"Número de réplicas por punto de diseño (por defecto {REPLICAS}).",
    )
    analizador.add_argument(
        "--pasos",
        type=int,
        default=PASOS_ENFRIAMIENTO,
        help=f"Pasos de enfriamiento por corrida (por defecto {PASOS_ENFRIAMIENTO}).",
    )
    analizador.add_argument(
        "--bootstrap",
        type=int,
        default=NUMERO_BOOTSTRAP,
        help=f"Remuestreos bootstrap (por defecto {NUMERO_BOOTSTRAP}).",
    )
    argumentos = analizador.parse_args()

    DIRECTORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
    if argumentos.graph:
        DIRECTORIO_GRAFICAS.mkdir(parents=True, exist_ok=True)

    # 1) Un solo valor r para todo el experimento.
    valor_r = generar_valor_r(SEMILLA_MATRICULA)
    niveles_temperatura, fracciones = calcular_niveles_temperatura(
        valor_r, TEMPERATURA_MINIMA, TEMPERATURA_MAXIMA
    )

    print("=" * 78)
    print("TAREA 3 - AJUSTE DE TEMPERATURA DEL RECOCIDO SIMULADO MEDIANTE DOE")
    print("Opción A: diseño factorial completo 3 x 3")
    print("=" * 78)
    print(f"Semilla de la matrícula: {SEMILLA_MATRICULA}")
    print(f"r ~ Uniforme(0,1) generado una sola vez: r = {valor_r:.10f}")
    print(f"Rango de temperatura: [{TEMPERATURA_MINIMA}, {TEMPERATURA_MAXIMA}]")
    print("\nNiveles de temperatura (fracción -> T real):")
    for nombre in NOMBRES_NIVELES:
        print(f"  {nombre:5s}: {fracciones[nombre]:.6f} -> "
              f"T0 = {niveles_temperatura[nombre]:.6f}")
    print(f"\nTamaños: {TAMANOS}  |  réplicas por punto: {argumentos.replicas}")
    print(f"Pasos de enfriamiento por corrida: {argumentos.pasos}")

    # 2) Corridas del diseño experimental.
    datos = ejecutar_experimento(
        niveles_temperatura,
        replicas=argumentos.replicas,
        pasos_enfriamiento=argumentos.pasos,
    )

    ruta_corridas = DIRECTORIO_RESULTADOS / "doe_temperatura_corridas.csv"
    datos.to_csv(ruta_corridas, index=False, float_format="%.8f")
    print(f"\nCorridas guardadas en: {ruta_corridas}")

    # 3) ANOVA de dos factores.
    modelo_anova, tabla_anova = realizar_anova(datos)
    mostrar_anova(tabla_anova)

    ruta_anova = DIRECTORIO_RESULTADOS / "doe_temperatura_anova.csv"
    tabla_anova.rename(columns={
        "sum_sq": "suma_cuadrados",
        "df": "grados_libertad",
        "mean_sq": "cuadrado_medio",
        "F": "estadistico_F",
        "PR(>F)": "p_valor",
        "eta_cuadrada": "eta_cuadrada",
    }).to_csv(ruta_anova, float_format="%.8f")
    print(f"Tabla ANOVA guardada en: {ruta_anova}")

    # 3b) Pruebas de respaldo por tamaño (Kruskal-Wallis y Tukey HSD).
    texto_posthoc = realizar_pruebas_posthoc(datos)
    ruta_posthoc = DIRECTORIO_RESULTADOS / "doe_temperatura_posthoc.txt"
    ruta_posthoc.write_text(texto_posthoc, encoding="utf-8")
    print(f"\nPruebas post-hoc guardadas en: {ruta_posthoc}")

    # 4) Bootstrap de la mediana en cada punto de diseño.
    tabla_bootstrap = calcular_bootstrap_por_punto(
        datos, numero_remuestreos=argumentos.bootstrap
    )
    ruta_bootstrap = DIRECTORIO_RESULTADOS / "doe_temperatura_bootstrap.csv"
    tabla_bootstrap.to_csv(ruta_bootstrap, index=False, float_format="%.8f")

    print("\n" + "=" * 78)
    print("MEDIANA BOOTSTRAP E IC 95 % POR PUNTO DE DISEÑO")
    print("=" * 78)
    tabla_bootstrap_impresa = tabla_bootstrap.copy()
    tabla_bootstrap_impresa["mediana (IC 95 %)"] = tabla_bootstrap_impresa.apply(
        lambda fila: f"{fila['mediana']:.3f} "
                     f"[{fila['ic_inferior']:.3f}, {fila['ic_superior']:.3f}]",
        axis=1,
    )
    print(tabla_bootstrap_impresa[
        ["tamano", "temperatura_nivel", "temperatura", "mediana (IC 95 %)"]
    ].to_string(index=False, float_format=lambda valor: f"{valor:.3f}"))
    print(f"\nTabla bootstrap guardada en: {ruta_bootstrap}")

    # 5) Tabla resumen: mejor temperatura por tamaño.
    resumen = construir_resumen_por_tamano(tabla_bootstrap)
    ruta_resumen = DIRECTORIO_RESULTADOS / "doe_temperatura_resumen.csv"
    resumen.to_csv(ruta_resumen, index=False, float_format="%.8f")

    print("\n" + "=" * 78)
    print("RESUMEN: TEMPERATURA CON MEJOR DESEMPEÑO POR TAMAÑO")
    print("=" * 78)
    resumen_impreso = resumen.copy()
    resumen_impreso["mediana (IC 95 %)"] = resumen_impreso.apply(
        lambda fila: f"{fila['mediana_bootstrap']:.3f} "
                     f"[{fila['ic_inferior']:.3f}, {fila['ic_superior']:.3f}]",
        axis=1,
    )
    print(resumen_impreso[[
        "tamano", "mejor_nivel", "mejor_temperatura", "mediana (IC 95 %)"
    ]].to_string(index=False, float_format=lambda valor: f"{valor:.3f}"))
    print(f"\nResumen guardado en: {ruta_resumen}")

    # 6) Metadatos del experimento para reproducibilidad.
    metadatos = {
        "opcion": "A - factorial completo 3x3",
        "semilla_matricula": SEMILLA_MATRICULA,
        "valor_r": valor_r,
        "fracciones": fracciones,
        "temperatura_minima": TEMPERATURA_MINIMA,
        "temperatura_maxima": TEMPERATURA_MAXIMA,
        "niveles_temperatura": niveles_temperatura,
        "tamanos": list(TAMANOS),
        "replicas": argumentos.replicas,
        "pasos_enfriamiento": argumentos.pasos,
        "remuestreos_bootstrap": argumentos.bootstrap,
        "nivel_confianza": NIVEL_CONFIANZA,
        "alfa": ALFA,
        "semilla_instancia_base": SEMILLA_INSTANCIA_BASE,
        "semilla_corrida_base": SEMILLA_CORRIDA_BASE,
        "semilla_bootstrap_base": SEMILLA_BOOTSTRAP_BASE,
    }
    ruta_metadatos = DIRECTORIO_RESULTADOS / "doe_temperatura_metadatos.json"
    with open(ruta_metadatos, "w", encoding="utf-8") as archivo:
        json.dump(metadatos, archivo, ensure_ascii=False, indent=2)
    print(f"Metadatos guardados en: {ruta_metadatos}")

    # 7) Figuras solicitadas.
    if argumentos.graph:
        ruta_efectos = DIRECTORIO_GRAFICAS / "6.doe_efectos_principales.png"
        ruta_interaccion = DIRECTORIO_GRAFICAS / "6.doe_interaccion.png"
        ruta_distribuciones = DIRECTORIO_GRAFICAS / "6.doe_distribuciones.png"
        ruta_diagnostico = DIRECTORIO_GRAFICAS / "6.doe_diagnostico_anova.png"

        graficar_efectos_principales(datos, tabla_anova, ruta_efectos)
        graficar_interaccion(datos, tabla_anova, ruta_interaccion)
        graficar_distribuciones(datos, ruta_distribuciones)
        graficar_diagnostico(modelo_anova, ruta_diagnostico)

        print("\nFiguras generadas:")
        for ruta in (ruta_efectos, ruta_interaccion, ruta_distribuciones, ruta_diagnostico):
            print(f"  {ruta}")


if __name__ == "__main__":
    main()
