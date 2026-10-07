import numpy as np
import matplotlib.pyplot as plt
import random
import math_func
from snapshots import SnapshotRecorder


NUM_ABEJAS = 80
LIMITE_ABANDONO = 20 # Numero de intentos antes de abandonar una fuente de alimento
MAX_ITERACIONES = 200
LIMITE_INF = -512 # Limites del espacio de búsqueda
LIMITE_SUP = 512
DIMENSIONES = 2 # Buscamos x e y (2 dimensiones)
SOFT_STOP = -959 # Minimo global de la funcion delimitada.
FUNCION_OBJETIVO = math_func.funcion_objetivo1

def distancia_euclidiana(fuentes, i):
    # Calculo de distancia euclidiana de i a todas las abejas
    distancias = np.linalg.norm(fuentes - fuentes[i], axis=1)

    # Evitar que la propia abeja sea seleccionada
    distancias[i] = np.inf

    # Obtener los índices de las 3 abejas más cercanas
    indices_vecinas = np.argsort(distancias)[:3]

    return indices_vecinas

def algoritmo_abc(display_graphics=False, snapshots=False):
    # Inicialización aleatoria de las fuentes de alimento (soluciones)
    # Matriz de forma (NUM_ABEJAS, DIMENSIONES)
    fuentes = np.random.uniform(LIMITE_INF, LIMITE_SUP, (NUM_ABEJAS, DIMENSIONES))

    # Inicializacion de radios
    radios = []
    # historial de fitness
    historial_fitness = []

    # Inicializacion de graficos y primer iteración
    if display_graphics:
        puntos_rojos, fig = math_func.init_graphic(FUNCION_OBJETIVO)
        math_func.draw_points(fig, puntos_rojos, fuentes, fun=FUNCION_OBJETIVO)
        if snapshots:
            recorder = SnapshotRecorder(iteraciones_clave=[0, 100, 199])
            recorder(0, fuentes, function=FUNCION_OBJETIVO)

    
    # Evaluar la calidad (fitness) de cada fuente
    fitness = np.array([FUNCION_OBJETIVO(f[0], f[1]) for f in fuentes])
    
    # Contador de intentos fallidos por fuente (para el límite de abandono)
    intentos = np.zeros(NUM_ABEJAS)
    
    # Guardar la mejor solución encontrada hasta ahora
    mejor_indice = np.argmin(fitness)
    mejor_solucion = fuentes[mejor_indice].copy()
    mejor_fitness = fitness[mejor_indice]

    print(f"Iniciando ABC. Mejor fitness inicial: {mejor_fitness:.4f}")

    # Bucle principal
    for iteracion in range(MAX_ITERACIONES):
        if mejor_fitness <= SOFT_STOP:
            print(f"Condicion de paro suave alcanzada. \n{mejor_fitness:.4f} <= {SOFT_STOP}")
            break
        
        # --- FASE DE ABEJAS OBRERAS ---
        for i in range(NUM_ABEJAS):
            indices_vecinas = distancia_euclidiana(fuentes, i)

            # De esas 3 vecinas, seleccionar la que tenga
            # el mejor fitness (menor valor)
            k = random.choice(indices_vecinas)
            
            # Seleccionar una dimensión aleatoria (j) para modificar
            j = random.randint(0, DIMENSIONES - 1)
            
            # Generar nueva solución candidata (v)
            # v_ij = x_ij + phi * (x_ij - x_kj)
            phi = random.uniform(-1, 1)
            nueva_fuente = fuentes[i].copy()
            nueva_fuente[j] = fuentes[i][j] + phi * (fuentes[i][j] - fuentes[k][j])
            
            # Asegurar que la nueva solución esté dentro de los límites
            nueva_fuente = np.clip(nueva_fuente, LIMITE_INF, LIMITE_SUP)
            
            # Evaluar nueva solución
            nuevo_fitness = FUNCION_OBJETIVO(nueva_fuente[0], nueva_fuente[1])
            
            # Si la nueva solución es mejor (menor valor, ya que minimizamos), reemplazar
            if nuevo_fitness < fitness[i]:
                fuentes[i] = nueva_fuente
                fitness[i] = nuevo_fitness
                intentos[i] = 0  # Reiniciar contador de intentos
            else:
                intentos[i] += 1  # Aumentar contador de intentos fallidos

        # FASE DE ABEJAS OBSERVADORAS
        # Calcular probabilidades basadas en la calidad (inverso del fitness para minimización)
        # Para minimización, un fitness menor es mejor. Usamos 1/(1+fitness) para manejar negativos.
        calidad = 1.0 / (1.0 + fitness - np.min(fitness) + 1e-6)
        probabilidades = calidad / np.sum(calidad)
        
        for _ in range(NUM_ABEJAS):
            # Seleccionar una fuente basada en la probabilidad (ruleta)
            i = np.random.choice(range(NUM_ABEJAS), p=probabilidades)
            
            # La abeja espectadora realiza la misma búsqueda local que la obrera
            indices_vecinas = distancia_euclidiana(fuentes, i)
            k = random.choice(indices_vecinas)
            j = random.randint(0, DIMENSIONES - 1)
            
            phi = random.uniform(-1, 1)
            nueva_fuente = fuentes[i].copy()
            nueva_fuente[j] = fuentes[i][j] + phi * (fuentes[i][j] - fuentes[k][j])
            
            nueva_fuente = np.clip(nueva_fuente, LIMITE_INF, LIMITE_SUP)
            nuevo_fitness = FUNCION_OBJETIVO(nueva_fuente[0], nueva_fuente[1])
            
            if nuevo_fitness < fitness[i]:
                fuentes[i] = nueva_fuente
                fitness[i] = nuevo_fitness
                intentos[i] = 0
            else:
                intentos[i] += 1

        # --- FASE DE ABEJAS EXPLORADORAS ---
        # Si una fuente ha superado el límite de intentos, se abandona y se busca una nueva aleatoria
        for i in range(NUM_ABEJAS):
            if intentos[i] > LIMITE_ABANDONO:
                # Generar nueva solución aleatoria
                fuentes[i] = np.random.uniform(LIMITE_INF, LIMITE_SUP, DIMENSIONES)
                fitness[i] = FUNCION_OBJETIVO(fuentes[i][0], fuentes[i][1])
                intentos[i] = 0

        # --- ACTUALIZAR MEJOR SOLUCIÓN GLOBAL ---
        mejor_indice_actual = np.argmin(fitness)
        if fitness[mejor_indice_actual] < mejor_fitness:
            mejor_fitness = fitness[mejor_indice_actual]
            mejor_solucion = fuentes[mejor_indice_actual].copy()

        historial_fitness.append(mejor_fitness)
            
        # Imprimir progreso
        print(f"Iteración {iteracion}: Mejor f(x,y) = {mejor_fitness:.4f} en x={mejor_solucion[0]:.2f}, y={mejor_solucion[1]:.2f}")
        promedio_x = np.sum(fuentes[:, 0]) / iteracion
        promedio_y = np.sum(fuentes[:, 1]) / iteracion
        sum_radio = 0
        for i in range(NUM_ABEJAS):
            sum_radio += np.sqrt((fuentes[i][0] - promedio_x)**2 + (fuentes[i][1] - promedio_y)**2)
        radio = (1 / NUM_ABEJAS) * sum_radio
        print(f"Radio de la generación: {radio}")
        radios.append(radio)

        # Gráfica animada
        if display_graphics:
            math_func.draw_points(fig, puntos_rojos, fuentes, fun=FUNCION_OBJETIVO, pausa=0)
            if snapshots:
                recorder.capture(iteracion, fuentes, function=FUNCION_OBJETIVO)
    if display_graphics:
        plt.ioff()
        if snapshots:
            recorder.plot_individual(FUNCION_OBJETIVO, LIMITE_INF, LIMITE_SUP)
        plt.show()

    return mejor_solucion, mejor_fitness, radios, historial_fitness

# Ejecucion
if __name__ == "__main__":
    EJECUCIONES = 30
    historiales = []
    for i in range(EJECUCIONES):
        mejor_solucion, mejor_fitness, radios, historial_fitness = algoritmo_abc()
        historiales.append(historial_fitness)

    mejor_x = mejor_solucion[0]
    mejor_y = mejor_solucion[1]

    print("\n--- RESULTADO FINAL ---")
    print(f"Mejor solución encontrada: x = {mejor_x:.4f}, y = {mejor_y:.4f}")
    print(f"Valor mínimo de la función: {mejor_fitness:.4f}")

    # Calculo de la media de convergencia
    # Encontrar la ejecución que tomó más generaciones
    max_generations_reached = max(len(h) for h in historiales)

    # Igualar el tamaño de todas las listas repitiendo su último valor alcanzado
    padded_histories = []
    for h in historiales:
        padding = [h[-1]] * (max_generations_reached - len(h))
        padded_histories.append(h + padding)
        
    # Promediar verticalmente (generación por generación)
    mean_convergence = np.mean(padded_histories, axis=0)

    # 3. Funciones de Matplotlib para generar la grafica
    plt.figure(figsize=(8, 5)) # Crea la ventana/pantalla del gráfico
    plt.plot(mean_convergence, color='blue', linewidth=2) # Dibuja la línea con los datos
    plt.title('Promedio de convergencia')
    plt.xlabel('Ejecucion')
    plt.ylabel('Mejor fitness promedio')
    plt.grid(True, linestyle='--', alpha=0.6) # Activa la cuadrícula de fondo
    plt.show()