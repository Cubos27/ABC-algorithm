import numpy as np
import abc_algorithm as abc

mejor_solucion, mejor_fitness, radios, historial_fitness = abc.algoritmo_abc(display_graphics=True, snapshots=True)

mejor_x = mejor_solucion[0]
mejor_y = mejor_solucion[1]

print("\n--- RESULTADO FINAL ---")
print(f"Mejor solución encontrada: x = {mejor_x:.4f}, y = {mejor_y:.4f}")
print(f"Valor mínimo de la función: {mejor_fitness:.4f}")