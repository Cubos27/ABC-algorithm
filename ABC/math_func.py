import numpy as np
import matplotlib.pyplot as plt

def funcion_objetivo1(x, y, lim_inferior=0, lim_superior=0):
    """Calcula el valor de f(x, y) según la fórmula proporcionada."""
    termino_1 = -(y + 47) * np.sin(np.sqrt(np.abs(y + (x / 2) + 47)))
    
    termino_2 = -x * np.sin(np.sqrt(np.abs(x - (y + 47))))
    
    return termino_1 + termino_2

def init_graphic(function):
    plt.ion()

    # Configuración de la figura y superficie
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x_grid = np.linspace(-512, 512, 100)
    y_grid = np.linspace(-512, 512, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = function(X, Y)

    # Dibujar superficie
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.5)
    ax.set_title('Evolución de fuentes de alimento en ABC')

    # Crear el objeto de los puntos rojos vacío
    puntos_rojos = ax.scatter([], [], [], color='red', s=60, edgecolors='black', depthshade=False)
    return puntos_rojos, fig

def draw_points(fig, puntos_rojos, posiciones, fun=None, pausa=0.05, z_arr=None):
    """
    posiciones: array (n_individuos, 2) con coordenadas REALES en el dominio.
    Si se pasa z_arr, se usa directamente; si no, se calcula con fun.
    """
    posiciones = np.asarray(posiciones)
    if posiciones.shape[1] != 2:
        raise ValueError(f"Se esperaban 2 columnas, se recibió {posiciones.shape}")
    x_arr, y_arr = posiciones[:, 0], posiciones[:, 1]

    # Si el array z es nulo, se debe calcular con la funcion proporcionada
    if z_arr is None:
        z_arr = fun(x_arr, y_arr)
    
    puntos_rojos._offsets3d = (x_arr, y_arr, z_arr)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(pausa)

if __name__ == "__main__":
    x = 512
    y = 404
    
    resultado = funcion_objetivo1(x, y, lim_inferior=-512, lim_superior=512)
    print(f"El resultado de f({x}, {y}) es: {resultado}")