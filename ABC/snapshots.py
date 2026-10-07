"""
snapshots_3d.py
Módulo para capturar y graficar snapshots de poblaciones sobre una
función objetivo 3D. Todo se accede a través de la clase SnapshotRecorder.
"""

import numpy as np
import matplotlib.pyplot as plt


class SnapshotRecorder:
    """
    Grabador de snapshots desacoplado del algoritmo.

    Uso típico
    ----------
        recorder = SnapshotRecorder(iteraciones_clave={0, 50, 100})

        for it in range(max_iter + 1):
            # ... tu algoritmo ...
            recorder(it, posiciones, function=funcion_objetivo)

        recorder.plot(funcion_objetivo, LB, UB, guardar_en='resultado.png')

    Parámetros
    ----------
    iteraciones_clave : iterable de hashables
        En qué iteraciones capturar. Puede ser ints, strings, lo que sea.
    """

    # -----------------------------------------------------
    # Constructor
    # -----------------------------------------------------
    def __init__(self, iteraciones_clave):
        self.iteraciones_clave = set(iteraciones_clave)
        self.storage = {}

    # -----------------------------------------------------
    # API de captura
    # -----------------------------------------------------
    def __call__(self, iteracion, posiciones, function=None, z=None):
        """Captura un snapshot si `iteracion` está en las claves."""
        if iteracion not in self.iteraciones_clave:
            return
        self._store(iteracion, posiciones, function=function, z=z)

    def capture(self, iteracion, posiciones, function=None, z=None):
        """Alias explícito de __call__."""
        self(iteracion, posiciones, function=function, z=z)

    def force_capture(self, iteracion, posiciones, function=None, z=None):
        """Fuerza la captura sin importar si está en las claves."""
        self._store(iteracion, posiciones, function=function, z=z)

    # -----------------------------------------------------
    # API de consulta
    # -----------------------------------------------------
    def __contains__(self, it):
        return it in self.storage

    def __len__(self):
        return len(self.storage)

    def __iter__(self):
        return iter(self.storage)

    def __getitem__(self, it):
        return self.storage[it]

    def keys(self):
        return self.storage.keys()

    def values(self):
        return self.storage.values()

    def items(self):
        return self.storage.items()

    def clear(self):
        self.storage.clear()

    def reset(self):
        """Vacía el storage — útil entre corridas con el mismo recorder."""
        self.storage.clear()

    # -----------------------------------------------------
    # Helpers internos
    # -----------------------------------------------------
    @staticmethod
    def _parse_bounds(lb, ub):
        lb_x, lb_y = (lb, lb) if np.isscalar(lb) else lb
        ub_x, ub_y = (ub, ub) if np.isscalar(ub) else ub
        return lb_x, ub_x, lb_y, ub_y

    @staticmethod
    def _build_surface(function, lb_x, ub_x, lb_y, ub_y, resolucion):
        x_grid = np.linspace(lb_x, ub_x, resolucion)
        y_grid = np.linspace(lb_y, ub_y, resolucion)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = function(X, Y)
        return X, Y, Z

    @staticmethod
    def _parse_plot_kwargs(kwargs):
        opts = {
            'ncols': kwargs.pop('ncols', None),
            'figsize_por_panel': kwargs.pop('figsize_por_panel', (7, 6)),
            'cmap': kwargs.pop('cmap', 'viridis'),
            'alpha_superficie': kwargs.pop('alpha_superficie', 0.5),
            'color_individuos': kwargs.pop('color_individuos', 'red'),
            'resaltar_mejor': kwargs.pop('resaltar_mejor', True),
            'resolucion': kwargs.pop('resolucion', 150),
            'mostrar': kwargs.pop('mostrar', True),
            'guardar_en': kwargs.pop('guardar_en', None),
            'dpi': kwargs.pop('dpi', 300),
        }
        if kwargs:
            raise TypeError(f"Argumentos no reconocidos: {list(kwargs.keys())}")
        return opts

    @staticmethod
    def _draw_on_ax(ax, X, Y, Z, snap, clave, opts):
        """Dibuja la superficie y los individuos de un snapshot sobre un ax."""
        pos, z_pos = snap['pos'], snap['z']

        ax.plot_surface(X, Y, Z, cmap=opts['cmap'],
                        edgecolor='none',
                        alpha=opts['alpha_superficie'])
        ax.scatter(pos[:, 0], pos[:, 1], z_pos,
                   color=opts['color_individuos'], s=50,
                   edgecolors='black', depthshade=False)

        idx = int(np.argmin(z_pos))
        ax.set_title(f'{clave}  |  mejor f = {z_pos[idx]:.4f}',
                     fontweight='bold')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$f$')

        if opts['resaltar_mejor']:
            ax.scatter(pos[idx, 0], pos[idx, 1], z_pos[idx],
                       color='gold', marker='*', s=300,
                       edgecolors='black', depthshade=False, zorder=10)

    # -----------------------------------------------------
    # API de gráfica
    # -----------------------------------------------------
    def plot(self, function, lb, ub, **kwargs):
        """Cuadrícula de subplots en una sola figura (comportamiento original)."""
        if not self.storage:
            raise ValueError("No hay snapshots capturados")

        opts = self._parse_plot_kwargs(kwargs)
        lb_x, ub_x, lb_y, ub_y = self._parse_bounds(lb, ub)
        X, Y, Z = self._build_surface(function, lb_x, ub_x, lb_y, ub_y,
                                      opts['resolucion'])

        claves = sorted(self.storage.keys())
        n = len(claves)
        ncols = opts['ncols'] if opts['ncols'] is not None else min(n, 3)
        nrows = int(np.ceil(n / ncols))

        ancho, alto = opts['figsize_por_panel']
        fig = plt.figure(figsize=(ancho * ncols, alto * nrows))

        for i, clave in enumerate(claves, start=1):
            ax = fig.add_subplot(nrows, ncols, i, projection='3d')
            self._draw_on_ax(ax, X, Y, Z, self.storage[clave], clave, opts)

        plt.tight_layout()

        if opts['guardar_en']:
            fig.savefig(opts['guardar_en'], dpi=opts['dpi'],
                        bbox_inches='tight')

        if opts['mostrar']:
            plt.show()

        return fig

    def plot_individual(self, function, lb, ub,
                        carpeta=None, prefijo='snapshot', **kwargs):
        """
        Crea una figura independiente por cada snapshot.

        Parámetros
        ----------
        function : callable
        lb, ub   : límites del dominio
        carpeta  : str o None
            Si se especifica, guarda cada figura como
            '{carpeta}/{prefijo}_{clave}.png'.
        prefijo  : str
            Prefijo del nombre de archivo.
        **kwargs : mismos que plot()
            (figsize_por_panel se usa como tamaño de cada figura)

        Devuelve
        --------
        dict : {clave: fig}  con todas las figuras creadas
        """
        if not self.storage:
            raise ValueError("No hay snapshots capturados")

        opts = self._parse_plot_kwargs(kwargs)
        lb_x, ub_x, lb_y, ub_y = self._parse_bounds(lb, ub)
        X, Y, Z = self._build_surface(function, lb_x, ub_x, lb_y, ub_y,
                                      opts['resolucion'])

        figuras = {}
        for clave in sorted(self.storage.keys()):
            fig = plt.figure(figsize=opts['figsize_por_panel'])
            ax = fig.add_subplot(111, projection='3d')
            self._draw_on_ax(ax, X, Y, Z, self.storage[clave], clave, opts)
            plt.tight_layout()

            if carpeta is not None:
                import os
                os.makedirs(carpeta, exist_ok=True)
                ruta = os.path.join(carpeta, f'{prefijo}_{clave}.png')
                fig.savefig(ruta, dpi=opts['dpi'], bbox_inches='tight')

            figuras[clave] = fig

        if opts['mostrar']:
            plt.show()

        return figuras

    # -----------------------------------------------------
    # Implementación interna
    # -----------------------------------------------------
    def _store(self, iteracion, posiciones, function=None, z=None):
        """Valida y guarda un snapshot en el storage."""
        posiciones = np.asarray(posiciones, dtype=float)
        if posiciones.ndim != 2 or posiciones.shape[1] < 2:
            raise ValueError(
                f"posiciones debe ser (n, ≥2). Recibido: {posiciones.shape}"
            )

        if z is None:
            if function is None:
                raise ValueError("Debes pasar 'function' o 'z'")
            z = np.asarray(
                function(posiciones[:, 0], posiciones[:, 1]), dtype=float
            )
        else:
            z = np.asarray(z, dtype=float)

        self.storage[iteracion] = {
            'pos': posiciones.copy(),
            'z': z.copy(),
        }

    def _plot_grid(self, function, lb_x, ub_x, lb_y, ub_y, **opts):
        """Dibuja la cuadrícula de subplots, uno por snapshot."""
        x_grid = np.linspace(lb_x, ub_x, opts['resolucion'])
        y_grid = np.linspace(lb_y, ub_y, opts['resolucion'])
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = function(X, Y)

        claves = sorted(self.storage.keys())
        n = len(claves)
        ncols = opts['ncols'] if opts['ncols'] is not None else min(n, 3)
        nrows = int(np.ceil(n / ncols))

        ancho, alto = opts['figsize_por_panel']
        fig = plt.figure(figsize=(ancho * ncols, alto * nrows))

        for i, clave in enumerate(claves, start=1):
            snap = self.storage[clave]
            pos, z_pos = snap['pos'], snap['z']

            ax = fig.add_subplot(nrows, ncols, i, projection='3d')
            ax.plot_surface(X, Y, Z, cmap=opts['cmap'],
                            edgecolor='none',
                            alpha=opts['alpha_superficie'])
            ax.scatter(pos[:, 0], pos[:, 1], z_pos,
                       color=opts['color_individuos'], s=50,
                       edgecolors='black', depthshade=False)

            idx = int(np.argmin(z_pos))
            titulo = f'{clave}  |  mejor f = {z_pos[idx]:.4f}'

            if opts['resaltar_mejor']:
                ax.scatter(pos[idx, 0], pos[idx, 1], z_pos[idx],
                           color='gold', marker='*', s=300,
                           edgecolors='black', depthshade=False, zorder=10)

            ax.set_title(titulo, fontweight='bold')
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
            ax.set_zlabel('$f$')

        plt.tight_layout()

        if opts['guardar_en']:
            fig.savefig(opts['guardar_en'], dpi=opts['dpi'],
                        bbox_inches='tight')

        if opts['mostrar']:
            plt.show()

        return fig

    # -----------------------------------------------------
    # Representación
    # -----------------------------------------------------
    def __repr__(self):
        return (f"SnapshotRecorder(iteraciones_clave="
                f"{sorted(self.iteraciones_clave)}, "
                f"capturadas={sorted(self.storage.keys())})")