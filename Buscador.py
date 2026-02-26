"""
Buscador.py — Algoritmos de búsqueda sobre el laberinto
========================================================
Clase Buscador:
  - Recibe un Laberinto y un Grafo ya construidos
  - Implementa BFS, DFS y A*
  - Muestra una animación paso a paso de cómo explora el laberinto
  - Al terminar resalta el camino encontrado

Colores de la animación:
  - Azul claro  → celda explorada (visitada)
  - Amarillo    → celda en la frontera (por explorar)
  - Verde       → camino final encontrado
  - Azul        → inicio (S)
  - Rosa        → meta (G)

Uso:
    from Laberinto import Laberinto
    from Grafo import Grafo
    from Buscador import Buscador

    laberinto = Laberinto(tamano=21)
    grafo     = Grafo(laberinto)
    buscador  = Buscador(laberinto, grafo)

    buscador.bfs()    # Breadth-First Search
    buscador.dfs()    # Depth-First Search
    buscador.a_star() # A*
"""

import heapq
import time
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation

from Laberinto import Laberinto
from Grafo import Grafo
from typing import Optional

class Buscador:

    # Valores extra para la animación (además de los de Laberinto)
    EXPLORADO = 4   # celda ya visitada por el algoritmo
    FRONTERA  = 5   # celda en cola/pila, pendiente de explorar
    CAMINO    = 6   # camino final encontrado

    def __init__(self, laberinto: Laberinto, grafo: Grafo, heuristica=None):
        """
        Args:
            laberinto  : Objeto Laberinto ya generado.
            grafo      : Objeto Grafo construido a partir del laberinto.
            heuristica : Función heurística opcional para A*.
                         Debe aceptar (nodo, meta) y devolver un número.
                         Si no se pasa, usa distancia Manhattan por defecto.
        """
        self.laberinto  = laberinto
        self.grafo      = grafo
        # Si no se pasa heurística, usa Manhattan por defecto
        # Línea __init__
        # __init__ — línea 63
        self.heuristica = heuristica if heuristica is not None else lambda nodo, meta: abs(nodo[0] - meta[0]) + abs(
            nodo[1] - meta[1])

    # ------------------------------------------------------------------ #
    #  Utilidades compartidas                                             #
    # ------------------------------------------------------------------ #

    def _reconstruir_camino(self, origen: dict, nodo_final: tuple) -> list:
        """
        Reconstruye el camino desde la meta hasta el inicio siguiendo
        el diccionario de padres (origen).

        Args:
            origen     : Diccionario {nodo: nodo_padre} generado durante la búsqueda.
            nodo_final : Nodo de llegada (la meta).

        Returns:
            Lista de nodos desde el inicio hasta la meta.

        ¿Cómo funciona?
        Durante la búsqueda, cada vez que se visita un nodo se guarda
        de dónde vino (su padre). Al llegar a la meta, se sigue la
        cadena de padres hacia atrás hasta llegar al inicio, y luego
        se invierte la lista para tenerla en orden correcto.
        """
        camino  = []
        actual  = nodo_final
        while actual is not None:
            camino.append(actual)
            actual = origen.get(actual)
        camino.reverse()
        return camino

    @staticmethod
    def _heuristica(self, nodo: tuple, meta: tuple) -> float:
        """
        Heurística de distancia Manhattan para A*.

        ¿Por qué Manhattan y no Euclidiana?
        En el laberinto solo se puede mover en 4 direcciones (arriba, abajo,
        izquierda, derecha), no en diagonal. La distancia Manhattan refleja
        exactamente ese tipo de movimiento.

        Fórmula: |fila_actual - fila_meta| + |columna_actual - columna_meta|
        """
        return abs(nodo[0] - meta[0]) + abs(nodo[1] - meta[1])

    # ------------------------------------------------------------------ #
    #  Animación                                                          #
    # ------------------------------------------------------------------ #

    def _animar(self, pasos: list, camino: list, titulo: str) -> None:
        """
        Muestra la animación de la búsqueda paso a paso.

        Args:
            pasos  : Lista de matrices numpy, una por cada paso de la búsqueda.
            camino : Lista de nodos que forman el camino final.
            titulo : Nombre del algoritmo para mostrar en el título.

        Paleta de colores (índice → color):
            0 → CAMINO    gris claro  (celda transitable)
            1 → PARED     azul oscuro (pared)
            2 → INICIO    azul        (S)
            3 → META      rosa        (G)
            4 → EXPLORADO celda azul claro (ya visitada)
            5 → FRONTERA  amarillo    (en cola/pila)
            6 → CAMINO    verde       (solución final)
        """
        colores = [
            "#e8e8e0",  # 0 camino
            "#1a1a2e",  # 1 pared
            "#00b4d8",  # 2 inicio
            "#f72585",  # 3 meta
            "#a8dadc",  # 4 explorado
            "#ffd166",  # 5 frontera
            "#06d6a0",  # 6 camino final
        ]
        mapa_colores = ListedColormap(colores)

        figura, ejes = plt.subplots(figsize=(8, 8))
        figura.patch.set_facecolor("#0d0d1a")
        ejes.set_facecolor("#0d0d1a")

        imagen = ejes.imshow(
            pasos[0],
            cmap=mapa_colores,
            vmin=0, vmax=6,
            interpolation="nearest",
            aspect="equal",
        )

        # Anotaciones fijas de inicio y meta
        fila_inicio, columna_inicio = self.laberinto.inicio
        fila_meta,   columna_meta   = self.laberinto.meta
        ejes.text(columna_inicio, fila_inicio, "S", color="white",
                  fontsize=9, fontweight="bold", ha="center", va="center")
        ejes.text(columna_meta, fila_meta, "G", color="white",
                  fontsize=9, fontweight="bold", ha="center", va="center")

        # Leyenda
        elementos_leyenda = [
            mpatches.Patch(facecolor="#1a1a2e", edgecolor="gray", label="Pared"),
            mpatches.Patch(facecolor="#e8e8e0", edgecolor="gray", label="Camino"),
            mpatches.Patch(facecolor="#00b4d8", edgecolor="gray", label="Inicio (S)"),
            mpatches.Patch(facecolor="#f72585", edgecolor="gray", label="Meta (G)"),
            mpatches.Patch(facecolor="#a8dadc", edgecolor="gray", label="Explorado"),
            mpatches.Patch(facecolor="#ffd166", edgecolor="gray", label="Frontera"),
            mpatches.Patch(facecolor="#06d6a0", edgecolor="gray", label="Camino final"),
        ]
        ejes.legend(handles=elementos_leyenda, loc="upper right", fontsize=8,
                    framealpha=0.4, facecolor="#1a1a2e", labelcolor="white")

        titulo_texto = ejes.set_title(titulo, color="white", fontsize=13,
                                      fontweight="bold", pad=10)
        ejes.axis("off")

        total_pasos = len(pasos)

        def actualizar(frame):
            """
            Función que matplotlib llama en cada frame de la animación.
            Actualiza la imagen con el estado del laberinto en ese paso.
            Si es el último frame, muestra el camino final.
            """
            if frame < total_pasos:
                imagen.set_data(pasos[frame])
                titulo_texto.set_text(f"{titulo} — paso {frame + 1}/{total_pasos}")
            else:
                # Último frame: pintar el camino final
                matriz_final = pasos[-1].copy()
                for nodo in camino:
                    fila, columna = nodo
                    if matriz_final[fila][columna] not in (
                        self.laberinto.INICIO, self.laberinto.META
                    ):
                        matriz_final[fila][columna] = self.CAMINO
                imagen.set_data(matriz_final)
                pasos_camino = len(camino)
                titulo_texto.set_text(f"{titulo} — camino encontrado ({pasos_camino} pasos)")
            return [imagen]

        # interval: milisegundos entre frames (menor = más rápido)
        animacion = FuncAnimation(
            figura,
            actualizar,
            frames=total_pasos + 1,
            interval=50,
            blit=True,
            repeat=False,
        )

        plt.tight_layout()
        plt.show()

    def _matriz_base(self) -> np.ndarray:
        """
        Devuelve una copia limpia de la matriz del laberinto
        para usarla como punto de partida de cada frame de animación.
        """
        return self.laberinto.matriz.copy().astype(int)

    def _celebrar(self, titulo: str, tiempo_segundos: float, pasos_camino: int) -> None:
        """
        Muestra una ventana de celebración con confeti y el tiempo de búsqueda.

        Args:
            titulo          : Nombre del algoritmo.
            tiempo_segundos : Tiempo que tardó el algoritmo en encontrar la ruta.
            pasos_camino    : Longitud del camino encontrado.

        El confeti se genera lanzando partículas de colores aleatorios
        desde la parte superior de la figura con posiciones y tamaños aleatorios.
        """
        figura_cel, ejes_cel = plt.subplots(figsize=(7, 5))
        figura_cel.patch.set_facecolor("#0d0d1a")
        ejes_cel.set_facecolor("#0d0d1a")
        ejes_cel.axis("off")

        # Generar confeti: puntos de colores aleatorios
        colores_confeti = ["#f72585", "#ffd166", "#06d6a0", "#00b4d8",
                           "#ff9f1c", "#e9c46a", "#a8dadc", "#ffffff"]
        cantidad_confeti = 300
        x_confeti = np.random.uniform(0, 1, cantidad_confeti)
        y_confeti = np.random.uniform(0, 1, cantidad_confeti)
        colores_aleatorios = np.random.choice(colores_confeti, cantidad_confeti)
        tamanios_aleatorios = np.random.uniform(50, 300, cantidad_confeti)

        ejes_cel.scatter(x_confeti, y_confeti, c=colores_aleatorios,
                         s=tamanios_aleatorios, alpha=0.8, zorder=2)

        # Texto de celebración
        ejes_cel.text(0.5, 0.72, "¡Ruta encontrada!", color="white",
                      fontsize=22, fontweight="bold", ha="center", va="center",
                      transform=ejes_cel.transAxes, zorder=3)

        ejes_cel.text(0.5, 0.55, titulo, color="#ffd166",
                      fontsize=15, ha="center", va="center",
                      transform=ejes_cel.transAxes, zorder=3)

        ejes_cel.text(0.5, 0.38,
                      f"⏱ Tiempo de búsqueda: {tiempo_segundos:.4f} segundos",
                      color="#a8dadc", fontsize=12, ha="center", va="center",
                      transform=ejes_cel.transAxes, zorder=3)

        ejes_cel.text(0.5, 0.25,
                      f"📍 Longitud del camino: {pasos_camino} pasos",
                      color="#06d6a0", fontsize=12, ha="center", va="center",
                      transform=ejes_cel.transAxes, zorder=3)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    #  BFS — Búsqueda por Amplitud

    def bfs(self) -> list:
        """
        Búsqueda por amplitud (BFS).

        ¿Cómo funciona?
        Usa una cola (FIFO). Explora todos los vecinos del nodo actual
        antes de avanzar al siguiente nivel. Esto garantiza que el primer
        camino encontrado sea el más corto en número de pasos.

        Estructura usada: deque (cola doble, eficiente para agregar y quitar
        por ambos extremos).

        Complejidad:
            Tiempo  : O(V + E) donde V = nodos, E = aristas
            Espacio : O(V)
        """
        inicio  = self.laberinto.inicio
        meta    = self.laberinto.meta
        cola    = deque([inicio])           # cola FIFO
        origen  = {inicio: None}            # diccionario de padres
        pasos   = []                        # frames para la animación

        tiempo_inicio = time.time()         # iniciar cronómetro

        while cola:
            actual = cola.popleft()         # sacar el primero de la cola

            # Capturar el estado actual de la matriz para animación
            matriz_paso = self._matriz_base()
            for nodo in origen:
                fila, columna = nodo
                if matriz_paso[fila][columna] not in (
                    self.laberinto.INICIO, self.laberinto.META
                ):
                    matriz_paso[fila][columna] = self.EXPLORADO
            for nodo in cola:
                fila, columna = nodo
                if matriz_paso[fila][columna] not in (
                    self.laberinto.INICIO, self.laberinto.META
                ):
                    matriz_paso[fila][columna] = self.FRONTERA
            pasos.append(matriz_paso)

            if actual == meta:
                break

            for vecino, peso in self.grafo.vecinos(actual):
                if vecino not in origen:
                    origen[vecino] = actual  # guardar de dónde vino
                    cola.append(vecino)

        tiempo_total = time.time() - tiempo_inicio   # detener cronómetro
        camino = self._reconstruir_camino(origen, meta)
        self._animar(pasos, camino, "BFS — Búsqueda por Amplitud")
        self._celebrar("BFS — Búsqueda por Amplitud", tiempo_total, len(camino))
        return camino

    # ------------------------------------------------------------------ #
    #  DFS — Depth-First Search                                           #
    # ------------------------------------------------------------------ #

    def dfs(self) -> list:
        """
        Búsqueda por profundidad (DFS).

        ¿Cómo funciona?
        Usa una pila (LIFO). Siempre explora el camino más profundo
        antes de retroceder. No garantiza el camino más corto, pero
        usa menos memoria que BFS en muchos casos.

        Estructura usada: lista de Python usada como pila (append/pop).

        ¿Por qué se usa una bandera encontrado en lugar de return dentro del while?
        Usar return dentro del while cortaría el flujo antes de llegar a la
        animación y al retorno del camino. Con la bandera, el while termina
        naturalmente y el resto del código siempre se ejecuta.

        Complejidad:
            Tiempo  : O(V + E)
            Espacio : O(V)
        """
        inicio    = self.laberinto.inicio
        meta      = self.laberinto.meta
        pila      = [inicio]                # pila LIFO
        origen    = {inicio: None}
        visitados = set()
        pasos     = []
        encontrado = False                  # bandera para controlar el while

        tiempo_inicio = time.time()         # iniciar cronómetro

        while pila and not encontrado:
            actual = pila.pop()             # sacar el último de la pila

            if actual in visitados:
                continue
            visitados.add(actual)

            # Capturar estado para animación
            matriz_paso = self._matriz_base()
            for nodo in visitados:
                fila, columna = nodo
                if matriz_paso[fila][columna] not in (
                    self.laberinto.INICIO, self.laberinto.META
                ):
                    matriz_paso[fila][columna] = self.EXPLORADO
            for nodo in pila:
                fila, columna = nodo
                if matriz_paso[fila][columna] not in (
                    self.laberinto.INICIO, self.laberinto.META
                ):
                    matriz_paso[fila][columna] = self.FRONTERA
            pasos.append(matriz_paso)

            if actual == meta:
                encontrado = True           # activa la bandera y termina el while
            else:
                for vecino, peso in self.grafo.vecinos(actual):
                    if vecino not in visitados:
                        origen[vecino] = actual
                        pila.append(vecino)

        # Siempre se llega aquí, sin importar cómo terminó el while
        tiempo_total = time.time() - tiempo_inicio   # detener cronómetro
        camino = self._reconstruir_camino(origen, meta)
        self._animar(pasos, camino, "DFS — Búsqueda por Profundidad")
        self._celebrar("DFS — Búsqueda por Profundidad", tiempo_total, len(camino))
        return camino

    # ------------------------------------------------------------------ #
    #  A* Search                                                          #
    # ------------------------------------------------------------------ #

    def a_star(self) -> list:
        """
        Búsqueda A* (A estrella).

        ¿Cómo funciona?
        Combina el costo real del camino recorrido (g) con una estimación
        del costo restante hasta la meta (h, heurística). La función de
        evaluación es:
            f(n) = g(n) + h(n)

        Siempre expande el nodo con menor f(n), lo que lo hace más
        eficiente que BFS y DFS cuando la heurística es buena.

        Heurística usada: distancia Manhattan, admisible para laberintos
        porque nunca sobreestima el costo real (solo movimientos en 4 direcciones).

        Estructura usada: montículo mínimo (heap) para obtener siempre
        el nodo con menor f(n) en O(log n).

        Complejidad:
            Tiempo  : O(E log V) en el peor caso
            Espacio : O(V)
        """
        inicio = self.laberinto.inicio
        meta   = self.laberinto.meta

        # heap: (f, g, nodo)
        # f = costo total estimado, g = costo real acumulado
        heap   = [(0 + self.heuristica(inicio, meta), 0, inicio)]
        origen = {inicio: None}
        costo  = {inicio: 0}              # costo real acumulado por nodo
        pasos  = []

        tiempo_inicio = time.time()       # iniciar cronómetro

        while heap:
            f_actual, g_actual, actual = heapq.heappop(heap)

            # Capturar estado para animación
            matriz_paso = self._matriz_base()
            for nodo in origen:
                fila, columna = nodo
                if matriz_paso[fila][columna] not in (
                    self.laberinto.INICIO, self.laberinto.META
                ):
                    matriz_paso[fila][columna] = self.EXPLORADO
            for _, _, nodo in heap:
                fila, columna = nodo
                if matriz_paso[fila][columna] not in (
                    self.laberinto.INICIO, self.laberinto.META
                ):
                    matriz_paso[fila][columna] = self.FRONTERA
            pasos.append(matriz_paso)

            if actual == meta:
                break

            for vecino in self.grafo.vecinos(actual):
                # Costo de moverse a un vecino = costo actual + 1
                nuevo_costo = g_actual + 1

                if vecino not in costo or nuevo_costo < costo[vecino]:
                    costo[vecino]  = nuevo_costo
                    origen[vecino] = actual
                    f = nuevo_costo + self.heuristica(vecino, meta)
                    heapq.heappush(heap, (f, nuevo_costo, vecino))

        tiempo_total = time.time() - tiempo_inicio   # detener cronómetro
        camino = self._reconstruir_camino(origen, meta)
        self._animar(pasos, camino, "A* — Búsqueda Informada")
        self._celebrar("A* — Búsqueda Informada", tiempo_total, len(camino))
        return camino


# ------------------------------------------------------------------ #
#  Función principal con parámetros explícitos                        #
# ------------------------------------------------------------------ #

def main(n: int, coordenada_salida: Optional[tuple], coordenada_meta: Optional[tuple], heuristica=None):
    """
    Función principal del proyecto.

    Recibe los insumos del problema, construye el laberinto y el grafo,
    y ejecuta los tres algoritmos de búsqueda.

    Args:
        n                 : Dimensión del laberinto (tamaño N x N, debe ser impar).
        coordenada_salida : Tupla (fila, columna) del punto de inicio.
                            Si se pasa None, usa la esquina superior izquierda (1, 1).
        coordenada_meta   : Tupla (fila, columna) del punto de llegada.
                            Si se pasa None, usa el centro de la matriz.
        heuristica        : Función heurística opcional para A*.
                            Si se pasa None, usa distancia Manhattan por defecto.

    Ejemplo de uso:
        main(n=21, coordenada_salida=(1,1), coordenada_meta=(10,10))
        main(n=31, coordenada_salida=None, coordenada_meta=None)
    """
    # 1. Construir el laberinto con las dimensiones indicadas
    laberinto = Laberinto(tamano=n)

    # 2. Sobreescribir inicio y meta si se pasaron coordenadas explícitas
    if coordenada_salida is not None:
        laberinto.inicio = coordenada_salida
        laberinto.matriz[coordenada_salida[0]][coordenada_salida[1]] = laberinto.INICIO

    if coordenada_meta is not None:
        laberinto.meta = coordenada_meta
        laberinto.matriz[coordenada_meta[0]][coordenada_meta[1]] = laberinto.META

    # 3. Transformar la matriz en grafo
    grafo = Grafo(laberinto)

    # 4. Crear el buscador con la heurística indicada (o Manhattan por defecto)
    buscador = Buscador(laberinto, grafo, heuristica=heuristica)

    print(f"Laberinto {n}x{n} | Salida: {laberinto.inicio} | Meta: {laberinto.meta}")
    print("-" * 50)

    # 5. Ejecutar los tres algoritmos e imprimir rutas
    print("Corriendo BFS — Búsqueda por Amplitud...")
    camino_bfs = buscador.bfs()
    print(f"BFS — ruta calculada: {camino_bfs}")
    print(f"BFS — pasos: {len(camino_bfs)}")
    print("-" * 50)

    print("Corriendo DFS — Búsqueda por Profundidad...")
    camino_dfs = buscador.dfs()
    print(f"DFS — ruta calculada: {camino_dfs}")
    print(f"DFS — pasos: {len(camino_dfs)}")
    print("-" * 50)

    print("Corriendo A* — Búsqueda Informada...")
    camino_astar = buscador.a_star()
    print(f"A*  — ruta calculada: {camino_astar}")
    print(f"A*  — pasos: {len(camino_astar)}")


# ------------------------------------------------------------------ #
#  Punto de entrada                                                   #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    main(
        n=21,
        coordenada_salida=None,   # None = usa esquina superior izquierda (1,1)
        coordenada_meta=None,     # None = usa el centro de la matriz
    )
