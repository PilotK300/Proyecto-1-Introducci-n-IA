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

    buscador.bfs()    # Busqueda por Amplitud
    buscador.dfs()    # Busqueda por Profundidad
    buscador.a_star() # A*
"""

import heapq
import time
from collections import deque
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation

from Laberinto import Laberinto
from Grafo import Grafo


def _manhattan(nodo: tuple, meta: tuple) -> float:
    """
    Heuristica de distancia Manhattan para A*.

    Funcion suelta fuera de la clase para evitar dependencia de self.
    Puede ser reemplazada por cualquier funcion (nodo, meta) -> float.

    Por que Manhattan y no Euclidiana?
    En el laberinto solo se puede mover en 4 direcciones (arriba, abajo,
    izquierda, derecha), no en diagonal. La distancia Manhattan refleja
    exactamente ese tipo de movimiento y nunca sobreestima el costo real.

    Formula: |fila_actual - fila_meta| + |columna_actual - columna_meta|
    """
    return abs(nodo[0] - meta[0]) + abs(nodo[1] - meta[1])


class Buscador:

    # Valores extra para la animacion (ademas de los de Laberinto)
    EXPLORADO = 4   # celda ya visitada por el algoritmo
    FRONTERA  = 5   # celda en cola/pila, pendiente de explorar
    CAMINO    = 6   # camino final encontrado

    def __init__(self, laberinto: Laberinto, grafo: Grafo, heuristica=None):
        """
        Args:
            laberinto  : Objeto Laberinto ya generado.
            grafo      : Objeto Grafo construido a partir del laberinto.
            heuristica : Funcion heuristica opcional para A*.
                         Debe aceptar (nodo, meta) y devolver un numero.
                         Si no se pasa, usa distancia Manhattan por defecto.
        """
        self.laberinto  = laberinto
        self.grafo      = grafo
        self.macro_adj  = {}
        # Si no se pasa heuristica usa Manhattan, de lo contrario la funcion recibida
        self.heuristica = heuristica if heuristica is not None else _manhattan

    # ------------------------------------------------------------------ #
    #  LOGICA DE ABSTRACCION (MACRO-GRAFO)                                #
    # ------------------------------------------------------------------ #

    def es_nodo_decision(self, pos: tuple[int,int]) -> bool:
        """
        Determina si una posicion es un nodo de decision.

        Un nodo de decision es aquel donde el agente debe elegir
        una direccion: inicio, meta, intersecciones o callejones sin salida.
        Los pasillos rectos (grado 2) no son nodos de decision.
        """
        if pos == self.laberinto.inicio or pos == self.laberinto.meta:
            return True
        grado = len(self.grafo.vecinos(pos))
        return grado != 2

    def _explorar_corredor(self, origen: tuple[int,int], primer_paso: tuple[int,int]) -> tuple:
        """
        Recorre un corredor recto desde origen hasta el siguiente nodo de decision.

        Args:
            origen      : Nodo de decision desde donde empieza el corredor.
            primer_paso : Primera celda del corredor.

        Returns:
            Tupla (camino, destino) donde camino es la lista de celdas
            recorridas y destino es el nodo de decision al final del corredor.
        """
        camino = [origen, primer_paso]
        anterior, actual = origen, primer_paso
        while not self.es_nodo_decision(actual):
            vecinos = self.grafo.vecinos(actual)
            # Desempacar (nodo, peso) y filtrar el nodo anterior
            siguiente = [n for n, peso in vecinos if n != anterior][0]
            camino.append(siguiente)
            anterior, actual = actual, siguiente
        return camino, actual

    def construir_macro_grafo(self) -> None:
        """
        Construye el macro-grafo conectando solo los nodos de decision.

        En lugar de trabajar con todas las celdas, el macro-grafo
        agrupa los corredores rectos en una sola arista con peso igual
        a la longitud del corredor. Esto reduce el espacio de busqueda.
        """
        self.macro_adj = {}
        # Solo los nodos de decision son nodos del macro-grafo
        nodos_reales = [n for n in self.grafo.adyacencia if self.es_nodo_decision(n)]
        for nodo in nodos_reales:
            self.macro_adj[nodo] = []
            for v_inmediato, peso in self.grafo.vecinos(nodo):
                pasos, destino = self._explorar_corredor(nodo, v_inmediato)
                self.macro_adj[nodo].append({
                    "destino": destino,
                    "peso": len(pasos) - 1,     # peso = longitud del corredor
                    "camino_detallado": pasos
                })

    def _reconstruir_ruta_completa(self, origen_macro: dict, meta: tuple) -> list:
        """
        Reconstruye la ruta completa paso a paso a partir de los macro-nodos.

        Args:
            origen_macro : Diccionario {nodo: (padre, celdas_corredor)}.
            meta         : Nodo de llegada.

        Returns:
            Lista completa de celdas desde inicio hasta meta.
        """
        camino_final = []
        actual = meta
        while actual is not None and actual != self.laberinto.inicio:
            datos = origen_macro.get(actual)
            if datos is None:
                break
            padre, celdas_corredor = datos
            # Agregar las celdas intermedias del corredor en orden inverso
            camino_final.extend(reversed(celdas_corredor[1:]))
            actual = padre
        camino_final.append(self.laberinto.inicio)
        camino_final.reverse()
        return camino_final

    # ------------------------------------------------------------------ #
    #  Utilidades compartidas                                             #
    # ------------------------------------------------------------------ #

    def _reconstruir_camino(self, origen: dict, nodo_final: tuple) -> list:
        """
        Reconstruye el camino desde la meta hasta el inicio siguiendo
        el diccionario de padres (origen).

        Args:
            origen     : Diccionario {nodo: nodo_padre} generado durante la busqueda.
            nodo_final : Nodo de llegada (la meta).

        Returns:
            Lista de nodos desde el inicio hasta la meta.

        Como funciona?
        Durante la busqueda, cada vez que se visita un nodo se guarda
        de donde vino (su padre). Al llegar a la meta, se sigue la
        cadena de padres hacia atras hasta llegar al inicio, y luego
        se invierte la lista para tenerla en orden correcto.
        """
        camino = []
        actual = nodo_final
        while actual is not None:
            camino.append(actual)
            actual = origen.get(actual)
        camino.reverse()
        return camino

    # ------------------------------------------------------------------ #
    #  Animacion                                                          #
    # ------------------------------------------------------------------ #

    def _animar(self, pasos: list, camino: list, titulo: str) -> None:
        """
        Muestra la animacion de la busqueda paso a paso.

        Args:
            pasos  : Lista de matrices numpy, una por cada paso de la busqueda.
            camino : Lista de nodos que forman el camino final.
            titulo : Nombre del algoritmo para mostrar en el titulo.

        Paleta de colores (indice -> color):
            0 -> CAMINO    gris claro  (celda transitable)
            1 -> PARED     azul oscuro (pared)
            2 -> INICIO    azul        (S)
            3 -> META      rosa        (G)
            4 -> EXPLORADO celda azul claro (ya visitada)
            5 -> FRONTERA  amarillo    (en cola/pila)
            6 -> CAMINO    verde       (solucion final)
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
            Funcion que matplotlib llama en cada frame de la animacion.
            Actualiza la imagen con el estado del laberinto en ese paso.
            Si es el ultimo frame, muestra el camino final.
            """
            if frame < total_pasos:
                imagen.set_data(pasos[frame])
                titulo_texto.set_text(f"{titulo} — paso {frame + 1}/{total_pasos}")
            else:
                # Ultimo frame: pintar el camino final
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

        # Se guarda en variable para evitar que el recolector de basura
        # de Python elimine la animacion antes de que termine de mostrarse
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
        para usarla como punto de partida de cada frame de animacion.
        """
        return self.laberinto.matriz.copy().astype(int)

    def _celebrar(self, titulo: str, tiempo_segundos: float, pasos_camino: int, costo=None) -> None:
        """
        Muestra una ventana de celebracion con confeti y el tiempo de busqueda.

        Args:
            titulo          : Nombre del algoritmo.
            tiempo_segundos : Tiempo que tardo el algoritmo en encontrar la ruta.
            pasos_camino    : Longitud del camino encontrado.
            costo           : Costo total de la ruta (solo A*, None para BFS y DFS).
        """
        figura_cel, ejes_cel = plt.subplots(figsize=(7, 5))
        figura_cel.patch.set_facecolor("#0d0d1a")
        ejes_cel.set_facecolor("#0d0d1a")
        ejes_cel.axis("off")
        # Posiciones verticales de cada texto
        pos_titulo    = 0.75
        pos_algoritmo = 0.58
        pos_tiempo    = 0.42
        pos_pasos     = 0.28
        pos_costo     = 0.13

        ejes_cel.text(0.5, pos_titulo, "Ruta encontrada!", color="white",
                      fontsize=22, fontweight="bold", ha="center", va="center",
                      transform=ejes_cel.transAxes, zorder=3)

        ejes_cel.text(0.5, pos_algoritmo, titulo, color="#ffd166",
                      fontsize=15, ha="center", va="center",
                      transform=ejes_cel.transAxes, zorder=3)

        ejes_cel.text(0.5, pos_tiempo,
                      f"Tiempo de busqueda: {tiempo_segundos:.4f} segundos",
                      color="#a8dadc", fontsize=12, ha="center", va="center",
                      transform=ejes_cel.transAxes, zorder=3)

        ejes_cel.text(0.5, pos_pasos,
                      f"Longitud del camino: {pasos_camino} pasos",
                      color="#06d6a0", fontsize=12, ha="center", va="center",
                      transform=ejes_cel.transAxes, zorder=3)

        # Solo A* muestra el costo
        if costo is not None:
            ejes_cel.text(0.5, pos_costo,
                          f"Costo total de la ruta: {costo}",
                          color="#f72585", fontsize=12, ha="center", va="center",
                          transform=ejes_cel.transAxes, zorder=3)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    #  BFS — Busqueda por Amplitud                                        #
    # ------------------------------------------------------------------ #

    def bfs(self) -> list:
        """
        Busqueda por amplitud (BFS).

        Como funciona?
        Usa una cola (FIFO). Explora todos los vecinos del nodo actual
        antes de avanzar al siguiente nivel. Esto garantiza que el primer
        camino encontrado sea el mas corto en numero de pasos.

        Estructura usada: deque (cola doble, eficiente para agregar y quitar
        por ambos extremos).

        Complejidad:
            Tiempo  : O(V + E) donde V = nodos, E = aristas
            Espacio : O(V)
        """
        inicio = self.laberinto.inicio
        meta   = self.laberinto.meta
        cola   = deque([inicio])        # cola FIFO
        origen = {inicio: None}         # diccionario de padres
        pasos  = []                     # frames para la animacion

        tiempo_inicio = time.time()     # iniciar cronometro

        while cola:
            actual = cola.popleft()     # sacar el primero de la cola

            # Capturar el estado actual de la matriz para animacion
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
                    origen[vecino] = actual     # guardar de donde vino
                    cola.append(vecino)

        tiempo_total = time.time() - tiempo_inicio      # detener cronometro
        camino = self._reconstruir_camino(origen, meta)
        self._animar(pasos, camino, "BFS — Busqueda por Amplitud")
        self._celebrar("BFS — Busqueda por Amplitud", tiempo_total, len(camino))
        return camino

    # ------------------------------------------------------------------ #
    #  DFS — Busqueda por Profundidad                                     #
    # ------------------------------------------------------------------ #

    def dfs(self) -> list:
        """
        Busqueda por profundidad (DFS).

        Como funciona?
        Usa una pila (LIFO). Siempre explora el camino mas profundo
        antes de retroceder. No garantiza el camino mas corto, pero
        usa menos memoria que BFS en muchos casos.

        Estructura usada: lista de Python usada como pila (append/pop).

        Por que se usa una bandera encontrado en lugar de return dentro del while?
        Usar return dentro del while cortaria el flujo antes de llegar a la
        animacion y al retorno del camino. Con la bandera, el while termina
        naturalmente y el resto del codigo siempre se ejecuta.

        Complejidad:
            Tiempo  : O(V + E)
            Espacio : O(V)
        """
        inicio     = self.laberinto.inicio
        meta       = self.laberinto.meta
        pila       = [inicio]               # pila LIFO
        origen     = {inicio: None}
        visitados  = set()
        pasos      = []
        encontrado = False                  # bandera para controlar el while

        tiempo_inicio = time.time()         # iniciar cronometro

        while pila and not encontrado:
            actual = pila.pop()             # sacar el ultimo de la pila

            if actual in visitados:
                continue
            visitados.add(actual)

            # Capturar estado para animacion
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

        # Siempre se llega aqui, sin importar como termino el while
        tiempo_total = time.time() - tiempo_inicio      # detener cronometro
        camino = self._reconstruir_camino(origen, meta)
        self._animar(pasos, camino, "DFS — Busqueda por Profundidad")
        self._celebrar("DFS — Busqueda por Profundidad", tiempo_total, len(camino))
        return camino

    # ------------------------------------------------------------------ #
    #  A* — Busqueda Informada                                            #
    # ------------------------------------------------------------------ #

    def a_star(self) -> list:
        """
        Busqueda A* (A estrella).

        Como funciona?
        Combina el costo real del camino recorrido (g) con una estimacion
        del costo restante hasta la meta (h, heuristica). La funcion de
        evaluacion es:
            f(n) = g(n) + h(n)

        Siempre expande el nodo con menor f(n), lo que lo hace mas
        eficiente que BFS y DFS cuando la heuristica es buena.

        Heuristica usada: distancia Manhattan, admisible para laberintos
        porque nunca sobreestima el costo real (solo movimientos en 4 direcciones).

        Estructura usada: monticulo minimo (heap) para obtener siempre
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
        costo  = {inicio: 0}            # costo real acumulado por nodo
        pasos  = []

        tiempo_inicio = time.time()     # iniciar cronometro

        while heap:
            f_actual, g_actual, actual = heapq.heappop(heap)

            # Capturar estado para animacion
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

            for vecino, peso in self.grafo.vecinos(actual):
                # Costo de moverse a un vecino = costo actual + peso
                nuevo_costo = g_actual + peso

                if vecino not in costo or nuevo_costo < costo[vecino]:
                    costo[vecino]  = nuevo_costo
                    origen[vecino] = actual
                    f = nuevo_costo + self.heuristica(vecino, meta)
                    heapq.heappush(heap, (f, nuevo_costo, vecino))

        tiempo_total = time.time() - tiempo_inicio      # detener cronometro
        costo_final  = costo.get(meta, 0)               # costo real hasta la meta
        camino = self._reconstruir_camino(origen, meta)
        self._animar(pasos, camino, "A* — Busqueda Informada")
        self._celebrar("A* — Busqueda Informada", tiempo_total, len(camino), costo_final)
        return camino

    # ------------------------------------------------------------------ #
    #  A* OPTIMIZADO (MACRO-GRAFO)                                        #
    # ------------------------------------------------------------------ #

    def astar_macro(self) -> tuple[list,int]:
        """
        A* optimizado usando el macro-grafo.

        En lugar de explorar celda a celda, opera sobre nodos de decision
        conectados por aristas con peso igual a la longitud del corredor.
        Esto reduce significativamente el numero de nodos expandidos.

        Returns:
            Tupla (camino_completo, nodos_expandidos).
        """
        self.construir_macro_grafo()
        inicio, meta = self.laberinto.inicio, self.laberinto.meta
        heap             = [(0 + self.heuristica(inicio, meta), 0, inicio)]
        origen_macro     = {inicio: None}
        costo_g          = {inicio: 0}
        nodos_expandidos = 0

        while heap:
            f, g_actual, actual = heapq.heappop(heap)
            nodos_expandidos += 1

            if actual == meta:
                break

            for arista in self.macro_adj.get(actual, []):
                vecino  = arista["destino"]
                peso    = arista["peso"]
                nuevo_g = g_actual + peso
                if vecino not in costo_g or nuevo_g < costo_g[vecino]:
                    costo_g[vecino]      = nuevo_g
                    origen_macro[vecino] = (actual, arista["camino_detallado"])
                    f_nuevo = nuevo_g + self.heuristica(vecino, meta)
                    heapq.heappush(heap, (f_nuevo, nuevo_g, vecino))

        camino = self._reconstruir_ruta_completa(origen_macro, meta)
        return camino, nodos_expandidos


# ------------------------------------------------------------------ #
#  Funcion principal con parametros explicitos                        #
# ------------------------------------------------------------------ #

def main(n: int, coordenada_salida: Optional[tuple], coordenada_meta: Optional[tuple], heuristica=None) -> None:
    """
    Funcion principal del proyecto.

    Recibe los insumos del problema, construye el laberinto y el grafo,
    y ejecuta los tres algoritmos de busqueda.

    Args:
        n                 : Dimension del laberinto (tamano N x N, debe ser impar).
        coordenada_salida : Tupla (fila, columna) del punto de inicio.
                            Si se pasa None, usa la esquina superior izquierda (1, 1).
        coordenada_meta   : Tupla (fila, columna) del punto de llegada.
                            Si se pasa None, usa el centro de la matriz.
        heuristica        : Funcion heuristica opcional para A*.
                            Si se pasa None, usa distancia Manhattan por defecto.

    Ejemplo de uso:
        main(n=21, coordenada_salida=(1,1), coordenada_meta=(10,10))
        main(n=31, coordenada_salida=None, coordenada_meta=None)
    """
    # 1. Construir el laberinto con las dimensiones indicadas
    laberinto = Laberinto(tamano=n)

    # 2. Sobreescribir inicio y meta si se pasaron coordenadas explicitas
    if coordenada_salida is not None:
        laberinto.inicio = coordenada_salida
        laberinto.matriz[coordenada_salida[0]][coordenada_salida[1]] = laberinto.INICIO

    if coordenada_meta is not None:
        laberinto.meta = coordenada_meta
        laberinto.matriz[coordenada_meta[0]][coordenada_meta[1]] = laberinto.META

    # 3. Transformar la matriz en grafo
    grafo = Grafo(laberinto)

    # 4. Crear el buscador con la heuristica indicada (o Manhattan por defecto)
    buscador = Buscador(laberinto, grafo, heuristica=heuristica)

    print(f"Laberinto {n}x{n} | Salida: {laberinto.inicio} | Meta: {laberinto.meta}")
    print("-" * 50)

    # 5. Ejecutar los tres algoritmos e imprimir rutas
    print("Corriendo BFS — Busqueda por Amplitud...")
    camino_bfs = buscador.bfs()
    print(f"BFS — ruta calculada: {camino_bfs}")
    print(f"BFS — pasos: {len(camino_bfs)}")
    print("-" * 50)

    print("Corriendo DFS — Busqueda por Profundidad...")
    camino_dfs = buscador.dfs()
    print(f"DFS — ruta calculada: {camino_dfs}")
    print(f"DFS — pasos: {len(camino_dfs)}")
    print("-" * 50)

    print("Corriendo A* — Busqueda Informada...")
    camino_astar = buscador.a_star()
    print(f"A*  — ruta calculada: {camino_astar}")
    print(f"A*  — pasos: {len(camino_astar)}")


# ------------------------------------------------------------------ #
#  Punto de entrada                                                   #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    main(
        n=55,
        coordenada_salida=None,   # None = usa esquina superior izquierda (1,1)
        coordenada_meta=None,     # None = usa el centro de la matriz
    )