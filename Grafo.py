"""
Grafo.py — Construcción del grafo a partir del laberinto
=========================================================
Clase Grafo:
  - Recibe un objeto Laberinto como parámetro
  - Construye una lista de adyacencia (diccionario)
  - Cada nodo es una celda de camino (fila, columna)
  - Dos nodos están conectados si son adyacentes sin pared entre ellos

Uso:
    from Laberinto import Laberinto
    from Grafo import Grafo

    laberinto = Laberinto(tamano=21)
    grafo = Grafo(laberinto)
    grafo.mostrar_adyacencia()
"""

from Laberinto import Laberinto


class Grafo:

    def __init__(self, laberinto_entrada: Laberinto):
        """
        Construye el grafo a partir de un laberinto.

        Args:
            laberinto_entrada: Objeto Laberinto ya generado.

        La lista de adyacencia es un diccionario donde:
            - Cada clave es una tupla (fila, columna) que representa un nodo
            - Su valor es una lista de tuplas vecinas con las que está conectado

        Ejemplo:
            {
                (1, 1): [(1, 3), (3, 1)],
                (1, 3): [(1, 1), (1, 5)],
                ...
            }
        """
        self.laberinto = laberinto_entrada
        # Diccionario principal: nodo → lista de vecinos conectados
        self.adyacencia = {}
        self._construir()

    def _es_camino(self, fila: int, columna: int) -> bool:
        """
        Verifica si una celda es camino, inicio o meta (es decir, no es pared).

        Se consideran transitables los valores:
            CAMINO (0 o 1 según tu configuración), INICIO y META.
        Se excluye solo PARED.
        """
        valor = int(self.laberinto.matriz[fila][columna])
        return valor != self.laberinto.PARED

    def _construir(self) -> None:
        """
        Recorre toda la matriz y construye la lista de adyacencia.

        Para cada celda que no sea pared:
        1. Se agrega como nodo al diccionario (si no existe aún).
        2. Se revisan sus 4 vecinos directos (arriba, abajo, izquierda, derecha)
           a distancia 1 (no 2, porque aquí nos movemos celda a celda).
        3. Si el vecino tampoco es pared, se agrega como conexión.

        ¿Por qué distancia 1 y no 2?
        En _generar() del laberinto se trabaja con distancia 2 porque se
        construye la estructura. Aquí ya está construida, entonces simplemente
        revisamos si la celda de al lado es transitable o no.
        """
        tamano = self.laberinto.tamano
        # Desplazamientos a los 4 vecinos directos: arriba, abajo, izquierda, derecha
        direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for fila in range(tamano):
            for columna in range(tamano):
                if not self._es_camino(fila, columna):
                    continue  # saltar paredes

                nodo_actual = (fila, columna)

                # Inicializar el nodo en el diccionario si no existe
                if nodo_actual not in self.adyacencia:
                    self.adyacencia[nodo_actual] = []

                # Revisar los 4 vecinos directos
                for delta_fila, delta_columna in direcciones:
                    fila_vecina    = fila    + delta_fila
                    columna_vecina = columna + delta_columna

                    # Verificar que el vecino esté dentro de los límites
                    if not (0 <= fila_vecina < tamano and 0 <= columna_vecina < tamano):
                        continue

                    # Si el vecino es camino, agregar la conexión
                    if self._es_camino(fila_vecina, columna_vecina):
                        nodo_vecino = (fila_vecina, columna_vecina)
                        self.adyacencia[nodo_actual].append((nodo_vecino, 1))

    def vecinos(self, nodo: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Devuelve la lista de nodos conectados a un nodo dado.

        Args:
            nodo: Tupla (fila, columna) del nodo a consultar.

        Returns:
            Lista de tuplas vecinas. Lista vacía si el nodo no existe.

        Ejemplo:
            grafo.vecinos((1, 1))  →  [(1, 3), (3, 1)]
        """
        return self.adyacencia.get(nodo, [])

    def cantidad_nodos(self) -> int:
        """Devuelve el número total de nodos (celdas transitables) en el grafo."""
        return len(self.adyacencia)

    def cantidad_aristas(self) -> int:
        """
        Devuelve el número total de aristas (conexiones) en el grafo.

        Como el grafo es no dirigido (si A conecta con B, B conecta con A),
        se divide entre 2 para no contar cada conexión dos veces.
        """
        total = sum(len(vecinos) for vecinos in self.adyacencia.values())
        return total // 2

    def mostrar_adyacencia(self) -> None:
        """
        Imprime la lista de adyacencia completa en consola.

        Solo muestra los primeros 10 nodos para no saturar la terminal.
        Útil para verificar que el grafo se construyó correctamente.
        """
        print(f"=== Grafo del Laberinto ===")
        print(f"Nodos     : {self.cantidad_nodos()}")
        print(f"Aristas   : {self.cantidad_aristas()}")
        print(f"Inicio    : {self.laberinto.inicio}")
        print(f"Meta      : {self.laberinto.meta}")
        print()
        print("Lista de adyacencia (primeros 10 nodos):")
        for i, (nodo, lista_vecinos) in enumerate(self.adyacencia.items()):
            print(f"  {nodo} → {lista_vecinos}")
            if i >= 9:
                print(f"  ... ({self.cantidad_nodos() - 10} nodos más)")
                break


# ------------------------------------------------------------------ #
#  Punto de entrada                                                   #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    laberinto = Laberinto(tamano=21)
    grafo = Grafo(laberinto)
    grafo.mostrar_adyacencia()