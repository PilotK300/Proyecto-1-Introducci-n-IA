"""
maze.py — Generador de Laberintos con Aldous-Broder
====================================================
Clase Maze:
  - Tamaño configurable (debe ser impar para centrar la meta)
  - Generación aleatoria con algoritmo Aldous-Broder
  - Meta en el centro de la matriz
  - Entrada en la esquina superior izquierda
  - Visualización con matplotlib
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches #Esto es para dar formas mejor froma al laberinto
from matplotlib.colors import ListedColormap



class Laberinto:
    CAMINO = 0
    PARED = 1
    INICIO = 2
    META = 3
    def __init__(self, tamano: int = 21):
        """
        :param tamano: El tamano de la matriz, siempre va a ser impar y minimo 5 para
        manteer el algoritmo de Aldous-Broder

        """
        if tamano <= 5:
            tamano = 5
        if tamano % 2 == 0:
            tamano += 1
        self.tamano = tamano
        self.matriz = np.ones((self.tamano, self.tamano), dtype=int) #Definir int para no tener problemas
        self.inicio = (1, 1)  # esquina superior izquierda
        self.meta = (tamano // 2, tamano // 2)  # centro exacto

        self._generar()
    def _vecinos_de_celda(self, fila: int, columna: int) -> list[tuple[int, int]]:
        """
        Devuelve las celdas vecinas a distancia 2 desde (fila, columna).

        ¿Por qué distancia 2 y no 1?
        En la matriz, las celdas reales (donde puede haber camino) están en
        posiciones impares: (1,1), (1,3), (1,5)...
        Las posiciones pares son paredes que separan esas celdas.
        Entonces para ir de una celda a la siguiente hay que saltar 2 posiciones,
        pasando por encima de la pared intermedia.

        Ejemplo visual en una fila:
          col:  0   1   2   3   4
               [█] [·] [█] [·] [█]
                    ↑       ↑
                celda1   celda2  → distancia 2 entre ellas

        Los 4 desplazamientos posibles son: arriba, abajo, izquierda, derecha.
        Solo se agregan vecinos que queden dentro del borde interior de la matriz
        (sin contar la primera y última fila/columna, que son paredes externas fijas).
        """
        vecinos = []
        for delta_fila, delta_columna in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            fila_vecina    = fila    + delta_fila
            columna_vecina = columna + delta_columna
            if 1 <= fila_vecina < self.tamano - 1 and 1 <= columna_vecina < self.tamano - 1:
                vecinos.append((fila_vecina, columna_vecina))
        return vecinos
    def _eliminar_pared(self, fila1: int, columna1: int, fila2: int, columna2: int) -> None:
        """
        Elimina la pared que separa dos celdas adyacentes (a distancia 2).

        Como las celdas siempre están a distancia 2, la pared que las separa
        está exactamente en el punto medio entre ambas. Se calcula con:
            fila_pared    = (fila1    + fila2)    // 2
            columna_pared = (columna1 + columna2) // 2

        Ejemplo: celda A en (1,1) y celda B en (1,3)
            punto medio → (1, 2), que es la pared entre ellas
            se cambia ese valor de 0 (PARED) a 1 (CAMINO), abriéndola.

        Visualmente antes y después:
            Antes:  [A=·] [█] [B=·]
            Después:[A=·] [·] [B=·]  ← pared eliminada, ahora son camino continuo
        """
        fila_pared    = (fila1    + fila2)    // 2
        columna_pared = (columna1 + columna2) // 2
        self.matriz[fila_pared][columna_pared] = self.CAMINO

    def _generar(self) -> None:
            """
            Genera el laberinto usando el algoritmo Aldous-Broder.

            ¿Cómo funciona?
            1. Se listan todas las celdas válidas (posiciones impares de la matriz).
            2. Se empieza desde la celda de inicio (1,1) y se marca como visitada.
            3. Se elige un vecino al azar de la celda actual.
               - Si ese vecino NO ha sido visitado: se abre la pared entre ambos
                 (se llama a _eliminar_pared) y se marca el vecino como visitado.
               - Si ya fue visitado: no se hace nada, solo se mueve ahí.
            4. La celda actual pasa a ser ese vecino elegido.
            5. Se repite hasta que TODAS las celdas hayan sido visitadas.

            ¿Por qué funciona?
            Al visitar cada celda nueva por primera vez, siempre se conecta desde
            alguna celda ya visitada. Esto garantiza que el laberinto sea "perfecto":
            existe exactamente un camino entre cualquier par de celdas, sin bucles
            ni celdas aisladas.

            La aleatoriedad del paseo hace que cada laberinto generado sea único.
            """
            # Recolectar todas las celdas válidas (índices impares dentro del borde)
            todas_las_celdas = [
                (fila, columna)
                for fila in range(1, self.tamano - 1, 2) #inicio,fin, avanza de a 2
                for columna in range(1, self.tamano - 1, 2)
            ]
            total_celdas = len(todas_las_celdas)
            visitadas = set()

            # Empezar desde la celda de inicio
            celda_actual = self.inicio
            self.matriz[celda_actual[0]][celda_actual[1]] = self.CAMINO
            visitadas.add(celda_actual)
            while len(visitadas) < total_celdas:
                vecinos = self._vecinos_de_celda(*celda_actual) #correcto, desempaca Ej: (3, 5) → fila=3, columna=5
                celda_siguiente = random.choice(vecinos)

                if celda_siguiente not in visitadas:
                    # Conectar celda_actual → celda_siguiente abriendo la pared entre ellas
                    self.matriz[celda_siguiente[0]][celda_siguiente[1]] = self.CAMINO
                    self._eliminar_pared(celda_actual[0], celda_actual[1],
                                         celda_siguiente[0], celda_siguiente[1])
                    visitadas.add(celda_siguiente)

                celda_actual = celda_siguiente

            # Marcar inicio y meta con sus valores especiales
            self.matriz[self.inicio[0]][self.inicio[1]] = self.INICIO
            self.matriz[self.meta[0]][self.meta[1]] = self.META

    def __str__(self) -> str:
        """
        Convierte la matriz en texto para imprimirla en consola.

        Recorre cada celda de la matriz y la reemplaza por un símbolo visual:
          0 (PARED)  → █  bloque sólido
          1 (CAMINO) → espacio en blanco
          2 (INICIO) → S
          3 (META)   → G

        Python llama a este metodo automáticamente cuando se hace print(laberinto).
        """
        simbolos = {self.PARED: "█", self.CAMINO: " ", self.INICIO: "S", self.META: "G"}
        filas = []
        for fila in self.matriz:
            filas.append("".join(simbolos[int(celda)] for celda in fila))
        return "\n".join(filas)
    # ------------------------------------------------------------------ #
    #  Visualización con matplotlib                                       #
    # ------------------------------------------------------------------ #
    def mostrar(self, titulo: str = "Laberinto — Aldous-Broder") -> None:
        """
        Muestra el laberinto como imagen en una ventana usando matplotlib.

        Pasos:
        1. Se define una paleta de 4 colores, uno por cada valor de celda
           (PARED, CAMINO, INICIO, META).
        2. Se crea la figura con fondo oscuro.
        3. imshow() dibuja la matriz como una imagen donde cada celda es
           un píxel coloreado según su valor.
        4. Se añaden las letras S y G encima de las celdas de inicio y meta.
        5. Se agrega una leyenda explicando los colores.
        6. plt.show() abre la ventana con la imagen.
        """
        # Paleta: pared, camino, inicio, meta
        colores = ["#e8e8e0", "#3ded97", "#00b4d8", "#00b4d8"]
        mapa_colores = ListedColormap(colores)

        figura, ejes = plt.subplots(figsize=(8, 8))
        figura.patch.set_facecolor("#0d0d1a")
        ejes.set_facecolor("#0d0d1a")

        ejes.imshow(
            self.matriz,
            cmap=mapa_colores,
            vmin=0, vmax=3,
            interpolation="nearest",  # sin suavizado, celdas nítidas
            aspect="equal",  # celdas cuadradas, no rectangulares
        )

        # Anotaciones en inicio y meta
        fila_inicio, columna_inicio = self.inicio
        fila_meta, columna_meta = self.meta
        ejes.text(columna_inicio, fila_inicio, "S", color="white", fontsize=11,
                  fontweight="bold", ha="center", va="center")
        ejes.text(columna_meta, fila_meta, "G", color="white", fontsize=11,
                  fontweight="bold", ha="center", va="center")

        # Leyenda
        elementos_leyenda = [
            mpatches.Patch(facecolor="#3ded97", edgecolor="gray", label="Pared"),
            mpatches.Patch(facecolor="#e8e8e0", edgecolor="gray", label="Camino"),
            mpatches.Patch(facecolor="#00b4d8", edgecolor="gray", label="Inicio (S)"),
            mpatches.Patch(facecolor="#00b4d8", edgecolor="gray", label="Meta (G)"),
        ]
        ejes.legend(
            handles=elementos_leyenda,
            loc="upper right",
            fontsize=9,
            framealpha=0.4,
            facecolor="#1a1a2e",
            labelcolor="white",
        )

        ejes.set_title(titulo, color="white", fontsize=14, pad=12, fontweight="bold")
        ejes.axis("off")  # oculta los ejes numéricos, solo se ve el laberinto
        plt.tight_layout()
        plt.show()
    # ------------------------------------------------------------------ #
    #  Guardar imagen                                                     #
    # ------------------------------------------------------------------ #
    def guardar(self, ruta: str = "laberinto.png", resolucion: int = 150) -> None:
        """
        Guarda el laberinto como imagen PNG en disco, sin abrir ventana.

        Es igual a mostrar() pero en lugar de plt.show() usa fig.savefig(),
        que escribe el archivo directamente.

        Args:
            ruta:       Nombre o ruta del archivo de salida. Por defecto 'laberinto.png'.
            resolucion: Puntos por pulgada (DPI). Mayor valor = imagen más grande y nítida.
        """
        colores = ["#1a1a2e", "#e8e8e0", "#00b4d8", "#f72585"]
        mapa_colores = ListedColormap(colores)

        figura, ejes = plt.subplots(figsize=(8, 8))
        figura.patch.set_facecolor("#0d0d1a")
        ejes.set_facecolor("#0d0d1a")

        ejes.imshow(self.matriz, cmap=mapa_colores, vmin=0, vmax=3,
                    interpolation="nearest", aspect="equal")

        fila_inicio, columna_inicio = self.inicio
        fila_meta, columna_meta = self.meta
        ejes.text(columna_inicio, fila_inicio, "S", color="white", fontsize=11,
                  fontweight="bold", ha="center", va="center")
        ejes.text(columna_meta, fila_meta, "G", color="white", fontsize=11,
                  fontweight="bold", ha="center", va="center")

        ejes.axis("off")
        plt.tight_layout()
        figura.savefig(ruta, dpi=resolucion, bbox_inches="tight",
                       facecolor=figura.get_facecolor())
        plt.close(figura)  # libera memoria, importante si se generan muchos laberintos
        print(f"Laberinto guardado en: {ruta}")
    def mostrar_matriz(self) -> None:
        """Imprime los valores numéricos de la matriz en consola."""
        for fila in self.matriz:
            print(" ".join(str(int(celda)) for celda in fila))
if __name__ == "__main__":
    # Cambiar 'tamanio' para ajustar el laberinto (debe ser impar)
    laberinto = Laberinto(tamano=21)

    #print("=== Representación en consola ===")
    #print(laberinto)
    #print(f"\nTamaño : {laberinto.tamano}x{laberinto.tamano}")
    #print(f"Inicio : {laberinto.inicio}")
    #print(f"Meta   : {laberinto.meta}")

    # Mostrar en ventana
    laberinto.mostrar_matriz()
    laberinto.mostrar()

    # Guardar imagen (opcional)
    #laberinto.guardar("mi_laberinto.png")