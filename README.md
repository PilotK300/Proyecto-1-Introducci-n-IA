# Proyecto-1-Introducci-n-IA

## Archivos

### `Laberinto.py` — Clase `Laberinto`

Genera la estructura del laberinto usando una matriz de numpy.

**Atributos principales:**
- `tamano` — tamaño de la matriz (siempre impar, mínimo 5)
- `matriz` — matriz numpy de enteros con los valores de cada celda
- `inicio` — posición de entrada, siempre en la esquina superior izquierda `(1, 1)`
- `meta` — posición de la meta, siempre en el centro exacto de la matriz

**Valores de celda:**
| Valor | Constante | Significado |
|-------|-----------|-------------|
| 0     | `CAMINO`  | Celda transitable |
| 1     | `PARED`   | Celda bloqueada |
| 2     | `INICIO`  | Punto de entrada |
| 3     | `META`    | Punto de llegada |

**Métodos:**
- `_vecinos_de_celda(fila, columna)` — devuelve celdas vecinas a distancia 2
- `_eliminar_pared(fila1, columna1, fila2, columna2)` — abre el camino entre dos celdas
- `_generar()` — ejecuta el algoritmo Aldous-Broder
- `mostrar_matriz()` — imprime los valores numéricos de la matriz en consola
- `mostrar()` — visualiza el laberinto en una ventana con matplotlib
- `guardar(ruta)` — guarda el laberinto como imagen PNG

---

### `Grafo.py` — Clase `Grafo`

Convierte la matriz del laberinto en un grafo representado como lista de adyacencia.

**Atributos principales:**
- `laberinto` — referencia al objeto `Laberinto`
- `adyacencia` — diccionario donde cada clave es un nodo `(fila, columna)` y su valor es la lista de nodos vecinos conectados

**Estructura de la lista de adyacencia:**
```python
{
    (1, 1): [(1, 2), (2, 1)],
    (1, 2): [(1, 1), (1, 3)],
    ...
}
```

**Métodos:**
- `_es_camino(fila, columna)` — verifica si una celda es transitable
- `_construir()` — recorre la matriz y llena el diccionario de adyacencia
- `vecinos(nodo)` — devuelve los vecinos de un nodo dado
- `cantidad_nodos()` — total de celdas transitables
- `cantidad_aristas()` — total de conexiones (sin contar duplicados)
- `mostrar_adyacencia()` — imprime un resumen del grafo en consola

---

## Algoritmo de generación — Aldous-Broder

Se eligió el algoritmo **Aldous-Broder** para generar el laberinto porque es la mejor forma de garantizar **aleatoriedad uniforme**.

### ¿Qué significa aleatoriedad uniforme?

Significa que todos los laberintos posibles de un tamaño dado tienen exactamente la misma probabilidad de ser generados. Otros algoritmos como DFS recursivo o Prim tienden a generar laberintos con patrones reconocibles (por ejemplo, DFS produce pasillos largos y sinuosos). Aldous-Broder no tiene ese sesgo.

### ¿Cómo funciona?

1. Se listan todas las celdas válidas de la matriz (posiciones impares).
2. Se empieza desde la celda de inicio `(1, 1)` y se marca como visitada.
3. Se elige un vecino al azar de la celda actual.
   - Si ese vecino **no ha sido visitado**: se elimina la pared entre ambos y se marca como visitado.
   - Si **ya fue visitado**: no se hace nada, solo se mueve ahí.
4. La celda actual pasa a ser ese vecino.
5. Se repite hasta que **todas** las celdas hayan sido visitadas.

### ¿Por qué funciona?

Cada vez que se llega a una celda no visitada por primera vez, siempre se conecta desde alguna celda ya visitada. Esto garantiza que el laberinto sea **perfecto**: existe exactamente un camino entre cualquier par de celdas, sin bucles ni celdas aisladas.

### ¿Por qué distancia 2 entre celdas?

En la matriz, las celdas reales (donde puede existir un camino) están en posiciones impares: `(1,1)`, `(1,3)`, `(1,5)`... Las posiciones pares son paredes que separan esas celdas. Por eso al moverse de una celda a otra se salta de 2 en 2, pasando por encima de la pared intermedia.

```
col:  0   1   2   3   4
     [█] [·] [█] [·] [█]
          ↑       ↑
      celda1   celda2  → distancia 2
```

---

## Decisiones técnicas

**¿Por qué el tamaño siempre es impar?**
Para que exista una celda central exacta donde colocar la meta. Con tamaño impar, el centro es `tamano // 2`, que siempre cae en una posición impar válida (celda, no pared).

**¿Por qué `np.ones` en lugar de `np.zeros`?**
Porque en este proyecto `PARED = 1`. La matriz debe iniciar completamente llena de paredes, y `np.ones` rellena con `1`. Si se usara `np.zeros`, la matriz empezaría con caminos en lugar de paredes.

**¿Por qué lista de adyacencia y no matriz de adyacencia?**
Porque el laberinto es un grafo **disperso**: cada nodo tiene máximo 4 vecinos. Una matriz de adyacencia desperdiciaría mucha memoria guardando ceros. La lista de adyacencia solo guarda las conexiones que existen.

**¿Por qué clases separadas?**
Cada clase tiene una única responsabilidad (principio de responsabilidad única):
- `Laberinto` genera y representa la estructura.
- `Grafo` modela las conexiones para los algoritmos de búsqueda.
- `Buscador` (próximo paso) ejecutará los algoritmos sobre el grafo.

---

## Instalación y uso

```bash
pip install matplotlib numpy
```

Generar y visualizar el laberinto:
```bash
python Laberinto.py
```

Construir el grafo:
```bash
python Grafo.py
```

Cambiar el tamaño del laberinto editando la última línea de cada archivo:
```python
laberinto = Laberinto(tamano=31)