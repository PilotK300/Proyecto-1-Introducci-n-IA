# Proyecto 1 — Generador de Laberintos con IA

## Descripción

Proyecto de Introducción a la Inteligencia Artificial (Pontificia Universidad Javeriana) basado en el libro *Artificial Intelligence: A Modern Approach* (Russell & Norvig, 3ra edición).

El proyecto consiste en generar un laberinto de forma aleatoria usando una matriz, construir un grafo a partir de ella, y resolverlo usando algoritmos de búsqueda vistos en clase: BFS, DFS y A*.

---

## Estructura del proyecto

```
Proyecto/
├── Laberinto.py   # Genera y visualiza el laberinto
├── Grafo.py       # Construye el grafo a partir del laberinto
├── Buscador.py    # Algoritmos de búsqueda BFS, DFS y A*
└── README.md      # Este archivo
```

---

## Archivos

### `Laberinto.py` — Clase `Laberinto`

Genera la estructura del laberinto usando una matriz de numpy.

**Atributos principales:**
- `tamano` — tamaño de la matriz (siempre impar, mínimo 5)
- `matriz` — matriz numpy de enteros con los valores de cada celda
- `inicio` — posición de entrada, esquina superior izquierda `(1, 1)` por defecto
- `meta` — posición de la meta, centro exacto de la matriz por defecto

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

Convierte la matriz del laberinto en un grafo representado como lista de adyacencia con pesos.

**Atributos principales:**
- `laberinto` — referencia al objeto `Laberinto`
- `adyacencia` — diccionario donde cada clave es un nodo `(fila, columna)` y su valor es una lista de tuplas `(nodo_vecino, peso)`

**Estructura de la lista de adyacencia:**
```python
{
    (1, 1): [((1, 2), 1), ((2, 1), 1)],
    (1, 2): [((1, 1), 1), ((1, 3), 1)],
    ...
}
```

Cada arista tiene peso `1` porque moverse entre celdas adyacentes tiene el mismo costo.

**Métodos:**
- `_es_camino(fila, columna)` — verifica si una celda es transitable
- `_construir()` — recorre la matriz y llena el diccionario de adyacencia
- `vecinos(nodo)` — devuelve lista de `(nodo_vecino, peso)` para un nodo dado
- `cantidad_nodos()` — total de celdas transitables
- `cantidad_aristas()` — total de conexiones (sin contar duplicados)
- `mostrar_adyacencia()` — imprime un resumen del grafo en consola

---

### `Buscador.py` — Clase `Buscador` y función `main`

Implementa los tres algoritmos de búsqueda con animación paso a paso.

**Atributos principales:**
- `laberinto` — referencia al objeto `Laberinto`
- `grafo` — referencia al objeto `Grafo`
- `heuristica` — función heurística usada por A* (Manhattan por defecto)

**Valores extra para la animación:**
| Valor | Constante  | Color      | Significado |
|-------|------------|------------|-------------|
| 4     | `EXPLORADO`| Azul claro | Celda ya visitada |
| 5     | `FRONTERA` | Amarillo   | Celda en cola/pila |
| 6     | `CAMINO`   | Verde      | Camino final |

**Métodos:**
- `bfs()` — Búsqueda por Amplitud, garantiza el camino más corto
- `dfs()` — Búsqueda por Profundidad, no garantiza el camino más corto
- `a_star()` — Búsqueda Informada, usa heurística para guiarse hacia la meta
- `_heuristica(nodo, meta)` — distancia Manhattan, usada por defecto en A*
- `_reconstruir_camino(origen, nodo_final)` — reconstruye la ruta encontrada
- `_animar(pasos, camino, titulo)` — muestra la animación paso a paso
- `_celebrar(titulo, tiempo, pasos)` — ventana de celebración con tiempo de búsqueda

**Función `main`:**
```python
main(n, coordenada_salida, coordenada_meta, heuristica=None)
```
Recibe los insumos del problema y ejecuta los tres algoritmos en secuencia.

---

## Algoritmos de búsqueda

### BFS — Búsqueda por Amplitud

Usa una **cola FIFO**. Explora todos los vecinos del nodo actual antes de avanzar al siguiente nivel. Garantiza el **camino más corto** en número de pasos.

- Complejidad tiempo: O(V + E)
- Complejidad espacio: O(V)

### DFS — Búsqueda por Profundidad

Usa una **pila LIFO**. Siempre explora el camino más profundo antes de retroceder. No garantiza el camino más corto pero usa menos memoria en muchos casos.

- Complejidad tiempo: O(V + E)
- Complejidad espacio: O(V)

### A* — Búsqueda Informada

Combina el costo real acumulado `g(n)` con una estimación del costo restante `h(n)`:

```
f(n) = g(n) + h(n)
```

Siempre expande el nodo con menor `f(n)`. Es el más eficiente de los tres cuando la heurística es admisible (nunca sobreestima el costo real).

- Heurística por defecto: **distancia Manhattan** `|fila1 - fila2| + |col1 - col2|`
- Complejidad tiempo: O(E log V)
- Complejidad espacio: O(V)

**¿Por qué Manhattan y no Euclidiana?**
En el laberinto solo se puede mover en 4 direcciones (arriba, abajo, izquierda, derecha), no en diagonal. La distancia Manhattan refleja exactamente ese tipo de movimiento y nunca sobreestima, lo que la hace admisible para A*.

---

## Algoritmo de generación — Aldous-Broder

Se eligió el algoritmo **Aldous-Broder** para generar el laberinto porque garantiza **aleatoriedad uniforme**: todos los laberintos posibles de un tamaño dado tienen exactamente la misma probabilidad de ser generados. Algoritmos como DFS recursivo o Prim tienden a producir patrones reconocibles.

### ¿Cómo funciona?

1. Se listan todas las celdas válidas de la matriz (posiciones impares).
2. Se empieza desde `(1, 1)` y se marca como visitada.
3. Se elige un vecino al azar de la celda actual.
   - Si **no ha sido visitado**: se elimina la pared entre ambos y se marca como visitado.
   - Si **ya fue visitado**: solo se mueve ahí sin hacer nada.
4. Se repite hasta que todas las celdas hayan sido visitadas.

Esto garantiza un laberinto **perfecto**: existe exactamente un camino entre cualquier par de celdas, sin bucles ni celdas aisladas.

---

## Decisiones técnicas

**¿Por qué el tamaño siempre es impar?**
Para que exista una celda central exacta donde colocar la meta. Con tamaño impar, el centro `tamano // 2` siempre cae en una posición impar válida (celda, no pared).

**¿Por qué `np.ones` en lugar de `np.zeros`?**
Porque `PARED = 1`. La matriz debe iniciar completamente llena de paredes, y `np.ones` rellena con `1`. Con `np.zeros` la matriz empezaría con caminos.

**¿Por qué lista de adyacencia y no matriz de adyacencia?**
El laberinto es un grafo disperso: cada nodo tiene máximo 4 vecinos. Una matriz de adyacencia desperdiciaría memoria guardando ceros. La lista de adyacencia solo guarda las conexiones que existen.

**¿Por qué pesos en el grafo si todos valen 1?**
El requerimiento especifica "nodos, enlaces, pesos". Tener el peso explícito permite en el futuro cambiar el costo de moverse por ciertas celdas (por ejemplo, celdas con trampa que cuesten más) sin modificar los algoritmos.

**¿Por qué la heurística es un parámetro?**
Para cumplir el requerimiento de "lectura de la función heurística" y permitir pasar distintas heurísticas sin modificar el código. Por defecto usa Manhattan, pero se puede pasar cualquier función `(nodo, meta) → float`.

**¿Por qué clases separadas?**
Cada clase tiene una única responsabilidad:
- `Laberinto` — genera y representa la estructura.
- `Grafo` — modela las conexiones con pesos.
- `Buscador` — ejecuta los algoritmos sobre el grafo.

---

## Instalación y uso

```bash
pip install matplotlib numpy
```

Correr el proyecto completo:
```bash
python Buscador.py
```

Cambiar parámetros en el `__main__` de `Buscador.py`:
```python
main(
    n=21,                        # tamaño del laberinto (impar)
    coordenada_salida=None,      # None = esquina superior izquierda (1,1)
    coordenada_meta=None,        # None = centro de la matriz
)
```

Con coordenadas y heurística personalizadas:
```python
def euclidiana(nodo, meta):
    return ((nodo[0] - meta[0])**2 + (nodo[1] - meta[1])**2) ** 0.5

main(n=31, coordenada_salida=(1,1), coordenada_meta=(15,15), heuristica=euclidiana)
```