# K-means Paralelo con OpenMP + MPI

Implementación del algoritmo de clustering **K-means** en C++ con soporte para ejecución **serial y paralela**, utilizando **OpenMP** para paralelismo en memoria compartida y **MPI** para inicialización del entorno.

El proyecto incluye un pipeline completo de:
- generación de datos,
- ejecución experimental,
- recolección de métricas,
- visualización de resultados.

---

## 🚀 Objetivo

Evaluar el impacto del paralelismo en el algoritmo K-means, midiendo:

- Tiempo de ejecución
- Speedup
- Eficiencia paralela
- Escalabilidad con el tamaño de datos

Comparando directamente contra una implementación serial.

---

## 🧠 Descripción del algoritmo

K-means es un algoritmo iterativo que:

1. Inicializa \(k\) centroides
2. Asigna cada punto al centroide más cercano (distancia euclidiana)
3. Recalcula centroides como promedios
4. Repite hasta convergencia

La versión paralela acelera:

- Asignación de puntos a clusters
- Recomputo de centroides

---

## ⚙️ Tecnologías utilizadas

| Herramienta | Uso |
|------------|-----|
| C++ | Implementación principal |
| OpenMP | Paralelización |
| MPI (OpenMPI) | Inicialización del entorno |
| Python | Visualización y generación de datos |
| NumPy / Pandas | Manejo de datos |
| Matplotlib | Gráficas |

---

## 📂 Estructura del proyecto

.
├── proyecto.cpp
├── generar_datasets_kmeans_experimento.py
├── visualizar_metricas_experimento.py
├── visualizar_resultados_kmeans.py
├── metricas_sinteticos_experimento.csv
├── figuras/
│ ├── tiempo.png
│ ├── speedup.png
│ └── eficiencia.png
└── README.md


---

## 🛠️ Compilación

Requiere OpenMP y OpenMPI instalados.

