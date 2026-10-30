# Tarea 1 Deep Learning: Clasificación de datos astronómicos del SDSS mediante una DNN
  Integrante:
  * Jorge Gallegos Venegas
    
## Stack utilizado
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=flat)](https://www.python.org/)

Repositorio para la Tarea 1 del curso de Deep Learning . Consiste en la implementación y evaluación de un conjunto de diferentes modelos de Redes Neuronales Profundas (DNN) para datos tabulares, analizando el impacto del ajuste de hiperparámetros, arquitecturas y técnicas de regularización en un problema de clasificación multiclase.

## Descripción del Problema y Dataset

El proyecto aborda la clasificación supervisada de objetos astronómicos en tres categorías: galaxias (`GALAXY`), estrellas (`STAR`) y cuásares (`QSO`)[cite: 4, 5].

* **Fuente de datos:** Registros tabulares provenientes del Sloan Digital Sky Survey (SDSS), superando los 3 millones de observaciones.
* **Características:** Conjunto de variables numéricas que abarcan coordenadas espaciales, magnitudes fotométricas en distintas bandas de luz, corrimiento al rojo (redshift), mediciones en el espectro infrarrojo y métricas de calidad de observación.
* **Preprocesamiento y Análisis:** Análisis exploratorio mediante matrices de correlación para evaluar la relación entre variables, detectar multicolinealidad o redundancias y seleccionar los atributos idóneos para el entrenamiento.

## Trabajo Realizado

### 1. Diseño e Implementación Manual de DNNs
Se diseñaron y construyeron múltiples arquitecturas secuenciales de forma manual para encontrar un modelo que se ajustara adecuadamente al problema:

* **Variación de arquitectura:** Evaluación de modelos con distinta profundidad y capacidad, variando desde arquitecturas livianas de 4 capas ocultas hasta redes más profundas de 7 capas.
* **Técnicas de regularización:** Experimentación con técnicas para prevenir el sobreajuste, incorporando capas de Dropout (tanto con tasas fijas como diferenciadas por capa), regularización L2 en los pesos y normalización por lotes (BatchNormalization).
* **Optimizadores e hiperparámetros:** Comparación entre diferentes algoritmos de optimización (RMSprop y SGD con Momentum y aceleración Nesterov), ajustando parámetros como la tasa de aprendizaje (learning rate) y el tamaño de lote (batch size).
* **Curvas de convergencia:** Registro y visualización del comportamiento de la pérdida (loss) y la precisión (accuracy) a lo largo de las épocas de entrenamiento para validar el aprendizaje y detectar posible overfitting.

### 2. Búsqueda de Arquitectura Mediante AutoML
Como etapa comparativa, se empleó la herramienta AutoKeras para automatizar el proceso de Búsqueda de Arquitecturas de Redes Neuronales (NAS) sobre datos estructurados[cite: 1, 25]. Esto permitió evaluar las decisiones tomadas de forma automática por la herramienta (elección del optimizador, profundidad de capas y normalización) y contrastar la eficiencia y complejidad de la arquitectura encontrada por AutoML frente a los diseños elaborados manualmente.

## Fuentes de donde se obtuvo los datos
* https://skyserver.sdss.org/dr16/en/tools/search/sql.aspx
* https://zenodo.org/records/3768398



# Configuración previa

Se recomienda utilizar un entorno virtual (`.venv`) para ejecutar el proyecto de manera aislada.
## 1. Crear el entorno virtual
Abre la consola en la carpeta de trabajo seleccionada (por ejemplo, en el escritorio) y ejecuta:
```bash
python -m venv .venv
```

## 2. Activar el entorno virtual
Desde la misma carpeta, activa el entorno virtual:

```cmd
.venv\Scripts\activate
```

## 3. Instalar bibliotecas

Una vez dentro del entorno virtual activado, ejecuta los siguientes comandos para instalar las dependencias:

```bash
pip install tensorflow
pip install numpy pandas scikit-learn matplotlib
pip install keras
```

