# Informe Técnico: Clasificación de Letras del Lenguaje de Señas mediante Perceptrón Multicapa (MLP)

## 1. Descripción del problema de negocio

El lenguaje de señas es el principal medio de comunicación para la comunidad sorda. Sin embargo, el acceso a servicios básicos (salud, educación, atención al cliente) sigue siendo limitado debido a la escasez de intérpretes y a la falta de herramientas tecnológicas accesibles. La automatización del reconocimiento de señas mediante técnicas de aprendizaje profundo puede contribuir a reducir estas barreras, facilitando la inclusión y la interacción en entornos cotidianos.

Este proyecto explora el uso de un Perceptrón Multicapa (MLP) para reconocer letras individuales del alfabeto de señas. Aunque las arquitecturas convolucionales son más adecuadas para tareas de visión, el MLP permite evaluar las capacidades y limitaciones de un modelo completamente conectado en un problema de clasificación de imágenes, sentando las bases para futuras implementaciones con redes más complejas.

## 2. Objetivos del proyecto

### 2.1 Objetivo general
Evaluar la viabilidad de un Perceptrón Multicapa para clasificar un conjunto de letras del lenguaje de señas visualmente similares, analizando su desempeño y limitaciones.

### 2.2 Objetivos específicos
- Seleccionar y preprocesar un subconjunto de 6 letras (A, E, M, N, S, T) del dataset Sign Language MNIST.
- Implementar y comparar diferentes arquitecturas MLP variando el número de capas y neuronas.
- Optimizar hiperparámetros (optimizador, learning rate, batch size) mediante búsqueda sistemática.
- Analizar en profundidad las confusiones entre clases, especialmente M, N y S.
- Reflexionar sobre las limitaciones del MLP y proponer mejoras (arquitecturas convolucionales, aumento de datos).

## 3. KPIs (Indicadores Clave de Desempeño)

1. **Precisión global (Accuracy):** Porcentaje de clasificaciones correctas sobre el total de muestras. Es la métrica principal del rendimiento general.
2. **F1-Score por clase:** Especialmente crítico para las clases M, N y S, donde se espera un desempeño más bajo. Permite evaluar el equilibrio entre precisión y exhaustividad.
3. **Tasa de confusión entre pares (M-S, M-N, N-A):** Análisis de los errores sistemáticos más frecuentes. Este indicador cualitativo es fundamental para entender las limitaciones del modelo y guiar futuras mejoras.

## 4. Descripción de las fuentes de datos

Se utilizó el dataset **Sign Language MNIST** disponible en Kaggle (`datamunge/sign-language-mnist`). Este dataset contiene imágenes en escala de grises de 28x28 píxeles de manos realizando señas correspondientes a letras del alfabeto americano. El conjunto de entrenamiento cuenta con 27,455 muestras y el de prueba con 7,172 muestras, distribuidas en 24 clases (letras, excluyendo J y Z por su movimiento).

### Variante del proyecto
En lugar de trabajar con la totalidad de las 24 clases, se seleccionó un subconjunto de **6 letras con alta similitud visual**: A, E, M, N, S y T. Esta elección obedece a que:
- Sus configuraciones de dedos son fácilmente confundibles (ej. M, N y S utilizan el puño con variaciones de los dedos).
- Su clasificación representa un desafío significativo para un MLP, al no poder capturar relaciones espaciales sutiles.
- Evaluar el desempeño en este subconjunto permite diagnosticar las limitaciones del MLP y justificar la necesidad de arquitecturas más avanzadas.

## 5. Preparación y análisis exploratorio de datos (EDA)

### 5.1 Carga y filtrado
Se descargó el dataset mediante la librería `kagglehub` y se cargaron los archivos CSV de entrenamiento y prueba. Las etiquetas originales (0-24) se filtraron para conservar únicamente las clases seleccionadas (0, 4, 12, 13, 18, 19), y se re-mapearon a un rango continuo (0-5) para su uso en la red neuronal.

### 5.2 Análisis exploratorio
- **Distribución de clases:** Se visualizó la frecuencia de cada clase en el conjunto de entrenamiento, observando un balance razonable (entre 950 y 1,200 muestras por clase), lo que reduce el riesgo de sesgo.
- **Visualización de imágenes:** Se mostraron ejemplos representativos de cada letra, evidenciando variabilidad en postura, iluminación y posición de la mano. Estas diferencias aumentan la dificultad del problema.
- **Calidad de los datos:** No se detectaron valores nulos o corruptos; los píxeles están en el rango 0-255.

### 5.3 Preprocesamiento
- **Normalización:** Los valores de píxeles se dividieron por 255 para escalarlos al rango [0,1], acelerando la convergencia del gradiente.
- **Codificación one-hot:** Las etiquetas se transformaron a vectores de 6 dimensiones mediante `to_categorical` para usar la pérdida `categorical_crossentropy`.
- **Partición train/validation:** Se dividió el conjunto de entrenamiento en 80% entrenamiento y 20% validación, manteniendo la estratificación por clase (`stratify`) y fijando una semilla (`random_state=2026`) para garantizar reproducibilidad.

## 6. Metodología (CRISP-DM)

Se siguió el flujo de trabajo CRISP-DM adaptado a un proyecto de machine learning:

1. **Comprensión del negocio:** Definición del problema de accesibilidad y los objetivos.
2. **Comprensión de los datos:** Análisis del dataset Sign Language MNIST, selección de la variante.
3. **Preparación de los datos:** Filtrado, normalización, codificación y partición.
4. **Modelado:** Implementación de arquitecturas MLP, optimización de hiperparámetros.
5. **Evaluación:** Cálculo de métricas, análisis de matriz de confusión y ejemplos de error.
6. **Despliegue (futuro):** No se implementó en esta etapa, pero se proponen mejoras.

## 7. Implementación del Perceptrón Multicapa

Se experimentó con varias arquitecturas, desde una capa oculta simple hasta redes más profundas. La mejor configuración se obtuvo con una **arquitectura de una sola capa oculta**:

- **Capa de entrada:** 784 neuronas (píxeles aplanados).
- **Capa oculta:** 32 neuronas con activación **ReLU**.
- **Capa de salida:** 6 neuronas con activación **Softmax**.
- **Función de pérdida:** `categorical_crossentropy`.
- **Optimizador:** **Adam** con learning rate = 0.001.
- **Batch size:** 32.
- **Épocas:** 100 con **Early Stopping** (paciencia=20, restaurando los mejores pesos).

**Justificación:** La capa oculta con 32 neuronas permite capturar combinaciones lineales de los píxeles sin sobreajustar. ReLU evita el desvanecimiento del gradiente. Adam ajusta dinámicamente la tasa de aprendizaje, y el Early Stopping previene el sobreajuste.

Se realizó una búsqueda sistemática de hiperparámetros (Grid Search) variando optimizadores (Adam, RMSprop, SGD), learning rates (0.001, 0.0005, 0.0001) y batch sizes (32, 64, 128), confirmando que la configuración ganadora era la mencionada.

## 8. Entrenamiento y validación

El modelo se entrenó durante 55 épocas (detenido por Early Stopping). Las curvas de pérdida y precisión muestran:

- **Entrenamiento:** La pérdida disminuyó de 1.54 a valores cercanos a cero, y la precisión alcanzó el 100%.
- **Validación:** La pérdida de validación descendió de manera consistente hasta estabilizarse, con precisión final del 100% en validación.
- **Sobreajuste:** Aunque la precisión en validación fue perfecta, la precisión en prueba fue del 81.82%, indicando un sobreajuste moderado. El modelo memoriza ruido del conjunto de entrenamiento que no generaliza.

## 9. Evaluación del desempeño

### 9.1 Métricas globales
- **Test Accuracy:** 81.82%
- **Test Loss:** 0.9069

### 9.2 Métricas por clase

| Clase | Precisión | Recall | F1-Score | Soporte |
|-------|-----------|--------|----------|---------|
| A     | 0.89      | 1.00   | 0.94     | 331     |
| E     | 0.95      | 0.88   | 0.91     | 498     |
| M     | 0.85      | 0.77   | 0.81     | 394     |
| N     | 0.69      | 0.57   | 0.63     | 291     |
| S     | 0.55      | 0.65   | 0.59     | 246     |
| T     | 0.86      | 1.00   | 0.92     | 248     |

### 9.3 Matriz de confusión
La matriz de confusión muestra que las principales confusiones ocurren entre **M, N y S**, y también entre **N y A**. Por ejemplo, la clase M es confundida frecuentemente con S, y N con A y M. Esto confirma que el MLP tiene dificultades para distinguir patrones sutiles de posición de dedos.

## 10. Análisis de resultados y errores

### 10.1 Aciertos y errores
Se visualizaron ejemplos correcta e incorrectamente clasificados. Los aciertos corresponden a imágenes con posturas claras y buena iluminación. Los errores suelen darse en imágenes con:
- **Posturas borderline** (ej. una M con dedos separados que parece S).
- **Variaciones de iluminación o sombras** que alteran la intensidad de píxeles.
- **Manos descentradas** o con rotaciones.

### 10.2 Clases críticas
Las clases **M, N y S** presentan las mayores tasas de error debido a su similitud visual. La clase **N** también se confunde con **A** cuando el dedo pulgar no es claramente visible.

### 10.3 Limitaciones del MLP en imágenes
- **Pérdida de estructura espacial:** El aplanamiento de la imagen elimina las relaciones de vecindad entre píxeles, impidiendo que el modelo aprenda formas y bordes.
- **No invarianza:** El MLP no es invariante a traslaciones, rotaciones o escalas; pequeñas variaciones en la posición de la mano alteran drásticamente la entrada.
- **Sobreajuste:** La alta capacidad del modelo (incluso con una sola capa) tiende a memorizar el ruido de entrenamiento, reduciendo la capacidad de generalización.

## 11. Conclusiones

### 11.1 Desempeño alcanzado
El modelo MLP optimizado logró un **81.82% de accuracy** en el conjunto de prueba, demostrando que puede distinguir adecuadamente las clases con características muy diferenciadas (A, E, T). Sin embargo, su rendimiento en clases visualmente similares (M, N, S) es insuficiente, evidenciando las limitaciones del MLP para tareas de visión.

### 11.2 Decisiones de diseño clave
- La arquitectura simple (una capa oculta de 32 neuronas) superó a modelos más profundos, confirmando que el sobreajuste es un problema crítico.
- El optimizador Adam con learning rate 0.001 y batch size 32 proporcionó la mejor convergencia.
- El Early Stopping fue esencial para evitar el sobreajuste extremo.

### 11.3 Fortalezas y limitaciones
- **Fortalezas:** Simplicidad, bajo costo computacional, capacidad de aprender patrones globales de intensidad.
- **Limitaciones:** Falta de invarianza espacial, incapacidad para aprender características locales y tendencia al sobreajuste.

### 11.4 Mejoras propuestas
- **Aumento de datos:** Aplicar rotaciones, traslaciones y escalas para mejorar la generalización.
- **Regularización:** Incorporar Dropout o L2 para reducir el sobreajuste.
- **Arquitectura CNN:** En futuras iteraciones, sustituir el MLP por una red convolucional que preserve la estructura espacial y aprenda filtros de características (bordes, formas).

## 12. Referencias

- Kaggle Dataset: [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- Documentación de Keras/TensorFlow
- Materiales de la asignatura Técnicas Avanzadas de Machine Learning I
