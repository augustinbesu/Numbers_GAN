# GAN MNIST: Generación de Dígitos Manuscritos con Redes Generativas Adversarias

## Introducción Teórica

Las Redes Generativas Adversarias (GANs) representan uno de los avances más significativos en el campo del aprendizaje profundo en los últimos años. Introducidas por Ian Goodfellow y colaboradores en 2014, las GANs han revolucionado nuestra capacidad para generar datos sintéticos que imitan distribuciones de datos reales.

### Fundamentos de las GANs

Una GAN consiste en dos redes neuronales que compiten entre sí en un juego de suma cero:

1. **Generador (G)**: Aprende a crear datos sintéticos que parezcan reales.
2. **Discriminador (D)**: Aprende a distinguir entre datos reales y datos generados.

Estas redes se entrenan simultáneamente, donde:
- El generador intenta maximizar la probabilidad de que el discriminador cometa un error.
- El discriminador intenta minimizar su error de clasificación.

Matemáticamente, este juego se representa como un problema de minimización-maximización:

min_G max_D V(D, G) = E_{x ~ p_data(x)}[log D(x)] + E_{z ~ p_z(z)}[log(1 - D(G(z)))]

Donde:
- p_data(x) es la distribución de datos reales
- p_z(z) es la distribución del espacio latente (típicamente ruido aleatorio)
- G(z) es la salida del generador para una entrada z
- D(x) es la probabilidad que asigna el discriminador de que x sea real

### Modelo Híbrido: GAN con Elementos de WGAN-GP

Nuestro modelo implementa un enfoque híbrido que combina elementos de una GAN tradicional con técnicas de Wasserstein GAN con Penalización de Gradiente (WGAN-GP). Esta combinación aprovecha lo mejor de ambos enfoques:

- **De la GAN tradicional**: Utilizamos la función de pérdida BCE (Binary Cross Entropy) para la discriminación básica entre imágenes reales y falsas, lo que proporciona señales de entrenamiento claras.

- **De la WGAN-GP**: Incorporamos la penalización de gradiente para estabilizar el entrenamiento y evitar el colapso del modo. Esta técnica restringe la norma del gradiente del discriminador, lo que mejora la convergencia.

La principal diferencia entre una GAN tradicional y una WGAN es que esta última utiliza la distancia de Wasserstein (también conocida como "Earth Mover's Distance") como medida de similitud entre distribuciones, en lugar de la divergencia de Jensen-Shannon implícita en la GAN original. La WGAN-GP añade una penalización de gradiente para imponer la condición de Lipschitz en el discriminador (llamado "crítico" en WGAN).

Aunque no implementamos completamente la función de pérdida de Wasserstein, la incorporación de la penalización de gradiente y otras técnicas de estabilización nos permite obtener muchos de los beneficios de la WGAN-GP mientras mantenemos la simplicidad de entrenamiento de la GAN tradicional.

### GANs para Generación de Dígitos MNIST

En este proyecto, implementamos una GAN avanzada para generar imágenes de dígitos manuscritos similares a los del conjunto de datos MNIST. Nuestra implementación incluye varias mejoras sobre la GAN básica:

1. **Arquitectura Convolucional**: Utilizamos capas convolucionales tanto en el generador como en el discriminador para capturar mejor las estructuras espaciales.

2. **Clasificación Auxiliar**: Además de distinguir entre imágenes reales y falsas, nuestro discriminador también clasifica los dígitos (0-9), lo que mejora la calidad y diversidad de las imágenes generadas.

3. **Técnicas de Estabilización**: Implementamos varias técnicas para mejorar la estabilidad del entrenamiento, incluyendo penalización de gradiente, suavizado de etiquetas y entrenamiento asimétrico.

## Arquitectura del Modelo

### Generador

El generador transforma un vector de ruido aleatorio (espacio latente) en una imagen de 28×28 píxeles. Su arquitectura está diseñada para aumentar progresivamente la resolución espacial mientras refina los detalles de la imagen.

#### Estructura del Generador

Entrada: Vector de ruido z (dimensión 100)
↓
Capa Lineal → 256×7×7
↓
Normalización por Lotes
↓
ConvTranspuesta (7×7 → 14×14) + LeakyReLU
↓
Normalización por Lotes
↓
ConvTranspuesta (14×14 → 28×28) + LeakyReLU
↓
Normalización por Lotes
↓
Capas de Refinamiento (Conv 3×3)
↓
Capa Final (Conv 3×3) + Tanh
↓
Salida: Imagen 28×28×1

#### Explicación Detallada del Generador

El generador sigue un proceso de generación de "abajo hacia arriba":

1. **Vector de Ruido (z)**: El proceso comienza con un vector aleatorio de 100 dimensiones muestreado de una distribución normal. Este vector actúa como la "semilla" para la generación.

2. **Proyección Inicial**: Este vector se proyecta a un espacio de mayor dimensión (256×7×7 = 12,544 dimensiones) mediante una capa lineal completamente conectada.

3. **Normalización por Lotes**: La salida se normaliza para estabilizar el entrenamiento, ajustando la media y varianza de las activaciones.

4. **Upsampling con ConvTranspuesta**: Utilizamos convoluciones transpuestas para aumentar la resolución espacial de 7×7 a 14×14, y luego a 28×28, duplicando el tamaño en cada paso.

5. **Activaciones LeakyReLU**: Estas funciones de activación permiten un pequeño gradiente para entradas negativas, evitando el problema de "neuronas muertas" que puede ocurrir con ReLU estándar.

6. **Refinamiento**: Aplicamos convoluciones adicionales para refinar los detalles de la imagen, mejorando la calidad visual.

7. **Capa Final con Tanh**: Una convolución final seguida de una activación Tanh normaliza los valores de píxeles al rango [-1, 1], que es el rango esperado por el discriminador.

### Discriminador

El discriminador analiza una imagen y produce tres salidas diferentes, cada una con un propósito específico:

1. **Validez**: Probabilidad de que la imagen sea real (0-1)
2. **Clasificación**: Distribución de probabilidad sobre los 10 dígitos
3. **Características**: Representación de alto nivel para feature matching

#### Estructura del Discriminador

Entrada: Imagen 28×28×1
↓
Conv (28×28 → 14×14) + LeakyReLU + Dropout
↓
Conv (14×14 → 7×7) + BatchNorm + LeakyReLU + Dropout
↓
Conv (7×7 → 4×4) + BatchNorm + LeakyReLU + Dropout
↓
Conv (4×4 → 2×2) + BatchNorm + LeakyReLU + Dropout
↓
Aplanar
↓
[Rama de Validez]        [Rama de Clasificación]        [Rama de Características]
Linear → 1               Linear → 256                   Linear → 256
Sigmoid                  LeakyReLU + Dropout           LeakyReLU
                         Linear → 10
                         Softmax
↓                        ↓                              ↓
Validez (0-1)            Clasificación (10 clases)      Características (256-dim)

#### Explicación Detallada del Discriminador

El discriminador procesa la imagen a través de una serie de capas convolucionales y luego se divide en tres ramas especializadas:

1. **Extracción de Características Convolucionales**: Una serie de capas convolucionales con stride 2 reducen progresivamente la resolución espacial mientras aumentan el número de canales de características. Esto permite al modelo capturar patrones cada vez más abstractos.

2. **Regularización con Dropout**: Aplicamos dropout después de cada capa convolucional para prevenir el sobreajuste y mejorar la generalización.

3. **Aplanamiento**: Después de las convoluciones, las características espaciales se aplanan a un vector unidimensional.

4. **Múltiples Cabezas de Salida**: El discriminador tiene tres ramas de salida diferentes:
   - **Rama de Validez**: Determina si la imagen es real o falsa con una sola neurona y activación sigmoid.
   - **Rama de Clasificación**: Clasifica el dígito en una de las 10 clases (0-9) utilizando softmax.
   - **Rama de Características**: Extrae una representación de alto nivel para feature matching, que se utiliza en la función de pérdida del generador.

Esta arquitectura de múltiples tareas permite que el discriminador no solo distinga entre imágenes reales y falsas, sino que también aprenda características útiles para la clasificación de dígitos, lo que indirectamente mejora la calidad de las imágenes generadas.

## Proceso de Entrenamiento

El entrenamiento de una GAN es un proceso delicado que requiere equilibrar el aprendizaje del generador y el discriminador. Nuestro enfoque incorpora varias técnicas avanzadas para lograr un entrenamiento estable y efectivo.

### Hiperparámetros

Los hiperparámetros clave que controlan el comportamiento del modelo incluyen:

- **Dimensión Latente**: 100 (tamaño del vector de ruido de entrada)
- **Tasa de Aprendizaje**: 0.0001 (Generador), 0.0002 (Discriminador)
- **Tamaño de Lote**: 128
- **Épocas**: 100-200
- **Optimizador**: Adam (β₁=0.5, β₂=0.999)
- **Lambda GP**: 10 (peso para la penalización de gradiente)
- **N_Critic**: 2 (número de actualizaciones del discriminador por cada actualización del generador)

### Funciones de Pérdida

Nuestro modelo utiliza múltiples funciones de pérdida, cada una diseñada para abordar un aspecto específico del entrenamiento:

#### Pérdida Adversarial
La pérdida básica de GAN que mide qué tan bien el discriminador distingue entre imágenes reales y falsas, y qué tan bien el generador engaña al discriminador.

```python
adversarial_loss = nn.BCELoss()
d_real_loss = adversarial_loss(real_pred, real_label)
d_fake_loss = adversarial_loss(fake_pred, fake_label)
g_adv_loss = adversarial_loss(validity, real_label)
```

#### Pérdida Auxiliar
Mide qué tan bien el discriminador clasifica los dígitos en las imágenes reales.

```python
auxiliary_loss = nn.CrossEntropyLoss()
d_aux_loss = auxiliary_loss(real_aux, labels)
```

#### Penalización de Gradiente
Estabiliza el entrenamiento imponiendo la condición de Lipschitz en el discriminador, siguiendo el enfoque WGAN-GP.

```python
gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, gen_imgs.detach())
```

#### Feature Matching
Alinea las características de alto nivel de las imágenes generadas con las de las imágenes reales, lo que ayuda al generador a producir imágenes más realistas.

```python
g_feature_loss = F.mse_loss(fake_features.mean(0), real_features.detach().mean(0))
```

#### Pérdida de Entropía
Fomenta la diversidad en las imágenes generadas al maximizar la entropía de las predicciones de clase.

```python
entropy_loss = -torch.mean(torch.sum(F.log_softmax(pred_label, dim=1) * F.softmax(pred_label, dim=1), dim=1))
```

### Algoritmo de Entrenamiento

El proceso de entrenamiento sigue un patrón iterativo donde alternamos entre actualizar el discriminador y el generador:

#### Entrenamiento del Discriminador
En cada iteración, el discriminador se actualiza N_Critic veces (típicamente 2) para cada actualización del generador. Esto ayuda a mantener un discriminador "fuerte" que proporcione señales útiles al generador.

1. Procesar un lote de imágenes reales y calcular la pérdida real y auxiliar.
2. Generar imágenes falsas y calcular la pérdida falsa.
3. Calcular la penalización de gradiente para estabilizar el entrenamiento.
4. Combinar todas las pérdidas y actualizar los pesos del discriminador.

#### Entrenamiento del Generador
Después de actualizar el discriminador, actualizamos el generador una vez:

1. Generar un nuevo lote de imágenes falsas.
2. Calcular la pérdida adversarial (qué tan bien engañan al discriminador).
3. Calcular la pérdida de feature matching y entropía.
4. Combinar las pérdidas y actualizar los pesos del generador.

### Técnicas de Estabilización

El entrenamiento de GANs puede ser inestable, por lo que implementamos varias técnicas para mejorar la convergencia:

#### Suavizado de Etiquetas
En lugar de usar etiquetas binarias duras (0 y 1), utilizamos etiquetas suavizadas (por ejemplo, 0.9 en lugar de 1.0 para imágenes reales) para prevenir que el discriminador se vuelva demasiado confiado.

```python
real_label = 0.9 + 0.1 * torch.rand(current_batch_size, 1, device=device)  # Entre 0.9 y 1.0
fake_label = 0.0 + 0.1 * torch.rand(current_batch_size, 1, device=device)  # Entre 0.0 y 0.1
```

#### Entrenamiento Asimétrico
Entrenar el discriminador más veces que el generador ayuda a mantener el equilibrio entre ambos modelos.

```python
for _ in range(N_CRITIC):
    # Entrenar discriminador
# Entrenar generador una vez
```

#### Penalización de Gradiente
Esta técnica de WGAN-GP penaliza al discriminador si el gradiente se desvía de la norma 1, lo que estabiliza el entrenamiento.

```python
def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    # Interpolación entre muestras reales y falsas
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    # Calcular gradientes
    d_interpolates, _, _ = discriminator(interpolates)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                   grad_outputs=torch.ones_like(d_interpolates),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    # Penalizar desviaciones de la norma 1
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty
```

#### Inicialización de Pesos
Utilizamos una inicialización específica para los pesos de las capas convolucionales y lineales que ha demostrado funcionar bien para GANs.

```python
def weights_init_normal(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
```

## Evaluación y Visualización

Para evaluar el rendimiento de nuestro modelo, utilizamos diversas métricas y visualizaciones que proporcionan información sobre diferentes aspectos del proceso de generación y clasificación.

### Métricas de Entrenamiento

Durante el entrenamiento, monitoreamos varias métricas clave:

#### Pérdidas
- **Pérdida del Generador**: Indica qué tan bien el generador está engañando al discriminador.
- **Pérdida del Discriminador**: Muestra qué tan bien el discriminador distingue entre imágenes reales y falsas.

#### Precisiones
- **Precisión en Imágenes Reales**: Porcentaje de imágenes reales correctamente identificadas como reales.
- **Precisión en Imágenes Falsas**: Porcentaje de imágenes generadas correctamente identificadas como falsas.
- **Precisión de Clasificación**: Qué tan bien el discriminador clasifica los dígitos en las imágenes reales.

Estas métricas se visualizan en gráficas que muestran su evolución a lo largo del entrenamiento, permitiendo identificar problemas como el desvanecimiento de gradientes o el desequilibrio entre generador y discriminador.

### Visualización de Imágenes Generadas

Periódicamente guardamos imágenes generadas para evaluar visualmente la calidad y diversidad de la salida del generador. Esto incluye:

- **Muestras Individuales**: Imágenes generadas a partir de vectores de ruido aleatorio.
- **Grids de Imágenes**: Matrices de imágenes que muestran la variedad de dígitos generados.
- **Interpolaciones**: Transiciones suaves entre diferentes puntos del espacio latente.

### Matriz de Confusión

Para evaluar la capacidad de clasificación del discriminador, generamos una matriz de confusión que muestra:

- Qué dígitos se clasifican correctamente con mayor frecuencia.
- Qué pares de dígitos tienden a confundirse entre sí.
- Patrones sistemáticos en los errores de clasificación.

Esta visualización es particularmente útil para entender las fortalezas y debilidades del modelo en términos de clasificación.

### Visualización t-SNE

Para explorar la estructura del espacio latente y las representaciones aprendidas, utilizamos t-SNE (t-Distributed Stochastic Neighbor Embedding) para proyectar:

- Características de imágenes reales extraídas por el discriminador.
- Características de imágenes generadas extraídas por el discriminador.

Esta visualización permite ver cómo se agrupan las diferentes clases de dígitos y cómo se comparan las distribuciones de características reales y generadas.

## Aplicación de Evaluación

Hemos desarrollado una aplicación interactiva que permite evaluar visualmente el rendimiento del modelo. La aplicación:

1. Muestra imágenes reales del dataset MNIST o generadas por el modelo.
2. Permite al usuario ver la predicción del discriminador (real/falsa).
3. Muestra la clasificación del dígito realizada por el discriminador.
4. Proporciona estadísticas sobre la precisión del discriminador.

Esta herramienta facilita la evaluación cualitativa del modelo y ayuda a entender intuitivamente su comportamiento.

## Espacio Latente y Generación

El espacio latente de dimensión 100 funciona como el "espacio creativo" del generador. Cada punto en este espacio corresponde a una imagen generada, y la estructura de este espacio tiene propiedades interesantes:

### Interpolación en el Espacio Latente

Podemos generar transiciones suaves entre diferentes dígitos interpolando entre puntos en el espacio latente:

```python
# Interpolación lineal entre z1 y z2
alpha = 0.5  # Factor de interpolación
z_interp = alpha * z1 + (1 - alpha) * z2
img_interp = generator(z_interp)
```

### Aritmética Vectorial

Es posible realizar operaciones vectoriales en el espacio latente para manipular características específicas:

```python
# Si z_7 genera un 7 y z_1 genera un 1, podemos intentar:
z_new = z_7 - z_1 + z_9  # Intentar convertir un 9 en algo parecido a un 7
```

### Clustering Natural

Los puntos cercanos en el espacio latente tienden a generar imágenes similares, lo que sugiere que el modelo ha aprendido una representación estructurada del espacio de dígitos.

## Desafíos y Soluciones

El entrenamiento de GANs presenta varios desafíos característicos, para los cuales hemos implementado soluciones específicas:

### Modo de Colapso

**Problema**: El generador produce un conjunto limitado de muestras, ignorando la diversidad del dataset real.

**Soluciones**:
- Pérdida de entropía para fomentar la diversidad en las predicciones de clase.
- Feature matching para alinear las distribuciones de características reales y generadas.
- Penalización de gradiente para estabilizar el entrenamiento y prevenir comportamientos extremos.

### Equilibrio Generador-Discriminador

**Problema**: Si uno de los modelos se vuelve demasiado fuerte respecto al otro, el entrenamiento puede estancarse o diverger.

**Soluciones**:
- Entrenamiento asimétrico con N_CRITIC > 1 para mantener un discriminador competente.
- Suavizado de etiquetas para prevenir que el discriminador se vuelva demasiado confiado.
- Tasas de aprendizaje diferentes para el generador y el discriminador.

### Vanishing Gradients

**Problema**: Gradientes muy pequeños que dificultan el aprendizaje efectivo, especialmente en las primeras etapas del entrenamiento.

**Soluciones**:
- LeakyReLU en lugar de ReLU para permitir gradientes en la región negativa.
- Penalización de gradiente para mantener gradientes de magnitud razonable.
- Inicialización de pesos cuidadosamente diseñada para el contexto de GANs.

## Estructura del Proyecto

```
Improved_GAN/
├── config.py           # Configuración de hiperparámetros
├── data_loader.py      # Carga de datos
├── model.py            # Definición de modelos (Generator y Discriminator)
├── utils.py            # Funciones auxiliares
├── evaluation.py       # Funciones de evaluación
├── train.py            # Script de entrenamiento
├── app.py              # Aplicación de evaluación
├── data/               # Directorio para datos
├── models/             # Directorio para modelos guardados
├── images/             # Directorio para imágenes generadas
└── evaluation/         # Directorio para resultados de evaluación

```

## Flujo de Trabajo

1. **Configuración**: Ajustar hiperparámetros en `config.py` según necesidades.
2. **Entrenamiento**: Ejecutar `train.py` para entrenar los modelos y generar visualizaciones.
3. **Evaluación**: Utilizar `app.py` para evaluar interactivamente el rendimiento del modelo.
4. **Análisis**: Examinar las métricas, visualizaciones y muestras generadas para comprender el comportamiento del modelo.
