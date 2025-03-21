import torch

# Configuración de hiperparámetros
LATENT_DIM = 100
LR_G = 0.0002  # Tasa de aprendizaje del generador
LR_D = 0.0002  # Tasa de aprendizaje del discriminador
BATCH_SIZE = 128
EPOCHS = 200
SAMPLE_INTERVAL = 10
# La idea de la siguiente linea es que el discriminador se entrene más veces que el generador
# para que pueda aprender a distinguir entre real y falso mejor que el generador. Esto se haría en
# principio porque el generador conseguía generar números que eran indistinguibles de los reales, según el discriminador.
# De manera que una mejora sería implementar eso para que siga habiendo juego entre el generador y el discriminador.
N_CRITIC = 1  # ¿Entrenar el discriminador más veces que el generador? No es más que modificar este parámetro.
BETA1 = 0.5  # Beta1 para Adam
BETA2 = 0.999  # Beta2 para Adam

# Configuración de dispositivo (GPU/CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rutas de directorios
IMAGES_DIR = "images"
MODELS_DIR = "models"
EVALUATION_DIR = "evaluation"
DATA_DIR = "data" 