import torch

# Configuración de hiperparámetros
LATENT_DIM = 100
LR_G = 0.0001  # Tasa de aprendizaje del generador
LR_D = 0.0001  # Tasa de aprendizaje del discriminador
BATCH_SIZE = 128
EPOCHS = 25
SAMPLE_INTERVAL = 5
N_CRITIC = 1  # Entrenar el discriminador más veces que el generador
LAMBDA_GP = 10  # Peso para la penalización de gradiente (WGAN-GP)
BETA1 = 0.5  # Beta1 para Adam
BETA2 = 0.999  # Beta2 para Adam
AUX_WEIGHT = 1.0  # Peso para la pérdida auxiliar de clasificación

# Configuración de dispositivo (GPU/CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rutas de directorios
IMAGES_DIR = "images"
MODELS_DIR = "models"
EVALUATION_DIR = "evaluation"
DATA_DIR = "data" 