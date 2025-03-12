import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def save_generator_image(image, path):
    """Guarda una imagen generada por el generador"""
    # Asegurar que la imagen esté en CPU
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
    
    # Crear grid de imágenes
    grid = make_grid(image, normalize=True, value_range=(-1, 1))
    
    # Convertir a numpy para matplotlib
    grid_np = grid.permute(1, 2, 0).numpy()
    
    # Guardar imagen
    plt.figure(figsize=(10, 10))
    if grid_np.shape[2] == 1:  # Si es una imagen en escala de grises
        plt.imshow(grid_np.squeeze(), cmap='gray')
    else:
        plt.imshow(grid_np)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def weights_init_normal(m):
    """Inicializa los pesos de la red con una distribución normal"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0) 