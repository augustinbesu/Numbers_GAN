import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=1):
        super(Generator, self).__init__()
        
        # Tamaño inicial después de la primera capa
        self.init_size = 7
        
        # Capa de entrada: latent_dim -> 256*7*7
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 256 * self.init_size ** 2)
        )
        
        # Capas convolucionales mejoradas
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            
            # 7x7 -> 14x14
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Capa de refinamiento
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Capa final: 64 -> 1 canal (escala de grises)
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, channels=1):
        super(Discriminator, self).__init__()

        # Secuencia de capas convolucionales
        self.conv_blocks = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 14x14 -> 7x7
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 7x7 -> 4x4
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 4x4 -> 2x2
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        
        # Capa para determinar si la imagen es real o falsa (con sigmoid para GAN clásica)
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1),
            nn.Sigmoid()
        )
        
        # Capa para clasificación de dígitos
        self.aux_layer = nn.Sequential(
            nn.Linear(512 * 2 * 2, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )
        
        # Capa para extracción de características
        self.features_layer = nn.Sequential(
            nn.Linear(512 * 2 * 2, 256),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, img):
        features_conv = self.conv_blocks(img)
        features_flat = features_conv.view(features_conv.shape[0], -1)
        
        validity = self.adv_layer(features_flat)
        label = self.aux_layer(features_flat)
        features = self.features_layer(features_flat)
        
        return validity, label, features 