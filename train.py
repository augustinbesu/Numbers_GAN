import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import multiprocessing

from model import Generator, Discriminator
from utils import save_generator_image, weights_init_normal
from data_loader import get_dataloader, create_directories
from evaluation import (
    generate_tsne_visualization, 
    generate_evaluation_grid, 
    plot_training_metrics,
    plot_digit_confusion_matrix
)
from config import (
    LATENT_DIM, BATCH_SIZE, EPOCHS, SAMPLE_INTERVAL,
    DEVICE, IMAGES_DIR, MODELS_DIR, EVALUATION_DIR
)

# Parámetros de la GAN clásica
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0002
BETA1 = 0.5
BETA2 = 0.999

def train():
    print(f"Usando dispositivo: {DEVICE}")
    print("Implementación clásica de GAN con mejoras")
    
    # Crear directorios
    create_directories()
    
    # Cargar el dataset MNIST
    dataloader = get_dataloader(batch_size=BATCH_SIZE)
    
    # Guardar algunas imágenes reales para referencia
    real_batch = next(iter(dataloader))
    save_generator_image(real_batch[0][:16], f"{EVALUATION_DIR}/real_samples.png")

    # Inicializar modelos
    generator = Generator(LATENT_DIM).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    # Inicializar pesos
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Optimizadores - Adam estándar para GANs
    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2))

    # Funciones de pérdida
    adversarial_loss = nn.BCELoss()
    auxiliary_loss = nn.CrossEntropyLoss()

    # Vectores para almacenar métricas
    d_losses = []
    g_losses = []
    d_real_acc = []
    d_fake_acc = []
    aux_accuracies = []
    
    # Guardar ruido fijo para seguimiento de progreso
    fixed_noise = torch.randn(16, LATENT_DIM, device=DEVICE)

    # Entrenamiento
    print("Iniciando entrenamiento GAN clásica...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_d_real_acc = 0
        epoch_d_fake_acc = 0
        epoch_aux_acc = 0
        num_batches = 0
        
        for i, (real_imgs, labels) in enumerate(dataloader):
            # Configurar tamaño de batch actual
            current_batch_size = real_imgs.size(0)
            
            # Mover datos al dispositivo
            real_imgs = real_imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Etiquetas para pérdida adversarial
            real_label = torch.ones(current_batch_size, 1, device=DEVICE)
            fake_label = torch.zeros(current_batch_size, 1, device=DEVICE)
            
            # Aplicar suavizado de etiquetas (label smoothing)
            real_label = real_label * 0.9 + 0.1 * torch.rand_like(real_label)
            fake_label = fake_label + 0.1 * torch.rand_like(fake_label)
            
            # -----------------
            # Entrenar Discriminador
            # -----------------
            optimizer_D.zero_grad()
            
            # Calcular pérdida con imágenes reales
            real_pred, real_aux, _ = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_pred, real_label)
            
            # Calcular pérdida auxiliar (clasificación de dígitos)
            d_aux_loss = auxiliary_loss(real_aux, labels)
            
            # Calcular precisión en imágenes reales
            d_real_acc_batch = (real_pred > 0.5).float().mean().item()
            
            # Calcular precisión de clasificación
            pred_labels = real_aux.max(1, keepdim=True)[1]
            aux_acc = pred_labels.eq(labels.view_as(pred_labels)).float().mean().item()
            
            # Generar imágenes falsas
            z = torch.randn(current_batch_size, LATENT_DIM, device=DEVICE)
            gen_imgs = generator(z)
            
            # Calcular pérdida con imágenes falsas
            fake_pred, _, _ = discriminator(gen_imgs.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake_label)
            
            # Calcular precisión en imágenes falsas
            d_fake_acc_batch = (fake_pred < 0.5).float().mean().item()
            
            # Pérdida total del discriminador
            d_loss = d_real_loss + d_fake_loss + 0.5 * d_aux_loss
            
            # Retropropagación
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            # Entrenar Generador
            # -----------------
            optimizer_G.zero_grad()
            
            # Generar imágenes falsas
            z = torch.randn(current_batch_size, LATENT_DIM, device=DEVICE)
            gen_imgs = generator(z)
            
            # Calcular pérdida adversarial
            fake_pred, fake_aux, _ = discriminator(gen_imgs)
            g_loss = adversarial_loss(fake_pred, real_label)
            
            # Retropropagación
            g_loss.backward()
            optimizer_G.step()
            
            # Acumular métricas
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_d_real_acc += d_real_acc_batch
            epoch_d_fake_acc += d_fake_acc_batch
            epoch_aux_acc += aux_acc
            num_batches += 1
            
            # Mostrar progreso
            if i % 100 == 0:
                print(
                    f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                    f"[D real acc: {d_real_acc_batch:.2f}] [D fake acc: {d_fake_acc_batch:.2f}] "
                    f"[Aux acc: {aux_acc:.2f}]"
                )
        
        # Calcular promedios
        epoch_d_loss /= num_batches
        epoch_g_loss /= num_batches
        epoch_d_real_acc /= num_batches
        epoch_d_fake_acc /= num_batches
        epoch_aux_acc /= num_batches
        
        # Guardar métricas
        d_losses.append(epoch_d_loss)
        g_losses.append(epoch_g_loss)
        d_real_acc.append(epoch_d_real_acc)
        d_fake_acc.append(epoch_d_fake_acc)
        aux_accuracies.append(epoch_aux_acc)
        
        # Imprimir resumen de época
        print(
            f"\n[Epoch {epoch}/{EPOCHS}] "
            f"[D loss: {epoch_d_loss:.4f}] [G loss: {epoch_g_loss:.4f}] "
            f"[D real acc: {epoch_d_real_acc:.2f}] [D fake acc: {epoch_d_fake_acc:.2f}] "
            f"[Aux acc: {epoch_aux_acc:.2f}] "
            f"[Time: {(time.time() - start_time)/60:.2f} min]\n"
        )
        
        # Guardar imágenes generadas a intervalos regulares
        if epoch % SAMPLE_INTERVAL == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                gen_imgs = generator(fixed_noise)
                save_generator_image(gen_imgs, f"{IMAGES_DIR}/epoch_{epoch}.png")
                
                # Guardar modelos en puntos clave
                if epoch % 10 == 0 or epoch == EPOCHS - 1:
                    torch.save(generator.state_dict(), f"{MODELS_DIR}/generator_epoch_{epoch}.pth")
                    torch.save(discriminator.state_dict(), f"{MODELS_DIR}/discriminator_epoch_{epoch}.pth")

    # Guardar modelos finales
    torch.save(generator.state_dict(), f"{MODELS_DIR}/generator_final.pth")
    torch.save(discriminator.state_dict(), f"{MODELS_DIR}/discriminator_final.pth")

    # Graficar métricas
    plot_training_metrics(g_losses, d_losses, d_real_acc, d_fake_acc, aux_accuracies)

    # Generar visualización t-SNE de características
    generate_tsne_visualization(generator, discriminator, dataloader)
    
    # Generar grid de imágenes para evaluación final
    generate_evaluation_grid(generator)
    
    # Generar matriz de confusión para clasificación de dígitos
    plot_digit_confusion_matrix(generator, discriminator, get_dataloader(batch_size=64, train=False))
    
    print(f"¡Entrenamiento GAN completado en {(time.time() - start_time)/60:.2f} minutos!")
    print(f"Modelos guardados en la carpeta '{MODELS_DIR}'")
    print(f"Imágenes generadas guardadas en la carpeta '{IMAGES_DIR}'")
    print(f"Métricas y visualizaciones guardadas en la carpeta '{EVALUATION_DIR}'")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    train() 