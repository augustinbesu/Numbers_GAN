import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
    compute_gradient_penalty,
    plot_digit_confusion_matrix
)
from config import (
    LATENT_DIM, LR_G, LR_D, BATCH_SIZE, EPOCHS, SAMPLE_INTERVAL,
    N_CRITIC, LAMBDA_GP, BETA1, BETA2, AUX_WEIGHT, DEVICE,
    IMAGES_DIR, MODELS_DIR, EVALUATION_DIR
)

def train():
    print(f"Usando dispositivo: {DEVICE}")
    
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

    # Optimizadores
    optimizer_G = optim.Adam(generator.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(BETA1, BETA2))

    # Planificadores de tasa de aprendizaje
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5, verbose=True)

    # Funciones de pérdida
    adversarial_loss = nn.BCELoss()
    auxiliary_loss = nn.CrossEntropyLoss()

    # Vectores para almacenar métricas
    G_losses = []
    D_losses = []
    D_real_acc = []
    D_fake_acc = []
    
    # Guardar ruido fijo para seguimiento de progreso
    fixed_noise = torch.randn(16, LATENT_DIM, device=DEVICE)

    # Entrenamiento
    print("Iniciando entrenamiento...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_d_real_acc = 0
        epoch_d_fake_acc = 0
        batches_done = 0
        
        for i, (real_imgs, labels) in enumerate(dataloader):
            current_batch_size = real_imgs.size(0)
            
            # Configurar etiquetas con suavizado
            real_label = 0.9 + 0.1 * torch.rand(current_batch_size, 1, device=DEVICE)
            fake_label = 0.0 + 0.1 * torch.rand(current_batch_size, 1, device=DEVICE)
            
            # Mover datos al dispositivo
            real_imgs = real_imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # -----------------
            # Entrenar Discriminador
            # -----------------
            for _ in range(N_CRITIC):
                optimizer_D.zero_grad()
                
                # Calcular pérdida con imágenes reales
                real_pred, real_aux, real_features = discriminator(real_imgs)
                d_real_loss = adversarial_loss(real_pred, real_label)
                
                # Añadir pérdida auxiliar para mejorar la clasificación
                d_aux_loss = auxiliary_loss(real_aux, labels)
                
                # Calcular precisión en imágenes reales
                d_real_acc = (real_pred > 0.5).float().mean().item()
                
                # Calcular precisión de clasificación
                pred_labels = real_aux.max(1, keepdim=True)[1]
                aux_acc = pred_labels.eq(labels.view_as(pred_labels)).float().mean().item()
                
                # Generar imágenes falsas
                z = torch.randn(current_batch_size, LATENT_DIM, device=DEVICE)
                gen_imgs = generator(z)
                
                # Calcular pérdida con imágenes falsas
                fake_pred, fake_aux, fake_features = discriminator(gen_imgs.detach())
                d_fake_loss = adversarial_loss(fake_pred, fake_label)
                
                # Calcular precisión en imágenes falsas
                d_fake_acc = (fake_pred < 0.5).float().mean().item()
                
                # Calcular penalización de gradiente (WGAN-GP)
                gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, gen_imgs.detach())
                
                # Pérdida total del discriminador (incluye pérdida auxiliar)
                d_loss = d_real_loss + d_fake_loss + AUX_WEIGHT * d_aux_loss + LAMBDA_GP * gradient_penalty
                
                # Retropropagación
                d_loss.backward()
                optimizer_D.step()
            
            # -----------------
            # Entrenar Generador
            # -----------------
            optimizer_G.zero_grad()
            
            # Generar imágenes
            z = torch.randn(current_batch_size, LATENT_DIM, device=DEVICE)
            gen_imgs = generator(z)
            
            # Calcular pérdida adversarial
            validity, pred_label, _ = discriminator(gen_imgs)
            g_adv_loss = adversarial_loss(validity, real_label)
            
            # Añadir pérdida de entropía para mejorar la diversidad
            entropy_loss = -torch.mean(torch.sum(F.log_softmax(pred_label, dim=1) * F.softmax(pred_label, dim=1), dim=1))
            
            # Calcular pérdida de características (feature matching)
            _, _, real_features = discriminator(real_imgs)
            _, _, fake_features = discriminator(gen_imgs)
            g_feature_loss = F.mse_loss(fake_features.mean(0), real_features.detach().mean(0))
            
            # Pérdida total del generador
            g_loss = g_adv_loss + 0.1 * g_feature_loss - 0.1 * entropy_loss
            
            # Retropropagación
            g_loss.backward()
            optimizer_G.step()
            
            # Acumular métricas
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_d_real_acc += d_real_acc
            epoch_d_fake_acc += d_fake_acc
            batches_done += 1
            
            # Mostrar progreso
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                      f"[D real acc: {d_real_acc:.2f}] [D fake acc: {d_fake_acc:.2f}] "
                      f"[Aux acc: {aux_acc:.2f}]")
        
        # Calcular promedios
        epoch_g_loss /= batches_done
        epoch_d_loss /= batches_done
        epoch_d_real_acc /= batches_done
        epoch_d_fake_acc /= batches_done
        
        # Actualizar planificadores
        scheduler_G.step(epoch_g_loss)
        scheduler_D.step(epoch_d_loss)
        
        # Guardar métricas
        G_losses.append(epoch_g_loss)
        D_losses.append(epoch_d_loss)
        D_real_acc.append(epoch_d_real_acc)
        D_fake_acc.append(epoch_d_fake_acc)
        
        # Imprimir resumen de época
        print(f"\n[Epoch {epoch}/{EPOCHS}] "
              f"[D loss: {epoch_d_loss:.4f}] [G loss: {epoch_g_loss:.4f}] "
              f"[D real acc: {epoch_d_real_acc:.2f}] [D fake acc: {epoch_d_fake_acc:.2f}] "
              f"[Time: {(time.time() - start_time)/60:.2f} min]\n")
        
        # Guardar imágenes generadas a intervalos regulares
        if epoch % SAMPLE_INTERVAL == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                gen_imgs = generator(fixed_noise)
                save_generator_image(gen_imgs, f"{IMAGES_DIR}/epoch_{epoch}.png")
                
                # Guardar modelos en puntos clave
                if epoch % 20 == 0 or epoch == EPOCHS - 1:
                    torch.save(generator.state_dict(), f"{MODELS_DIR}/generator_epoch_{epoch}.pth")
                    torch.save(discriminator.state_dict(), f"{MODELS_DIR}/discriminator_epoch_{epoch}.pth")

    # Guardar modelos finales
    torch.save(generator.state_dict(), f"{MODELS_DIR}/generator_final.pth")
    torch.save(discriminator.state_dict(), f"{MODELS_DIR}/discriminator_final.pth")

    # Graficar métricas
    plot_training_metrics(G_losses, D_losses, D_real_acc, D_fake_acc)

    # Generar visualización t-SNE de características
    generate_tsne_visualization(generator, discriminator, dataloader)
    
    # Generar grid de imágenes para evaluación final
    generate_evaluation_grid(generator)
    
    # Generar matriz de confusión para clasificación de dígitos
    plot_digit_confusion_matrix(generator, discriminator, get_dataloader(batch_size=64, train=False))
    
    print(f"¡Entrenamiento completado en {(time.time() - start_time)/60:.2f} minutos!")
    print(f"Modelos guardados en la carpeta '{MODELS_DIR}'")
    print(f"Imágenes generadas guardadas en la carpeta '{IMAGES_DIR}'")
    print(f"Métricas y visualizaciones guardadas en la carpeta '{EVALUATION_DIR}'")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    train() 