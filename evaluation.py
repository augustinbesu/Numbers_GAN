import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from config import DEVICE, LATENT_DIM, BATCH_SIZE, EVALUATION_DIR
from utils import save_generator_image
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

def generate_tsne_visualization(generator, discriminator, dataloader, n_samples=500):
    """Genera visualización t-SNE de características reales y generadas"""
    print("Generando visualización t-SNE...")
    
    # Recopilar características
    real_features = []
    fake_features = []
    real_labels = []
    
    with torch.no_grad():
        # Recopilar características reales
        for imgs, labels in dataloader:
            if len(real_features) * imgs.size(0) >= n_samples:
                break
                
            imgs = imgs.to(DEVICE)
            _, _, features = discriminator(imgs)
            real_features.append(features.cpu().numpy())
            real_labels.append(labels.numpy())
        
        # Recopilar características falsas
        for _ in range(n_samples // BATCH_SIZE + 1):
            if len(fake_features) * BATCH_SIZE >= n_samples:
                break
                
            z = torch.randn(BATCH_SIZE, LATENT_DIM, device=DEVICE)
            fake_imgs = generator(z)
            _, _, features = discriminator(fake_imgs)
            fake_features.append(features.cpu().numpy())
    
    # Concatenar características
    real_features = np.vstack(real_features)[:n_samples]
    fake_features = np.vstack(fake_features)[:n_samples]
    real_labels = np.concatenate(real_labels)[:n_samples]
    
    # Combinar características
    combined_features = np.vstack([real_features, fake_features])
    combined_labels = np.concatenate([real_labels, np.ones(fake_features.shape[0]) * 10])  # 10 = fake
    
    # Aplicar t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(combined_features)
    
    # Visualizar
    plt.figure(figsize=(12, 10))
    
    # Colores para dígitos reales (0-9) y generados (10)
    colors = plt.cm.tab10(np.arange(10))
    colors = np.vstack([colors, [0, 0, 0, 1]])  # Negro para generados
    
    # Graficar puntos
    for i in range(11):  # 0-9 real, 10 fake
        mask = combined_labels == i
        plt.scatter(
            embedded[mask, 0], embedded[mask, 1],
            c=[colors[i]], label=f"{'Generado' if i == 10 else i}",
            alpha=0.7, s=10
        )
    
    plt.title("Visualización t-SNE de características reales y generadas")
    plt.legend(markerscale=2)
    plt.savefig(f"{EVALUATION_DIR}/tsne_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_evaluation_grid(generator, n_rows=10, n_cols=10):
    """Genera una cuadrícula de imágenes para evaluación final"""
    print("Generando cuadrícula de evaluación...")
    
    # Generar imágenes
    with torch.no_grad():
        z = torch.randn(n_rows * n_cols, LATENT_DIM, device=DEVICE)
        gen_imgs = generator(z)
    
    # Crear figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    
    # Mostrar imágenes
    for i, ax in enumerate(axes.flat):
        img = gen_imgs[i].cpu().squeeze().numpy()
        img = (img * 0.5 + 0.5)  # Desnormalizar
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{EVALUATION_DIR}/final_grid.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_metrics(G_losses, D_losses, D_real_acc, D_fake_acc):
    """Grafica las métricas de entrenamiento en imágenes separadas para mayor claridad"""
    # Configuración general de estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['font.size'] = 12
    
    # Crear directorio para métricas individuales
    metrics_dir = f"{EVALUATION_DIR}/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Preparar datos
    epochs = range(1, len(G_losses) + 1)
    
    # 1. Gráfico de pérdidas
    plt.figure(figsize=(12, 8))
    
    plt.plot(epochs, G_losses, 'g-', linewidth=2.5, label='Generador')
    plt.plot(epochs, D_losses, 'b-', linewidth=2.5, label='Discriminador')
    
    # Añadir líneas de tendencia suavizadas
    from scipy.signal import savgol_filter
    if len(G_losses) > 10:  # Solo si hay suficientes puntos
        window = min(15, len(G_losses) // 3 * 2 + 1)  # Asegurar que window es impar
        if window % 2 == 0:
            window += 1
        g_smooth = savgol_filter(G_losses, window, 3)
        d_smooth = savgol_filter(D_losses, window, 3)
        plt.plot(epochs, g_smooth, 'g--', linewidth=1.5, alpha=0.7, label='G (tendencia)')
        plt.plot(epochs, d_smooth, 'b--', linewidth=1.5, alpha=0.7, label='D (tendencia)')
    
    # Añadir anotaciones para valores mínimos
    min_g_loss = min(G_losses)
    min_g_epoch = G_losses.index(min_g_loss) + 1
    min_d_loss = min(D_losses)
    min_d_epoch = D_losses.index(min_d_loss) + 1
    
    plt.annotate(f'Min G: {min_g_loss:.4f} (Época {min_g_epoch})',
                xy=(min_g_epoch, min_g_loss), xytext=(min_g_epoch, min_g_loss*1.2),
                arrowprops=dict(facecolor='green', shrink=0.05, alpha=0.7),
                fontsize=12, color='green')
    
    plt.annotate(f'Min D: {min_d_loss:.4f} (Época {min_d_epoch})',
                xy=(min_d_epoch, min_d_loss), xytext=(min_d_epoch, min_d_loss*1.2),
                arrowprops=dict(facecolor='blue', shrink=0.05, alpha=0.7),
                fontsize=12, color='blue')
    
    plt.title('Pérdidas durante el Entrenamiento', fontsize=18, fontweight='bold')
    plt.xlabel('Épocas', fontsize=14)
    plt.ylabel('Pérdida', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Guardar figura
    plt.savefig(f"{metrics_dir}/losses.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Gráfico de precisión
    plt.figure(figsize=(12, 8))
    
    plt.plot(epochs, D_real_acc, 'r-', linewidth=2.5, label='Imágenes Reales')
    plt.plot(epochs, D_fake_acc, 'm-', linewidth=2.5, label='Imágenes Falsas')
    
    # Línea de referencia en 0.5 (aleatorio)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Aleatorio (0.5)')
    
    # Añadir líneas de tendencia suavizadas
    if len(D_real_acc) > 10:
        window = min(15, len(D_real_acc) // 3 * 2 + 1)
        if window % 2 == 0:
            window += 1
        real_smooth = savgol_filter(D_real_acc, window, 3)
        fake_smooth = savgol_filter(D_fake_acc, window, 3)
        plt.plot(epochs, real_smooth, 'r--', linewidth=1.5, alpha=0.7, label='Real (tendencia)')
        plt.plot(epochs, fake_smooth, 'm--', linewidth=1.5, alpha=0.7, label='Falsa (tendencia)')
    
    # Añadir anotaciones para valores máximos
    max_real_acc = max(D_real_acc)
    max_real_epoch = D_real_acc.index(max_real_acc) + 1
    max_fake_acc = max(D_fake_acc)
    max_fake_epoch = D_fake_acc.index(max_fake_acc) + 1
    
    plt.annotate(f'Max Real: {max_real_acc:.4f} (Época {max_real_epoch})',
                xy=(max_real_epoch, max_real_acc), xytext=(max_real_epoch, max_real_acc*0.9),
                arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.7),
                fontsize=12, color='red')
    
    plt.annotate(f'Max Falsa: {max_fake_acc:.4f} (Época {max_fake_epoch})',
                xy=(max_fake_epoch, max_fake_acc), xytext=(max_fake_epoch, max_fake_acc*0.9),
                arrowprops=dict(facecolor='magenta', shrink=0.05, alpha=0.7),
                fontsize=12, color='magenta')
    
    plt.title('Precisión del Discriminador', fontsize=18, fontweight='bold')
    plt.xlabel('Épocas', fontsize=14)
    plt.ylabel('Precisión', fontsize=14)
    plt.ylim([0, 1.05])
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Guardar figura
    plt.savefig(f"{metrics_dir}/accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Gráfico de diferencia de precisión
    plt.figure(figsize=(12, 8))
    
    diff_acc = [r - f for r, f in zip(D_real_acc, D_fake_acc)]
    
    # Colorear según si está balanceado o no
    colors = ['green' if abs(d) < 0.2 else 'orange' if abs(d) < 0.4 else 'red' for d in diff_acc]
    plt.bar(epochs, diff_acc, color=colors, alpha=0.7)
    
    # Línea de referencia en 0 (equilibrio perfecto)
    plt.axhline(y=0, color='blue', linestyle='-', alpha=0.5, label='Equilibrio')
    
    # Zonas de equilibrio
    plt.axhspan(-0.2, 0.2, alpha=0.2, color='green', label='Equilibrio óptimo')
    plt.axhspan(-0.4, -0.2, alpha=0.1, color='orange')
    plt.axhspan(0.2, 0.4, alpha=0.1, color='orange', label='Equilibrio aceptable')
    
    plt.title('Diferencia de Precisión (Real - Falsa)', fontsize=18, fontweight='bold')
    plt.xlabel('Épocas', fontsize=14)
    plt.ylabel('Diferencia', fontsize=14)
    plt.ylim([-1, 1])
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Guardar figura
    plt.savefig(f"{metrics_dir}/accuracy_difference.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Gráfico de relación pérdida-precisión
    plt.figure(figsize=(12, 8))
    
    # Calcular precisión promedio
    avg_acc = [(r + f) / 2 for r, f in zip(D_real_acc, D_fake_acc)]
    
    # Scatter plot con colores según la época
    scatter = plt.scatter(G_losses, avg_acc, c=epochs, cmap='viridis', 
                         s=100, alpha=0.7, edgecolors='w', linewidth=0.5)
    
    # Añadir flechas para mostrar la dirección del entrenamiento
    for i in range(0, len(epochs)-1, max(1, len(epochs)//15)):  # Mostrar ~15 flechas
        plt.annotate('', xy=(G_losses[i+1], avg_acc[i+1]), xytext=(G_losses[i], avg_acc[i]),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=1.5))
    
    # Marcar inicio y fin
    plt.scatter([G_losses[0]], [avg_acc[0]], c='blue', s=150, marker='o', label='Inicio')
    plt.scatter([G_losses[-1]], [avg_acc[-1]], c='red', s=150, marker='*', label='Fin')
    
    # Colorbar para mostrar la época
    cbar = plt.colorbar(scatter)
    cbar.set_label('Época', fontsize=12)
    
    plt.title('Relación Pérdida G vs. Precisión D', fontsize=18, fontweight='bold')
    plt.xlabel('Pérdida del Generador', fontsize=14)
    plt.ylabel('Precisión Promedio del Discriminador', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Guardar figura
    plt.savefig(f"{metrics_dir}/loss_vs_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Gráfico de convergencia
    plt.figure(figsize=(12, 8))
    
    # Calcular ratio de pérdidas G/D
    loss_ratio = [g/d if d > 0 else 0 for g, d in zip(G_losses, D_losses)]
    
    # Línea para el ratio
    plt.plot(epochs, loss_ratio, 'purple', linewidth=2.5, label='Ratio G/D')
    
    # Línea de referencia en 1.0 (equilibrio)
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Equilibrio (1.0)')
    
    # Añadir línea de tendencia suavizada
    if len(loss_ratio) > 10:
        window = min(15, len(loss_ratio) // 3 * 2 + 1)
        if window % 2 == 0:
            window += 1
        ratio_smooth = savgol_filter(loss_ratio, window, 3)
        plt.plot(epochs, ratio_smooth, 'purple', linestyle='--', linewidth=1.5, 
                alpha=0.7, label='Tendencia')
    
    plt.title('Convergencia (Ratio G/D)', fontsize=18, fontweight='bold')
    plt.xlabel('Épocas', fontsize=14)
    plt.ylabel('Ratio de Pérdidas (G/D)', fontsize=14)
    plt.ylim([0, max(3, max(loss_ratio) * 1.1)])
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Guardar figura
    plt.savefig(f"{metrics_dir}/convergence_ratio.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Crear una imagen de resumen (más pequeña)
    plt.figure(figsize=(15, 10))
    
    # Pérdidas
    plt.subplot(2, 2, 1)
    plt.plot(epochs, G_losses, 'g-', label='G')
    plt.plot(epochs, D_losses, 'b-', label='D')
    plt.title('Pérdidas', fontsize=14)
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Precisión
    plt.subplot(2, 2, 2)
    plt.plot(epochs, D_real_acc, 'r-', label='Real')
    plt.plot(epochs, D_fake_acc, 'm-', label='Falsa')
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.title('Precisión', fontsize=14)
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.ylim([0, 1.05])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Diferencia
    plt.subplot(2, 2, 3)
    plt.bar(epochs, diff_acc, color=colors, alpha=0.7)
    plt.axhline(y=0, color='blue', linestyle='-', alpha=0.5)
    plt.title('Diferencia Real-Falsa', fontsize=14)
    plt.xlabel('Épocas')
    plt.ylabel('Diferencia')
    plt.ylim([-1, 1])
    plt.grid(True, alpha=0.3)
    
    # Ratio
    plt.subplot(2, 2, 4)
    plt.plot(epochs, loss_ratio, 'purple', linewidth=2)
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
    plt.title('Ratio G/D', fontsize=14)
    plt.xlabel('Épocas')
    plt.ylabel('Ratio')
    plt.ylim([0, min(3, max(loss_ratio) * 1.1)])
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Resumen de Métricas de Entrenamiento', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Guardar figura de resumen
    plt.savefig(f"{EVALUATION_DIR}/training_metrics_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Métricas guardadas en {metrics_dir}/")
    print(f"Resumen guardado en {EVALUATION_DIR}/training_metrics_summary.png")

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Calcula la penalización de gradiente para WGAN-GP"""
    # Muestras aleatorias uniformes
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=DEVICE)
    
    # Interpolación lineal entre muestras reales y falsas
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Calcular puntuación del discriminador
    d_interpolates, _, _ = discriminator(interpolates)
    
    # Obtener gradientes con respecto a las entradas
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # Calcular la norma del gradiente
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

def plot_digit_confusion_matrix(generator, discriminator, dataloader):
    """Genera una matriz de confusión para la clasificación de dígitos"""
    print("Generando matriz de confusión para clasificación de dígitos...")
    
    # Recopilar predicciones
    real_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            if len(real_labels) >= 1000:  # Limitar a 1000 muestras para velocidad
                break
                
            imgs = imgs.to(DEVICE)
            _, aux_output, _ = discriminator(imgs)
            predictions = aux_output.max(1)[1].cpu().numpy()
            
            real_labels.extend(labels.numpy())
            pred_labels.extend(predictions)
    
    # Calcular matriz de confusión
    cm = confusion_matrix(real_labels, pred_labels)
    
    # Normalizar por fila (verdaderos)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Visualizar
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicción', fontsize=14)
    plt.ylabel('Verdadero', fontsize=14)
    plt.title('Matriz de Confusión - Clasificación de Dígitos', fontsize=18, fontweight='bold')
    
    # Calcular precisión global
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.5, 0.01, f"Precisión global: {accuracy:.4f}", ha="center", 
                fontsize=14, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.savefig(f"{EVALUATION_DIR}/metrics/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Matriz de confusión guardada en {EVALUATION_DIR}/metrics/confusion_matrix.png") 