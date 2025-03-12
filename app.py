import torch
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import random
import os

from model import Generator, Discriminator
from data_loader import get_mnist_dataset
from config import LATENT_DIM, DEVICE, MODELS_DIR

class GANEvaluationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Evaluador de GAN MNIST")
        self.root.geometry("800x650")
        self.root.configure(bg="#f5f5f5")
        
        # Cargar modelos
        self.generator = Generator(LATENT_DIM).to(DEVICE)
        self.discriminator = Discriminator().to(DEVICE)
        
        try:
            self.generator.load_state_dict(torch.load(f"{MODELS_DIR}/generator_final.pth"))
            self.discriminator.load_state_dict(torch.load(f"{MODELS_DIR}/discriminator_final.pth"))
            print("Modelos cargados correctamente.")
        except Exception as e:
            print(f"Error al cargar modelos: {e}")
            print("Asegúrate de haber entrenado los modelos primero.")
            self.root.destroy()
            return
        
        # Cargar dataset MNIST
        self.mnist_dataset = get_mnist_dataset(train=True)
        
        # Variables
        self.current_image = None
        self.is_real = None
        self.history = []  # Para almacenar historial de evaluaciones
        
        # Configurar estilo
        self._setup_style()
        
        # Crear widgets
        self._create_widgets()
        
        # Generar primera imagen
        self.generate_image()
    
    def _setup_style(self):
        """Configura el estilo de la aplicación"""
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12))
        style.configure("TLabel", font=("Arial", 12), background="#f5f5f5")
        style.configure("Header.TLabel", font=("Arial", 18, "bold"), background="#f5f5f5")
        style.configure("Result.TLabel", font=("Arial", 14, "bold"), background="#f5f5f5")
        style.configure("Info.TLabel", font=("Arial", 10), background="#f5f5f5")
    
    def _create_widgets(self):
        """Crea los widgets de la interfaz"""
        # Título
        header_label = ttk.Label(self.root, text="Evaluador de GAN MNIST", style="Header.TLabel")
        header_label.pack(pady=20)
        
        # Marco para la imagen
        image_frame = ttk.Frame(self.root, borderwidth=2, relief="groove")
        image_frame.pack(pady=10, padx=20, fill="both")
        
        # Imagen
        self.image_label = ttk.Label(image_frame)
        self.image_label.pack()
        
        # Información
        info_frame = ttk.Frame(self.root)
        info_frame.pack(pady=10)
        
        # Variables de texto
        self.source_var = tk.StringVar()
        self.digit_var = tk.StringVar()
        self.result_var = tk.StringVar(value="¿Real o Falsa?")
        self.confidence_var = tk.StringVar()
        self.stats_var = tk.StringVar()
        
        # Etiquetas
        source_label = ttk.Label(info_frame, textvariable=self.source_var, style="TLabel")
        source_label.pack(pady=5)
        
        digit_label = ttk.Label(info_frame, textvariable=self.digit_var, style="TLabel")
        digit_label.pack(pady=5)
        
        result_label = ttk.Label(info_frame, textvariable=self.result_var, style="Result.TLabel")
        result_label.pack(pady=5)
        
        confidence_label = ttk.Label(info_frame, textvariable=self.confidence_var, style="TLabel")
        confidence_label.pack(pady=5)
        
        stats_label = ttk.Label(info_frame, textvariable=self.stats_var, style="Info.TLabel")
        stats_label.pack(pady=10)
        
        # Botones
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=20)
        
        self.generate_button = ttk.Button(button_frame, text="Nueva Imagen", command=self.generate_image)
        self.generate_button.grid(row=0, column=0, padx=10)
        
        self.check_button = ttk.Button(button_frame, text="Verificar", command=self.check_result, state="disabled")
        self.check_button.grid(row=0, column=1, padx=10)
    
    def generate_image(self):
        """Genera una nueva imagen (real o falsa)"""
        # Decidir si mostrar imagen real o generada
        self.is_real = random.random() > 0.5
        
        if self.is_real:
            # Seleccionar imagen real aleatoria
            idx = random.randint(0, len(self.mnist_dataset) - 1)
            img, label = self.mnist_dataset[idx]
            img = img.unsqueeze(0).to(DEVICE)
            source_text = "Dataset MNIST (Real)"
            true_label = label
        else:
            # Generar imagen falsa
            z = torch.randn(1, LATENT_DIM, device=DEVICE)
            with torch.no_grad():
                img = self.generator(z)
            source_text = "Generada por GAN (Falsa)"
            true_label = None
        
        # Evaluar con el discriminador
        with torch.no_grad():
            validity, pred_label, _ = self.discriminator(img)
            confidence = validity.item()
            pred_digit = pred_label.argmax(dim=1).item() if not self.is_real else true_label
        
        # Actualizar interfaz
        self.source_var.set("Origen: ???")
        
        # Convertir tensor a imagen
        img_np = img.cpu().squeeze().numpy()
        img_np = (img_np * 0.5 + 0.5) * 255  # Desnormalizar
        img_pil = Image.fromarray(img_np.astype(np.uint8), mode='L').resize((280, 280), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Actualizar imagen
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk
        
        # Ocultar resultado
        self.result_var.set("Predicción del discriminador: ???")
        self.confidence_var.set("")
        self.digit_var.set(f"Dígito predicho: {pred_digit}")
        
        # Habilitar botón de verificar
        self.check_button.config(state="normal")
        self.generate_button.config(state="normal")
        
        self.current_image = {
            "tensor": img,
            "is_real": self.is_real,
            "confidence": confidence,
            "pred_digit": pred_digit,
            "source": source_text
        }
    
    def check_result(self):
        """Verifica si la imagen es real o falsa"""
        if self.current_image is None:
            return
        
        # Determinar si la imagen es real o falsa (la verdad)
        is_real = self.current_image["is_real"]
        
        # Determinar la predicción del discriminador
        # Si confianza > 0.5, el discriminador cree que es real
        # Si confianza < 0.5, el discriminador cree que es falsa
        discriminator_thinks_real = self.current_image["confidence"] > 0.5
        
        # Determinar si el discriminador acertó
        discriminator_correct = (is_real and discriminator_thinks_real) or (not is_real and not discriminator_thinks_real)
        
        # Mostrar la verdad sobre la imagen
        truth_text = "REAL (del dataset MNIST)" if is_real else "FALSA (generada por GAN)"
        self.source_var.set(f"Origen real: {truth_text}")
        
        # Mostrar la predicción del discriminador
        prediction_text = "REAL" if discriminator_thinks_real else "FALSA"
        self.result_var.set(f"Predicción del discriminador: {prediction_text}")
        
        # Mostrar si el discriminador acertó
        if discriminator_correct:
            accuracy_text = "✓ CORRECTO: El discriminador acertó"
        else:
            accuracy_text = "✗ INCORRECTO: El discriminador falló"
        
        self.confidence_var.set(accuracy_text)
        
        # Deshabilitar botón de verificar
        self.check_button.config(state="disabled")
        self.generate_button.config(state="normal")
        
        # Añadir al historial
        self.history.append({
            "is_real": is_real,
            "discriminator_thinks_real": discriminator_thinks_real,
            "discriminator_correct": discriminator_correct,
            "pred_digit": self.current_image["pred_digit"]
        })
        
        # Actualizar estadísticas
        self.update_stats()
    
    def update_stats(self):
        """Actualiza las estadísticas basadas en el historial"""
        if not self.history:
            return
        
        # Calcular estadísticas
        total = len(self.history)
        real_count = sum(1 for item in self.history if item["is_real"])
        fake_count = total - real_count
        
        # Aciertos del discriminador
        correct_count = sum(1 for item in self.history if item["discriminator_correct"])
        correct_real = sum(1 for item in self.history if item["is_real"] and item["discriminator_correct"])
        correct_fake = sum(1 for item in self.history if not item["is_real"] and item["discriminator_correct"])
        
        # Calcular precisiones
        overall_accuracy = (correct_count / total * 100) if total > 0 else 0
        real_accuracy = (correct_real / real_count * 100) if real_count > 0 else 0
        fake_accuracy = (correct_fake / fake_count * 100) if fake_count > 0 else 0
        
        # Actualizar etiquetas con información simplificada
        self.stats_var.set(
            f"Estadísticas (total: {total}):\n"
            f"Imágenes reales: {real_count}, Falsas: {fake_count}\n\n"
            f"Precisión del discriminador: {overall_accuracy:.1f}%\n"
            f"  - En reales: {real_accuracy:.1f}% ({correct_real}/{real_count})\n"
            f"  - En falsas: {fake_accuracy:.1f}% ({correct_fake}/{fake_count})\n\n"
            f"Último resultado:\n"
            f"  - Tipo: {'REAL' if self.current_image['is_real'] else 'FALSA'}\n"
            f"  - Predicción: {'CORRECTA' if self.history[-1]['discriminator_correct'] else 'INCORRECTA'}"
        )

def main():
    root = tk.Tk()
    app = GANEvaluationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 