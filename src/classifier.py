import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import kagglehub

# Configuração de diretórios
#os.environ['KAGGLEHUB_CACHE'] = 'C:/tmp/.kaggle'

print("⏬ Baixando dataset...")
dataset_path = kagglehub.dataset_download("d4rklucif3r/cat-and-dogs")
print(f"✅ Dataset baixado em: {dataset_path}")

# Verifique visualmente se as pastas cats/ e dogs/ estão nesse path
print("📂 Verificando estrutura do dataset...")
print("Conteúdo do diretório:", os.listdir(dataset_path))

train_dataset_path = os.path.join(kagglehub.dataset_download("d4rklucif3r/cat-and-dogs"), "dataset/training_set")
test_dataset_path = os.path.join(kagglehub.dataset_download("d4rklucif3r/cat-and-dogs"), "dataset/test_set")


print("Conteúdo do diretório Train:", os.listdir(train_dataset_path))

print("Conteúdo do diretório Test:", os.listdir(test_dataset_path))

# Preparar dados
print("📦 Criando geradores de dados...")
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)



train_gen = datagen.flow_from_directory(train_dataset_path, target_size=(128, 128), class_mode='binary')
val_gen = datagen.flow_from_directory(test_dataset_path, target_size=(128, 128), class_mode='binary')

print(f"🔎 Classes detectadas: {train_gen.class_indices}")
print(f"📊 Amostras de treino: {train_gen.samples}, Amostras de validação: {val_gen.samples}")

# Construção do modelo
print("🛠️ Criando modelo...")
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Treinamento
print("🚀 Iniciando treinamento...")
EPOCHS = 10
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=1,  # Mostra progresso por época
    callbacks=[
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"📈 Epoch {epoch+1}/{EPOCHS} — Loss: {logs['loss']:.4f}, Acc: {logs['accuracy']:.4f}, "
                f"Val_Loss: {logs['val_loss']:.4f}, Val_Acc: {logs['val_accuracy']:.4f}"
            )
        )
    ]
)

# Salvando o modelo
model.save("meu_modelo.keras")
print("✅ Modelo salvo como 'meu_modelo.keras'")
