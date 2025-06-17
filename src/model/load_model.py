from tensorflow.keras.models import load_model

def load_trained_model(model_path):
    model = load_model(model_path)
    return model

model = load_trained_model("meu_modelo.keras")  # Altere o caminho conforme necessário para o seu modelo treinado.