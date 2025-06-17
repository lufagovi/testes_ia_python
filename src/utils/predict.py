import logging
from tensorflow.keras.preprocessing import image
import numpy as np

# Configura o logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_image(image_path):
    logging.debug(f"Carregando imagem de: {image_path}")
    
    try:
        img = image.load_img(image_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        logging.debug(f"Imagem convertida para array com shape: {img_array.shape}")
        
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        logging.debug(f"Array expandido com shape (batch): {img_array.shape}")
        
        img_array /= 255.0  # Normalize the image
        logging.debug("Imagem normalizada (valores entre 0 e 1)")
        return img_array
    except Exception as e:
        logging.error(f"Erro ao carregar ou processar imagem: {e}")
        raise

def predict_image(model, image_path):
    logging.debug("Iniciando predição da imagem.")
    
    try:
        processed_image = load_and_preprocess_image(image_path)
        prediction = model.predict(processed_image)
        predicted_value = prediction[0][0]
        logging.debug(f"Predição bruta: {prediction}")
        logging.debug(f"Valor previsto (probabilidade): {predicted_value}")
        return {
            "raw_prediction": prediction.tolist(),
            "predicted_value": float(predicted_value),
            "label": int(predicted_value > 0.5)  # binariza com threshold de 0.5
        }
    except Exception as e:
        logging.error(f"Erro durante a predição: {e}")
        raise
