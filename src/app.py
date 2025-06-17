from flask import Flask, request, render_template, jsonify
import os
import logging
from model.load_model import load_model
from utils.predict import predict_image

from sentence_transformers import SentenceTransformer, util
import time  

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Carregar o modelo
MODEL_PATH = "meu_modelo.keras"
logging.info(f"Carregando modelo de: {MODEL_PATH}")
model = load_model(MODEL_PATH)

@app.route('/')
def index():
    logging.debug("Rota / acessada.")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logging.debug("Rota /predict acessada.")
    
    if 'image' not in request.files:
        logging.warning("Nenhum arquivo na requisição.")
        return render_template('index.html', prediction="Nenhum arquivo enviado.")
    
    file = request.files['image']
    
    if file.filename == '':
        logging.warning("Nenhum arquivo selecionado.")
        return render_template('index.html', prediction="Nenhum arquivo selecionado.")
    
    if file:
        try:
            # Salvar a imagem temporariamente
            save_dir = 'static'
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, file.filename)
            file.save(filepath)
            logging.info(f"Imagem salva em: {filepath}")
            
            # Fazer a previsão
            prediction = predict_image(model, filepath)
            logging.info(f"Predição final: {prediction}")
            
            return render_template('index.html', prediction=prediction)
        except Exception as e:
            logging.error(f"Erro durante o processo de predição: {e}")
            return render_template('index.html', prediction=f"Erro: {str(e)}")



@app.route('/semantic', methods=['GET', 'POST'])
def semantic():
    resultado = None
    entrada = ''
    opcoes_raw = ''

    if request.method == 'POST':
        entrada = request.form['entrada']
        opcoes_raw = request.form['opcoes']
        opcoes = [linha.strip() for linha in opcoes_raw.splitlines() if linha.strip()]

        if entrada and opcoes:
            t0 = time.time() 
            emb_entrada = model.encode(entrada, convert_to_tensor=True)
            emb_opcoes = model.encode(opcoes, convert_to_tensor=True)
            similaridades = util.cos_sim(emb_entrada, emb_opcoes)[0]
            idx = similaridades.argmax().item()
            score = float(similaridades[idx])

            t1 = time.time()  # <-- Fim da medição
            duracao_ms = int((t1 - t0) * 1000)

            if score >= 0.5:
                resultado = f"✅ Mais próxima: \"{opcoes[idx]}\" (score: {score:.2f})"
            else:
                resultado = f"❌ Nenhuma opção parece relevante (melhor score: {score:.2f})"

    return render_template('semantic.html', resultado=resultado, entrada=entrada, opcoes=opcoes_raw,duracao_ms=duracao_ms)        

if __name__ == '__main__':
    logging.info("Inicializando servidor Flask...")
    app.run(host='0.0.0.0', port=5000)
