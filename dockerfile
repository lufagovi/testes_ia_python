FROM python:3.10-slim

# Define a pasta de trabalho (não copia nada)
WORKDIR /app

# Instala dependências via requirements.txt que será montado com volume
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta do Flask
EXPOSE 5000

# Comando padrão (ajuste se precisar)
CMD ["python", "app.py"]
