text
# 🎯 Previsão de Cancelamento de Pedidos - Naive Bayes

Modelo Naive Bayes otimizado para prever cancelamento de pedidos.

## 📋 Instalação

pip install -r requirements.txt

text

## 🚀 Como Usar

1. Coloque os CSVs na pasta `data/`
2. Execute:

python main.py

text

## 📊 Resultados Esperados

- **F1-Score CANCELED**: ~71%
- **Acurácia**: ~97%
- **Recall**: ~89%

## 📂 Estrutura

- `src/preprocessing.py`: Pré-processamento dos dados
- `src/train.py`: Treinamento do modelo
- `src/evaluate.py`: Avaliação de métricas
- `main.py`: Execução completa

## ⚙️ Configuração

Parâmetros principais em `src/train.py`:
- `sampling_strategy=0.6`: Proporção SMOTE (60-40)
- `test_size=0.2`: 20% para teste