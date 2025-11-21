"""
Script principal - Executa todo o pipeline
"""

from src.preprocessing import preprocessar_tudo
from src.train import pipeline_completo
from src.evaluate import avaliar_modelo


def main():
    """Função principal"""
    
    print("\n" + "🎯" * 40)
    print("      PREVISÃO DE CANCELAMENTO - NAIVE BAYES")
    print("🎯" * 40 + "\n")
    
    # 1. Pré-processamento
    X, y = preprocessar_tudo(data_path='data/')
    
    # 2. Treinamento
    modelo, X_train, X_test, y_train, y_test = pipeline_completo(X, y)
    
    # 3. Avaliação
    metricas = avaliar_modelo(modelo, X_test, y_test)
    
    # Resultado final
    print("\n" + "=" * 80)
    print("🏆 RESULTADO FINAL")
    print("=" * 80)
    print(f"""
    Acurácia:         {metricas['accuracy']*100:.2f}%
    AUC-ROC:          {metricas['auc']*100:.2f}%
    
    CANCELED:
       Precisão:      {metricas['precision_cancel']*100:.2f}%
       Recall:        {metricas['recall_cancel']*100:.2f}%
       F1-Score:      {metricas['f1_cancel']*100:.2f}%
    """)
    print("=" * 80)


if __name__ == "__main__":
    main()
