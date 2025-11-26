import time
from src.preprocessing import preprocessar_tudo
from src.train import pipeline_completo
from src.evaluate import avaliar_modelo


def main():
    """Função principal - Versão Final Otimizada"""
    
    # Random state dinâmico
    random_state = int(time.time())
    
    print("\n" + "🎯" * 40)
    print("      PREVISÃO DE CANCELAMENTO - NAIVE BAYES")
    print("      CONFIGURAÇÃO OTIMIZADA: SMOTE 60-40")
    print("🎯" * 40)
    print(f"\nRandom state: {random_state}\n")
    
    # 1. Pré-processamento
    X, y = preprocessar_tudo(data_path='data/')
    
    # 2. Treinamento com SMOTE 60-40
    modelo, X_train, X_test, y_train, y_test = pipeline_completo(X, y, random_state=random_state)
    
    # 3. Avaliação
    metricas = avaliar_modelo(modelo, X_test, y_test)
    
    # Resultado final
    print("\n" + "=" * 80)
    print("🏆 RESULTADO FINAL - SMOTE 60-40")
    print("=" * 80)
    print(f"""
    📊 MÉTRICAS GERAIS:
       Acurácia:         {metricas['accuracy']*100:.2f}%
       AUC-ROC:          {metricas['auc']*100:.2f}%
    
       MÉTRICAS PARA CANCELED (classe de interesse):
       Precisão:         {metricas['precision_cancel']*100:.2f}%
       Recall:           {metricas['recall_cancel']*100:.2f}%
       F1-Score:         {metricas['f1_cancel']*100:.2f}%

    """)
    print("=" * 80)
    
    # Salvar informações importantes
    print("\n💾 INFORMAÇÕES PARA REPRODUÇÃO:")
    print(f"   Random state usado: {random_state}")
    print(f"   Técnica: SMOTE 60-40")
    print(f"   Features: 21")
    print(f"   Split: 80/20 stratified")
    print("=" * 80)


if __name__ == "__main__":
    main()

