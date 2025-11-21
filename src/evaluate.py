"""
Funções de avaliação
"""

from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)


def avaliar_modelo(modelo, X_test, y_test):
    """Avalia o modelo no conjunto de teste"""
    
    print("=" * 80)
    print("📊 AVALIAÇÃO DO MODELO")
    print("=" * 80)
    
    # Predições
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]
    
    # Métricas gerais
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n✅ Acurácia: {acc:.4f} ({acc*100:.2f}%)")
    print(f"✅ AUC-ROC:  {auc:.4f} ({auc*100:.2f}%)")
    
    # Relatório de classificação
    print("\n📋 RELATÓRIO DE CLASSIFICAÇÃO:")
    print("=" * 80)
    print(classification_report(y_test, y_pred,
                               target_names=['CANCELED', 'FINISHED'],
                               digits=4))
    
    # Matriz de confusão
    print("🎯 MATRIZ DE CONFUSÃO:")
    print("=" * 80)
    
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n                    Predito")
    print(f"                CANCELED  FINISHED   Total")
    print(f"Real CANCELED      {cm[0,0]:6d}    {cm[0,1]:6d}   {cm[0,0]+cm[0,1]:6d}")
    print(f"     FINISHED      {cm[1,0]:6d}    {cm[1,1]:6d}   {cm[1,0]+cm[1,1]:6d}")
    print(f"     Total         {cm[0,0]+cm[1,0]:6d}    {cm[0,1]+cm[1,1]:6d}   {len(y_test):6d}")
    
    # Métricas para CANCELED
    p_cancel = precision_score(y_test, y_pred, pos_label=0)
    r_cancel = recall_score(y_test, y_pred, pos_label=0)
    f1_cancel = f1_score(y_test, y_pred, pos_label=0)
    
    print("\n🎯 MÉTRICAS PARA CANCELED (classe de interesse):")
    print("=" * 80)
    print(f"   Precisão: {p_cancel:.4f} ({p_cancel*100:.2f}%)")
    print(f"   Recall:   {r_cancel:.4f} ({r_cancel*100:.2f}%)")
    print(f"   F1-Score: {f1_cancel:.4f} ({f1_cancel*100:.2f}%)")
    
    print("\n" + "=" * 80)
    
    return {
        'accuracy': acc,
        'auc': auc,
        'precision_cancel': p_cancel,
        'recall_cancel': r_cancel,
        'f1_cancel': f1_cancel
    }
