"""
Funções de treinamento
Configuração vencedora: SMOTE 60-40
"""

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from collections import Counter
import time


def split_dados(X, y, test_size=0.2, random_state=None):
    """Divide em treino e teste"""
    
    if random_state is None:
        random_state = int(time.time())
    
    print("\nDividindo em treino/teste...")
    print(f"Random state: {random_state}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"Treino: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Teste:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test, random_state


def balancear_dados(X_train, y_train, sampling_strategy=0.6, random_state=42):
    """
    Aplica SMOTE para balanceamento
    """
    
    print("\nBalanceando com SMOTE 60-40...")
    
    print(f"ANTES:  {len(y_train):,} amostras")
    print(f"   CANCELED: {(y_train==0).sum():,} ({(y_train==0).sum()/len(y_train)*100:.2f}%)")
    print(f"   FINISHED: {(y_train==1).sum():,} ({(y_train==1).sum()/len(y_train)*100:.2f}%)")
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"DEPOIS: {len(y_balanced):,} amostras")
    print(f"   CANCELED: {(y_balanced==0).sum():,} ({(y_balanced==0).sum()/len(y_balanced)*100:.2f}%)")
    print(f"   FINISHED: {(y_balanced==1).sum():,} ({(y_balanced==1).sum()/len(y_balanced)*100:.2f}%)")
    
    return X_balanced, y_balanced


def treinar_naive_bayes(X_train, y_train):
    """Treina Gaussian Naive Bayes"""
    
    print("\nTreinando Gaussian Naive Bayes...")
    
    modelo = GaussianNB()
    modelo.fit(X_train, y_train)
    
    print(f"Modelo treinado com {len(y_train):,} amostras")
    
    return modelo


def cross_validation(X_train, y_train, n_splits=6, random_state=42):
    """Executa cross-validation com SMOTE em cada fold"""
    
    print(f"\nCross-Validation ({n_splits}-Fold Stratified)...")
    
    modelo = GaussianNB()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    scores = cross_val_score(
        modelo, X_train, y_train,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    for i, score in enumerate(scores, 1):
        print(f"   Fold {i}: {score:.4f}")
    
    print(f"   {'─' * 30}")
    print(f"   Média: {scores.mean():.4f} (±{scores.std():.4f})")
    
    return scores


def pipeline_completo(X, y, random_state=None):
    """
    Pipeline completo de treinamento com SMOTE 60-40
    
    Configuração Final:
    - SMOTE 60
    - 6-Fold Stratified CV
    - Random state dinâmico (timestamp)
    
    Returns:
        modelo, X_train, X_test, y_train, y_test
    """
    
    print("=" * 80)
    print("PIPELINE DE TREINAMENTO - CONFIGURAÇÃO OTIMIZADA")
    print("=" * 80)
    print("   Técnica: SMOTE")
    print("   Algoritmo: Gaussian Naive Bayes")
    print("   Features: 21")
    print("=" * 80)
    
    # 1. Split (com random_state dinâmico)
    X_train, X_test, y_train, y_test, rs = split_dados(X, y, random_state=random_state)
    
    # 2. Balancear com SMOTE 60-40
    X_train_bal, y_train_bal = balancear_dados(X_train, y_train, sampling_strategy=0.6, random_state=rs)
    
    # 3. Cross-validation
    cross_validation(X_train_bal, y_train_bal, random_state=rs)
    
    # 4. Treinar modelo final
    modelo = treinar_naive_bayes(X_train_bal, y_train_bal)
    
    print("\n" + "=" * 80)
    print("TREINAMENTO CONCLUÍDO")
    print("=" * 80)
    
    return modelo, X_train, X_test, y_train, y_test
