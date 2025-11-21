"""
Funções de treinamento
"""

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from collections import Counter


def split_dados(X, y, test_size=0.2, random_state=42):
    """Divide em treino e teste"""
    
    print("\n📊 Dividindo em treino/teste...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"✅ Treino: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"✅ Teste:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test


def balancear_dados(X_train, y_train, sampling_strategy=0.6, random_state=42):
    """Aplica SMOTE para balanceamento"""
    
    print("\n⚖️ Balanceando com SMOTE...")
    
    print(f"ANTES:  {len(y_train):,} amostras")
    print(f"   CANCELED: {(y_train==0).sum():,}")
    print(f"   FINISHED: {(y_train==1).sum():,}")
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"DEPOIS: {len(y_balanced):,} amostras")
    print(f"   CANCELED: {(y_balanced==0).sum():,}")
    print(f"   FINISHED: {(y_balanced==1).sum():,}")
    
    return X_balanced, y_balanced


def treinar_naive_bayes(X_train, y_train):
    """Treina Gaussian Naive Bayes"""
    
    print("\n🤖 Treinando Naive Bayes...")
    
    modelo = GaussianNB()
    modelo.fit(X_train, y_train)
    
    print(f"✅ Modelo treinado com {len(y_train):,} amostras")
    
    return modelo


def cross_validation(X_train, y_train, n_splits=5):
    """Executa cross-validation"""
    
    print(f"\n🔄 Cross-Validation ({n_splits}-Fold)...")
    
    modelo = GaussianNB()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
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


def pipeline_completo(X, y):
    """
    Pipeline completo de treinamento
    
    Returns:
        modelo, X_train, X_test, y_train, y_test
    """
    
    print("=" * 80)
    print("🚀 PIPELINE DE TREINAMENTO")
    print("=" * 80)
    
    # 1. Split
    X_train, X_test, y_train, y_test = split_dados(X, y)
    
    # 2. Balancear
    X_train_bal, y_train_bal = balancear_dados(X_train, y_train, sampling_strategy=0.6)
    
    # 3. Cross-validation
    cross_validation(X_train_bal, y_train_bal)
    
    # 4. Treinar
    modelo = treinar_naive_bayes(X_train_bal, y_train_bal)
    
    print("\n" + "=" * 80)
    print("✅ TREINAMENTO CONCLUÍDO!")
    print("=" * 80)
    
    return modelo, X_train, X_test, y_train, y_test
