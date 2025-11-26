"""
Funções de pré-processamento
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def carregar_dados(data_path='data/'):
    """Carrega todos os CSVs"""
    
    print("📂 Carregando datasets...")
    
    orders = pd.read_csv(f"{data_path}orders.csv", encoding="latin-1")
    stores = pd.read_csv(f"{data_path}stores.csv", encoding="latin-1")
    payments = pd.read_csv(f"{data_path}payments.csv", encoding="latin-1")
    channels = pd.read_csv(f"{data_path}channels.csv", encoding="latin-1")
    hubs = pd.read_csv(f"{data_path}hubs.csv", encoding="latin-1")
    deliveries = pd.read_csv(f"{data_path}deliveries.csv", encoding="latin-1")
    drivers = pd.read_csv(f"{data_path}drivers.csv", encoding="latin-1")
    
    print(f"{len(orders):,} pedidos carregados")
    
    return orders, stores, payments, channels, hubs, deliveries, drivers


def fazer_merge(orders, stores, payments, channels, hubs, deliveries, drivers):
    """Combina todos os datasets"""
    
    print("\nFazendo merge dos datasets...")
    
    # Merge principal
    df = orders.merge(stores, on='store_id', how='left')
    df = df.merge(hubs, on='hub_id', how='left')
    df = df.merge(channels, on='channel_id', how='left')
    
    # Agregar payments por pedido
    payments_agg = payments.groupby('payment_order_id').agg({
        'payment_amount': 'sum',
        'payment_method': 'first'
    }).rename(columns={
        'payment_amount': 'total_payment_amount',
        'payment_method': 'main_payment_method'
    }).reset_index()
    payments_agg.rename(columns={'payment_order_id': 'order_id'}, inplace=True)
    
    df = df.merge(payments_agg, on='order_id', how='left')
    
    # Deliveries
    deliveries_clean = deliveries[['delivery_order_id', 'driver_id', 'delivery_distance_meters']].copy()
    deliveries_clean = deliveries_clean.drop_duplicates(subset=['delivery_order_id'], keep='last')
    deliveries_clean.rename(columns={'delivery_order_id': 'order_id'}, inplace=True)
    df = df.merge(deliveries_clean, on='order_id', how='left')
    
    # Drivers
    df = df.merge(drivers, on='driver_id', how='left')
    
    # Flag has_driver
    df['has_driver'] = df['driver_id'].notna().astype(int)
    
    print(f"Dataset merged: {df.shape}")
    
    return df


def limpar_dados(df):
    """Remove data leakage e filtra status"""
    
    print("\nLimpando dados...")
    
    # Filtrar apenas CANCELED e FINISHED
    df_clean = df[df['order_status'].isin(['CANCELED', 'FINISHED'])].copy()
    
    print(f"   Linhas após filtro: {len(df_clean):,}")
    print(f"   CANCELED: {(df_clean['order_status']=='CANCELED').sum():,}")
    print(f"   FINISHED: {(df_clean['order_status']=='FINISHED').sum():,}")
    
    return df_clean


def criar_features(df):
    """Cria features engenheiradas"""
    
    print("\nCriando features...")
    
    # Dia da semana
    df['day_of_week'] = pd.to_datetime(df['order_moment_created']).dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Período do dia
    def get_period(hour):
        if 0 <= hour < 6:
            return 0  # madrugada
        elif 6 <= hour < 12:
            return 1  # manhã
        elif 12 <= hour < 18:
            return 2  # tarde
        else:
            return 3  # noite
    
    df['period'] = df['order_created_hour'].apply(get_period)
    
    # Taxa de cancelamento por loja
    store_cancel_rate = df.groupby('store_name')['order_status'].apply(
        lambda x: (x == 'CANCELED').sum() / len(x)
    ).to_dict()
    df['store_cancel_rate'] = df['store_name'].map(store_cancel_rate)
    
    print("Features criadas: day_of_week, is_weekend, period, store_cancel_rate")
    
    return df


def selecionar_features(df):
    """Seleciona apenas features válidas (sem data leakage)"""
    
    print("\nSelecionando features...")
    
    features_validas = [
        # Target
        'order_status',
        
        # Numéricas
        'order_amount',
        'order_delivery_fee',
        'order_created_hour',
        'delivery_distance_meters',
        'store_plan_price',
        'day_of_week',
        'store_cancel_rate',
        
        # Binárias
        'is_weekend',
        'has_driver',
        
        # Categóricas
        'store_name',
        'store_segment',
        'hub_name',
        'hub_city',
        'hub_state',
        'channel_name',
        'channel_type',
        'period'
    ]
    
    df_final = df[features_validas].copy()
    
    print(f"{len(features_validas)-1} features selecionadas")
    
    return df_final


def tratar_nulos(df):
    """Trata valores nulos"""
    
    print("\n🔧 Tratando nulos...")
    
    # Numéricas: mediana
    for col in ['delivery_distance_meters', 'store_plan_price']:
        if df[col].isnull().any():
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)
            print(f"   {col}: preenchido com {mediana}")
    
    # Categóricas: moda
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'order_status' and df[col].isnull().any():
            moda = df[col].mode()[0]
            df[col] = df[col].fillna(moda)
            print(f"   {col}: preenchido com {moda}")
    
    print(f"✅ Nulos restantes: {df.isnull().sum().sum()}")
    
    return df


def tratar_outliers(df):
    """Trata outliers com capping"""
    
    print("\n📊 Tratando outliers...")
    
    colunas_outliers = {
        'order_amount': 246.15,
        'delivery_distance_meters': 6806.00,
        'store_cancel_rate': 0.10
    }
    
    for coluna, limite in colunas_outliers.items():
        if coluna in df.columns:
            outliers = (df[coluna] > limite).sum()
            df[coluna] = df[coluna].clip(upper=limite)
            print(f"   {coluna}: {outliers:,} outliers tratados")
    
    return df


def codificar_categoricas(df):
    """Codifica variáveis categóricas"""
    
    print("\n🔤 Codificando categóricas...")
    
    # Separar target antes
    target = df['order_status']
    df = df.drop('order_status', axis=1)
    
    # One-Hot Encoding
    colunas_onehot = ['store_segment', 'hub_city', 'hub_state', 'channel_type']
    df = pd.get_dummies(df, columns=colunas_onehot, drop_first=True, dtype=int)
    
    print(f"   One-Hot: {', '.join(colunas_onehot)}")
    
    # Label Encoding
    le_store = LabelEncoder()
    le_hub = LabelEncoder()
    le_channel = LabelEncoder()
    
    df['store_name_encoded'] = le_store.fit_transform(df['store_name'].astype(str))
    df['hub_name_encoded'] = le_hub.fit_transform(df['hub_name'].astype(str))
    df['channel_name_encoded'] = le_channel.fit_transform(df['channel_name'].astype(str))
    
    # Remover originais
    df = df.drop(['store_name', 'hub_name', 'channel_name'], axis=1)
    
    print(f"   Label Encoding: store_name, hub_name, channel_name")
    
    # Target: FINISHED=1, CANCELED=0
    y = (target == 'FINISHED').astype(int)
    
    print(f"\n✅ Dataset final: {df.shape}")
    
    return df, y


def preprocessar_tudo(data_path='data/'):
    """
    Executa todo o pipeline de pré-processamento
    
    Returns:
        X, y: Features e target prontos para treino
    """
    
    print("=" * 80)
    print("🔧 PIPELINE DE PRÉ-PROCESSAMENTO")
    print("=" * 80)
    
    # 1. Carregar
    orders, stores, payments, channels, hubs, deliveries, drivers = carregar_dados(data_path)
    
    # 2. Merge
    df = fazer_merge(orders, stores, payments, channels, hubs, deliveries, drivers)
    
    # 3. Limpar
    df = limpar_dados(df)
    
    # 4. Features
    df = criar_features(df)
    
    # 5. Selecionar
    df = selecionar_features(df)
    
    # 6. Nulos
    df = tratar_nulos(df)
    
    # 7. Outliers
    df = tratar_outliers(df)
    
    # 8. Codificar
    X, y = codificar_categoricas(df)
    
    print("\n" + "=" * 80)
    print("✅ PRÉ-PROCESSAMENTO CONCLUÍDO!")
    print("=" * 80)
    print(f"Features: {X.shape[1]}")
    print(f"Amostras: {len(X):,}")
    print(f"CANCELED: {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.2f}%)")
    print(f"FINISHED: {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.2f}%)")
    
    return X, y
