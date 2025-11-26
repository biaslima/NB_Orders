data_path='data/'

    
orders = pd.read_csv(f"{data_path}orders.csv", encoding="latin-1")
stores = pd.read_csv(f"{data_path}stores.csv", encoding="latin-1")
payments = pd.read_csv(f"{data_path}payments.csv", encoding="latin-1")
channels = pd.read_csv(f"{data_path}channels.csv", encoding="latin-1")
hubs = pd.read_csv(f"{data_path}hubs.csv", encoding="latin-1")
deliveries = pd.read_csv(f"{data_path}deliveries.csv", encoding="latin-1")
drivers = pd.read_csv(f"{data_path}drivers.csv", encoding="latin-1")

df = orders.merge(stores, on='store_id', how='left')
df = df.merge(hubs, on='hub_id', how='left')
df = df.merge(channels, on='channel_id', how='left')

payments_agg = payments.groupby('payment_order_id').agg({
    'payment_amount': 'sum',
    'payment_method': 'first'
}).rename(columns={
    'payment_amount': 'total_payment_amount',
    'payment_method': 'main_payment_method'
}).reset_index()
payments_agg.rename(columns={'payment_order_id': 'order_id'}, inplace=True)
    
df = df.merge(payments_agg, on='order_id', how='left')
    
deliveries_clean = deliveries[['delivery_order_id', 'driver_id', 'delivery_distance_meters']].copy()
deliveries_clean = deliveries_clean.drop_duplicates(subset=['delivery_order_id'], keep='last')
deliveries_clean.rename(columns={'delivery_order_id': 'order_id'}, inplace=True)
df = df.merge(deliveries_clean, on='order_id', how='left')

df = df.merge(drivers, on='driver_id', how='left')
    
df['has_driver'] = df['driver_id'].notna().astype(int)
