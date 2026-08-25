import pandas as pd
import numpy as np
import hdbscan
from sklearn.cluster import KMeans
import time

print("INICIANDO JANELAMENTO E EXTRAÇÃO DE CONCEITOS")

df = pd.read_csv('cmu_dataset_limpo.csv')
embeddings = np.load('embeddings_umap_15d.npy')

df['ano_calculado'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
mascara_validos = df['ano_calculado'].notna()

df = df[mascara_validos].copy()
embeddings = embeddings[mascara_validos]

df['janela_temporal'] = (df['ano_calculado'].astype(int) // 10) * 10
janelas = sorted(df['janela_temporal'].unique())

print(f"Dataset particionado em {len(janelas)} janelas temporais (décadas).")

df['conceito_id'] = -1
df['metodo_cluster'] = 'none'

min_cluster_size = 10

for janela in janelas:
    mascara_janela = (df['janela_temporal'] == janela)
    qtd_filmes = mascara_janela.sum()
    
    if qtd_filmes < min_cluster_size * 2:
        print(f"Janela {janela}: Ignorada (Apenas {qtd_filmes} filmes, insuficiente para clusters)")
        continue
        
    vetores_janela = embeddings[mascara_janela]
    
    tamanho_minimo_tema = max(15, int(qtd_filmes * 0.005))
    
    rigor_relevo = 5
    
    print(f"\nProcessando Janela {janela} ({qtd_filmes} filmes)...")
    print(f"  -> Exigindo min_cluster_size={tamanho_minimo_tema} e min_samples={rigor_relevo}")
    
    
    clusterizador = hdbscan.HDBSCAN(
        min_cluster_size=tamanho_minimo_tema, 
        min_samples=rigor_relevo, 
        metric='euclidean', 
        cluster_selection_method='leaf'
    )
    labels = clusterizador.fit_predict(vetores_janela)
    
    ruido = list(labels).count(-1)
    taxa_ruido = ruido / qtd_filmes
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    print(f"  -> HDBSCAN encontrou {num_clusters} temas. Taxa de ruído: {taxa_ruido*100:.1f}%")
    
    
    if taxa_ruido > 0.60 or num_clusters < 2:
        print(f"  -> [ALERTA] Janela instável detectada. Acionando Fallback (K-Means)...")
        k_otimo = max(2, int(np.sqrt(qtd_filmes / 2))) 
        kmeans = KMeans(n_clusters=k_otimo, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(vetores_janela)
        
        df.loc[mascara_janela, 'conceito_id'] = labels
        df.loc[mascara_janela, 'metodo_cluster'] = 'kmeans'
        print(f"  -> K-Means aplicou {k_otimo} temas forçados.")
    else:
        df.loc[mascara_janela, 'conceito_id'] = labels
        df.loc[mascara_janela, 'metodo_cluster'] = 'hdbscan'

df.to_csv('cmu_dataset_com_conceitos.csv', index=False)
print("\n" + "-" * 55)
print("FASE 2 CONCLUÍDA! O arquivo 'cmu_dataset_com_conceitos.csv' foi gerado.")