import numpy as np
import time
import umap

print("INICIANDO REDUÇÃO NÃO-LINEAR (UMAP)")

caminho_pca = 'embeddings_pca_100d.npy'
print(f"Carregando {caminho_pca}...")
embeddings_100d = np.load(caminho_pca)

redutor = umap.UMAP(
    n_components=15,
    n_neighbors=30,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

print("\nComprimindo de 100D para 15D... (O UMAP exige muito processamento, aguarde)")
tempo_inicio = time.time()

embeddings_15d = redutor.fit_transform(embeddings_100d)

tempo_total = time.time() - tempo_inicio
print(f"UMAP concluído em {tempo_total:.2f} segundos!")

arquivo_saida = 'embeddings_umap_15d.npy'
np.save(arquivo_saida, embeddings_15d)
print(f"Matriz densa salva com sucesso: {arquivo_saida}")
print("-" * 55)