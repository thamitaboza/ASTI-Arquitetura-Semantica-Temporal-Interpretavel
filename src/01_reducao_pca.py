import numpy as np
import time
from sklearn.decomposition import PCA

print("INICIANDO REDUÇÃO DE DIMENSIONALIDADE (PCA)")

caminho_matriz_original = 'embeddings_asti_bge_m3.npy'
print(f"Carregando {caminho_matriz_original} para a memória RAM...")
embeddings_1024d = np.load(caminho_matriz_original)

from sklearn.preprocessing import normalize
embeddings_1024d = normalize(embeddings_1024d, norm='l2')

print(f"Matriz original carregada. Formato: {embeddings_1024d.shape} (Filmes x Dimensões)")

n_componentes = 100
pca = PCA(n_components=n_componentes, random_state=42)

print(f"\nAplicando PCA para comprimir de 1024 para {n_componentes} dimensões...")
tempo_inicio = time.time()

embeddings_100d = pca.fit_transform(embeddings_1024d)

tempo_total = time.time() - tempo_inicio
print(f"Transformação concluída em {tempo_total:.2f} segundos!")

variancia_preservada = np.sum(pca.explained_variance_ratio_) * 100
print(f"Informação semântica preservada: {variancia_preservada:.2f}%")

arquivo_saida = 'embeddings_pca_100d.npy'
np.save(arquivo_saida, embeddings_100d)
print(f"Nova matriz salva com sucesso: {arquivo_saida}")
print(f"Formato final: {embeddings_100d.shape}")
print("-" * 55)