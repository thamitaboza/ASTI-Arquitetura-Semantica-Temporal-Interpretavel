import pandas as pd

df = pd.read_csv('cmu_dataset_com_conceitos.csv')

def inspecionar_temas():
    coluna_titulo = 'title' if 'title' in df.columns else 'wiki_movie_id'
    
    agrupado = df[df['conceito_id'] != -1].groupby(['janela_temporal', 'conceito_id'])
    
    print(f"{'DÉCADA':<10} | {'TEMA ID':<8} | {'FILMES REPRESENTATIVOS (Amostra)'}")
    print("-" * 100)
    
    for (decada, tema_id), grupo in agrupado:
        amostra = ", ".join(grupo[coluna_titulo].head(5).astype(str).tolist())
        print(f"{decada:<10} | {tema_id:<8} | {amostra}...")

if __name__ == "__main__":
    inspecionar_temas()