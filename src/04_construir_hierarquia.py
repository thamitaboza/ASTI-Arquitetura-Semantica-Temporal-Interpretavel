
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.semantic_avl import SemanticAVL, SemanticNode, _unit_vector, _validate_vector
from src.rb_tree import SemanticRBTree, SemanticNodeRB


@dataclass
class Cluster:
    key: float
    cluster_data: dict
    embedding: np.ndarray


def load_data(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(data_dir / "cmu_dataset_com_conceitos.csv")
    required = {"janela_temporal", "conceito_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"Colunas obrigatórias ausentes: {required - set(df.columns)}")
    embeddings = np.load(data_dir / "embeddings_asti_bge_m3.npy", mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[1] != 1024:
        raise ValueError("A matriz de embeddings deve ter formato (n, 1024).")

    original_path = data_dir / "cmu_dataset_limpo.csv"
    if original_path.exists():
        original = pd.read_csv(original_path, usecols=["wikipedia_id"])
        if len(original) != len(embeddings):
            raise ValueError("O dataset original e os embeddings têm tamanhos diferentes.")
        if "wikipedia_id" not in df:
            raise ValueError("wikipedia_id é necessário para alinhar os embeddings.")
        for ids in (original["wikipedia_id"], df["wikipedia_id"]):
            if ids.isna().any() or ids.duplicated().any():
                raise ValueError("wikipedia_id deve ser único e não nulo.")
        positions = pd.Index(original["wikipedia_id"]).get_indexer(df["wikipedia_id"])
        if (positions < 0).any():
            raise ValueError("Há filmes que não constam no dataset original.")
        print("Embeddings alinhados por wikipedia_id ao dataset original.")
    elif len(df) == len(embeddings):
        positions = np.arange(len(df))
        print("Alinhamento posicional: pressupõe CSV e embeddings na mesma ordem.")
    else:
        raise ValueError(
            "CSV e embeddings têm tamanhos diferentes. Disponibilize "
            "data/cmu_dataset_limpo.csv para alinhar por wikipedia_id."
        )
    df["_embedding_position"] = positions
    for column in required:
        values = pd.to_numeric(df[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not (values == np.floor(values)).all():
            raise ValueError(f"{column} deve conter apenas números inteiros finitos.")
        df[column] = values
    return df, embeddings


def build_clusters(df: pd.DataFrame, embeddings: np.ndarray) -> list[Cluster]:
    valid = df.loc[df["conceito_id"] != -1]
    clusters = []
    for (decada, conceito_id), group in valid.groupby(
        ["janela_temporal", "conceito_id"], sort=True
    ):
        vectors = embeddings[group["_embedding_position"].to_numpy(dtype=int)]
        if not np.isfinite(vectors).all():
            raise ValueError(f"Embeddings não finitos no grupo {(decada, conceito_id)}.")
        centroid = _validate_vector(np.mean(vectors, axis=0, dtype=np.float64))
        size = len(group)
        clusters.append(Cluster(
            key=float(decada) + (size / 10000.0) + (int(conceito_id) / 1000000.0),
            cluster_data={"janela_temporal": int(decada),
                          "conceito_id": int(conceito_id), "tamanho_do_cluster": size},
            embedding=centroid,
        ))
    if not clusters:
        raise ValueError("Nenhum cluster disponível após remover os ruídos.")
    return clusters


def sequential_search(clusters: list[Cluster], query_vector: np.ndarray) -> Optional[dict]:
    """Baseline com o mesmo cálculo de cosseno usado pelas árvores."""
    query = _unit_vector(_validate_vector(query_vector))
    best_score = -float("inf")
    best_data = None
    for cluster in clusters:
        score = float(np.dot(query, _unit_vector(cluster.embedding)))
        if score > best_score:
            best_score, best_data = score, cluster.cluster_data
    return best_data


def tree_stats(node: Optional[Union[SemanticNode, SemanticNodeRB]]) -> tuple[int, int]:
    """Retorna número real de nós e altura em níveis (árvore vazia = 0)."""
    if node is None:
        return 0, 0
    left_count, left_height = tree_stats(node.left)
    right_count, right_height = tree_stats(node.right)
    return 1 + left_count + right_count, 1 + max(left_height, right_height)


def measure(search: Callable[[np.ndarray], Optional[dict]],
            query_vector: np.ndarray) -> tuple[Optional[dict], float]:
    start = time.perf_counter()
    result = search(query_vector)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def main() -> None:
    df, embeddings = load_data(PROJECT_ROOT / "data")
    clusters = build_clusters(df, embeddings)
    avl = SemanticAVL()
    rb = SemanticRBTree()
    for cluster in clusters:
        avl.insert(cluster.key, cluster.cluster_data, cluster.embedding)
        rb.insert(cluster.key, cluster.cluster_data, cluster.embedding)

    rng = np.random.default_rng(42)
    index = int(rng.integers(len(clusters) // 4,
                             max(len(clusters) // 4 + 1, (3 * len(clusters) + 3) // 4)))
    query_vector = clusters[index].embedding
    baseline_result, baseline_ms = measure(lambda q: sequential_search(clusters, q), query_vector)
    avl_result, avl_ms = measure(avl.semantic_search, query_vector)
    rb_result, rb_ms = measure(rb.semantic_search, query_vector)
    avl_count, avl_height = tree_stats(avl.root)
    rb_count, rb_height = tree_stats(rb.root)

    print(f"\nFilmes: {len(df)} | Ruídos removidos: {(df['conceito_id'] == -1).sum()}")
    print(f"Clusters: {len(clusters)} | Consulta: índice {index}, {clusters[index].cluster_data}")
    header = (f"{'Cenário':<20} {'Número de Nós':>14} {'Altura':>8} "
              f"{'Rotações (RB)':>15} {'Recolorações (RB)':>18} {'Tempo de Busca (ms)':>21}")
    print("\n" + header)
    print("-" * len(header))
    rows = [("Sequencial plana", len(clusters), "—", "—", "—", baseline_ms),
            ("AVL", avl_count, avl_height, "—", "—", avl_ms),
            ("Rubro-Negra", rb_count, rb_height, rb.rotations, rb.recolorings, rb_ms)]
    for name, count, height, rotations, recolorings, elapsed in rows:
        print(f"{name:<20} {count:>14} {height:>8} {rotations:>15} "
              f"{recolorings:>18} {elapsed:>21.6f}")
    print("\nAltura em níveis; contadores acumulados durante a construção.")
    print("Tempos de uma única busca por cenário; as três buscas são exaustivas.")
    collisions = len(clusters) - avl_count
    if collisions:
        print(f"ATENÇÃO: {collisions} inserções substituíram chaves repetidas nas árvores.")
        print("O baseline contém todos os clusters; os conjuntos comparados são diferentes.")
    for name, result in (("Baseline", baseline_result), ("AVL", avl_result), ("RB", rb_result)):
        print(f"Resultado {name}: {result}")


if __name__ == "__main__":
    main()
