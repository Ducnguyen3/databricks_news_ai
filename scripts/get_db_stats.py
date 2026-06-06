import sqlite3
import os
from collections import Counter

def get_stats():
    db_path = "data/chroma/chroma.sqlite3"
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Count articles
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT string_value) FROM embedding_metadata WHERE key='article_id'")
    total_articles = cursor.fetchone()[0]

    # Count chunks
    cursor.execute("SELECT COUNT(*) FROM embeddings")
    total_chunks = cursor.fetchone()[0]

    # Count images
    cursor.execute("SELECT COUNT(*) FROM embedding_metadata WHERE key='has_images' AND bool_value=1")
    chunks_with_images = cursor.fetchone()[0]

    # Distribution of sources
    cursor.execute("SELECT string_value, COUNT(*) FROM embedding_metadata WHERE key='source' GROUP BY string_value")
    source_distribution = dict(cursor.fetchall())

    # Distribution of topics
    cursor.execute("SELECT string_value, COUNT(*) FROM embedding_metadata WHERE key='primary_topic' GROUP BY string_value")
    topic_distribution = dict(cursor.fetchall())

    print(f"Total articles: {total_articles}")
    print(f"Total chunks: {total_chunks}")
    print(f"Chunks with images: {chunks_with_images}")
    print("Sources distribution (chunks):")
    for src, cnt in source_distribution.items():
        print(f"  - {src}: {cnt}")
    print("Topics distribution (chunks):")
    for tpc, cnt in topic_distribution.items():
        print(f"  - {tpc}: {cnt}")

    conn.close()

if __name__ == '__main__':
    get_stats()
