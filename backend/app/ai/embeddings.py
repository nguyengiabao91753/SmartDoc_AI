# from sentence_transformers import SentenceTransformer
# from app.core.config import settings
# import numpy as np

# class Embedder:
#     def __init__(self, device: str = "cpu"):
#         self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device=device)
#         self.dim = self.model.get_sentence_embedding_dimension()

#     def embed_texts(self, texts: list):
#         # returns numpy array (N, dim) float32, normalized
#         emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
#         # normalize to unit vectors for cosine via inner product
#         norms = np.linalg.norm(emb, axis=1, keepdims=True)
#         norms[norms == 0] = 1.0
#         emb = emb / norms
#         return emb.astype('float32')