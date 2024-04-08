from langchain.embeddings.base import Embeddings
from embedding_tpu.text2vec.sentence_model import SentenceModel
from typing import List


class Word2VecEmbedding(Embeddings):
    model = SentenceModel(device='tpu')

    def embed_query(self, text: str) -> List[float]:
        embeddings_tpu = self.model.encode_tpu([text, "", "", ""])
        return embeddings_tpu.tolist()[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings_tpu = self.model.encode_tpu(texts)
        return embeddings_tpu.tolist()

