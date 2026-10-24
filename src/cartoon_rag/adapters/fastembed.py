"""Local multilingual E5 with pinned weights and explicit retrieval prefixes."""

from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from cartoon_rag.domain import EmbeddingBatch, EmbeddingIdentity, EmbeddingUnavailable


MODEL = "intfloat/multilingual-e5-small"
DIMENSIONS = 384


class E5Embedder:
    def __init__(self, *, revision: str, cache: Path, threads: int = 2) -> None:
        try:
            from fastembed import TextEmbedding
            from fastembed.common.model_description import ModelSource, PoolingType
            from huggingface_hub import snapshot_download
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise EmbeddingUnavailable(
                'Install search dependencies: python -m pip install -e ".[search]"'
            ) from exc
        try:
            directory = snapshot_download(
                repo_id=MODEL, revision=revision, cache_dir=str(cache),
                allow_patterns=["onnx/model.onnx", "tokenizer.json", "tokenizer_config.json",
                                "special_tokens_map.json", "config.json", "sentencepiece.bpe.model"],
            )
            if MODEL not in {item["model"] for item in TextEmbedding.list_supported_models()}:
                TextEmbedding.add_custom_model(
                    model=MODEL, pooling=PoolingType.MEAN, normalization=True,
                    sources=ModelSource(hf=MODEL), dim=DIMENSIONS, model_file="onnx/model.onnx",
                )
            self.model = TextEmbedding(
                model_name=MODEL, specific_model_path=directory,
                local_files_only=True, threads=threads,
            )
            self.tokenizer = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
            self.tokenizer.no_truncation()
        except Exception as exc:
            raise EmbeddingUnavailable("Cannot load E5 weights; check network and MODEL_CACHE") from exc
        self.identity = EmbeddingIdentity(
            provider="fastembed", model=MODEL, revision=revision,
            dimensions=DIMENSIONS, implementation_version=version("fastembed"),
            encoding="e5-query-passage-v1",
        )

    def _embed(self, texts: Sequence[str], prefix: str) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        inputs = [f"{prefix}: {text}" for text in texts]
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text")
        if any(len(self.tokenizer.encode(text).ids) > 512 for text in inputs):
            raise ValueError("E5 input exceeds 512 tokens; reduce CHUNK_CHARS or shorten the query")
        try:
            return tuple(tuple(float(value) for value in vector)
                         for vector in self.model.embed(inputs, batch_size=32))
        except Exception as exc:
            raise EmbeddingUnavailable("E5 inference failed") from exc

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        return EmbeddingBatch(self.identity, self._embed(texts, "passage"))

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed([text], "query")[0]
