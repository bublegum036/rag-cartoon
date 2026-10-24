"""Provider-independent values. No database or model SDK imports belong here."""

from dataclasses import dataclass
from enum import Enum


class Status(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT = "insufficient"
    CLARIFY = "clarify"


class Intent(str, Enum):
    SEARCH = "search"
    LOOKUP = "lookup"
    COMPARE_DURATION = "compare_duration"
    COUNT = "count"


@dataclass(frozen=True)
class Film:
    movie_id: str
    title: str
    year: int
    duration_minutes: int
    version: int
    active: bool
    source_id: str
    synopsis: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.active, bool):
            raise ValueError(f"Film {self.movie_id}: active must be a boolean")
        if not all((self.movie_id, self.title, self.source_id, self.synopsis)):
            raise ValueError("Film identifiers, title and synopsis must be nonempty")
        if self.duration_minutes <= 0 or self.version <= 0:
            raise ValueError("Duration and version must be positive")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    film: Film
    text: str

    @property
    def embedding_text(self) -> str:
        return f"{self.film.title}\n{self.text}"


@dataclass(frozen=True)
class Filters:
    duration_minutes_lt: int | None = None
    duration_minutes_lte: int | None = None
    movie_ids: tuple[str, ...] = ()

    def matches(self, film: Film) -> bool:
        return (
            film.active
            and (self.duration_minutes_lt is None
                 or film.duration_minutes < self.duration_minutes_lt)
            and (self.duration_minutes_lte is None
                 or film.duration_minutes <= self.duration_minutes_lte)
            and (not self.movie_ids or film.movie_id in self.movie_ids)
        )


@dataclass(frozen=True)
class QueryPlan:
    semantic_query: str
    intent: Intent = Intent.SEARCH
    filters: Filters = Filters()
    time_budget_minutes: int | None = None
    early_status: Status | None = None
    early_text: str = ""

    def __post_init__(self) -> None:
        if self.early_status is Status.ANSWERED:
            raise ValueError("An answered result requires evidence")
        if self.early_status is not None and not self.early_text:
            raise ValueError("Early response requires text")
        if self.intent is Intent.COMPARE_DURATION:
            if len(self.filters.movie_ids) != 1 or self.time_budget_minutes is None:
                raise ValueError("Duration comparison requires one film and a budget")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class EmbeddingIdentity:
    provider: str
    model: str
    revision: str
    dimensions: int
    implementation_version: str = ""
    encoding: str = ""

    def __post_init__(self) -> None:
        if not all((self.provider, self.model, self.revision)) or self.dimensions < 1:
            raise ValueError("Embedding identity requires provider, model, revision and positive dimensions")


@dataclass(frozen=True)
class EmbeddingBatch:
    identity: EmbeddingIdentity
    vectors: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class PublicationResult:
    index_name: str
    snapshot_id: str
    embedding: EmbeddingIdentity
    catalog_sha256: str
    chunks_sha256: str
    movies: int
    chunks: int
    schema_version: int = 1


@dataclass(frozen=True)
class Evidence:
    """Selected chunks grouped into one card per movie."""

    film: Film
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True)
class MovieMatch:
    film: Film
    chunks: tuple[Chunk, ...]
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    plan: QueryPlan
    retrieved: tuple[Hit, ...]
    movies: tuple[MovieMatch, ...]

    @property
    def context(self) -> tuple[Evidence, ...]:
        return tuple(Evidence(movie.film, movie.chunks) for movie in self.movies)


@dataclass(frozen=True)
class ExactResult:
    """Computed by a catalog adapter/code, never guessed by a language model."""

    value: int
    unit: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class Recommendation:
    movie_id: str
    title: str
    reason: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class Answer:
    status: Status
    text: str
    source_ids: tuple[str, ...] = ()
    recommendations: tuple[Recommendation, ...] = ()


@dataclass(frozen=True)
class Trace:
    question: str
    plan: QueryPlan
    retrieved: tuple[Hit, ...]
    context: tuple[Evidence, ...]
    exact_result: ExactResult | None
    answer: Answer


class RetrievalUnavailable(RuntimeError):
    """Infrastructure failure; must not become an insufficient answer."""


class EmbeddingUnavailable(RuntimeError):
    """Embedding dependency, weight download or inference failed."""


class GroundingError(ValueError):
    """Generated references or retrieved metadata violate the contract."""
