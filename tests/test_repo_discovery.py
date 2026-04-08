from execlint.analyzers.repo_discovery import discover_repositories
from execlint.models import ArxivPaper, RepoCandidate


class DummyGitHub:
    def search_repositories(
        self,
        query: str,
        limit: int = 8,
        max_results_inspected: int = 24,
    ) -> list[RepoCandidate]:
        return [
            RepoCandidate(
                name="vision-transformer-official",
                full_name="alice/vision-transformer-official",
                url="https://github.com/alice/vision-transformer-official",
                stars=150,
                description="Official implementation of vision transformer paper",
                owner_login="alice",
            ),
            RepoCandidate(
                name="misc-models",
                full_name="random/misc-models",
                url="https://github.com/random/misc-models",
                stars=1200,
                description="Collection of unrelated baselines",
                owner_login="random",
            ),
        ]


def test_discovery_ranks_by_overlap_author_and_official_wording() -> None:
    paper = ArxivPaper(
        arxiv_id="1234.5678",
        url="https://arxiv.org/abs/1234.5678",
        title="Vision Transformer for Medical Imaging",
        abstract="...",
        authors=["Alice Smith", "Bob Lee"],
    )

    ranked = discover_repositories(paper=paper, github=DummyGitHub())

    assert ranked
    assert ranked[0].full_name == "alice/vision-transformer-official"
    assert ranked[0].discovery_score >= ranked[1].discovery_score
    assert ranked[0].discovery_reasons
