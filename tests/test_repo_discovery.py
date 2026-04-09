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


class PaperLinkedGitHub:
    def search_repositories(
        self,
        query: str,
        limit: int = 8,
        max_results_inspected: int = 24,
    ) -> list[RepoCandidate]:
        return [
            RepoCandidate(
                name="paper-code",
                full_name="authors/paper-code",
                url="https://github.com/authors/paper-code",
                stars=3,
                description="Official implementation for the paper",
                owner_login="authors",
            ),
            RepoCandidate(
                name="popular-baseline",
                full_name="community/popular-baseline",
                url="https://github.com/community/popular-baseline",
                stars=900,
                description="Popular unrelated baseline collection",
                owner_login="community",
            ),
        ]


def test_discovery_prioritizes_paper_code_url_candidate() -> None:
    paper = ArxivPaper(
        arxiv_id="1234.5678",
        url="https://arxiv.org/abs/1234.5678",
        title="Paper Title",
        abstract="Benchmark details",
        authors=["Alice Smith"],
        code_url="https://github.com/authors/paper-code",
        code_url_source="arxiv_page",
    )

    ranked = discover_repositories(paper=paper, github=PaperLinkedGitHub())

    assert ranked[0].full_name == "authors/paper-code"
    assert any("paper_code_url" in reason for reason in ranked[0].discovery_reasons)


def test_discovery_collapses_duplicate_paper_code_url_candidate() -> None:
    paper = ArxivPaper(
        arxiv_id="1234.5678",
        url="https://arxiv.org/abs/1234.5678",
        title="Paper Title",
        code_url="https://github.com/authors/paper-code",
        code_url_source="arxiv_page",
    )

    ranked = discover_repositories(paper=paper, github=PaperLinkedGitHub())

    assert [repo.full_name for repo in ranked].count("authors/paper-code") == 1
    paper_repo = next(repo for repo in ranked if repo.full_name == "authors/paper-code")
    assert paper_repo.stars == 3
    assert any(reason == "paper_code_url(+1000)" for reason in paper_repo.discovery_reasons)
