import httpx
import pytest

from anime_mlops.config import MalSourceConfig
from anime_mlops.data.sources import mal

COMMIT = "a" * 40
CSV = b"id,title,mean\n1,Cowboy Bebop,8.75\n5,Trigun,8.22\n"


@pytest.fixture
def config():
    return MalSourceConfig(repo="owner/repo", file_path="data/anime.csv")


def make_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_download_writes_file_named_after_commit(config, tmp_path):
    requested = []

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(200, content=CSV)

    snapshot = mal.download_snapshot(COMMIT, config, tmp_path, make_client(handler))

    assert requested == [f"https://raw.githubusercontent.com/owner/repo/{COMMIT}/data/anime.csv"]
    assert snapshot.path.name == f"anime-standalone_{COMMIT}.csv"
    assert snapshot.path.read_bytes() == CSV
    assert snapshot.commit == COMMIT
    assert snapshot.size_bytes == len(CSV)
    assert len(snapshot.sha256) == 64


def test_download_is_idempotent(config, tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=CSV)

    client = make_client(handler)
    first = mal.download_snapshot(COMMIT, config, tmp_path, client)
    second = mal.download_snapshot(COMMIT, config, tmp_path, client)

    assert len(calls) == 1
    assert first == second


@pytest.mark.parametrize("version", ["main", "abc123", "A" * 40, ""])
def test_non_reproducible_version_is_rejected(config, tmp_path, version):
    client = make_client(lambda request: httpx.Response(200, content=CSV))

    with pytest.raises(mal.MalSourceError, match="Version MAL invalide"):
        mal.download_snapshot(version, config, tmp_path, client)


def test_http_error_raises_and_leaves_no_file(config, tmp_path):
    client = make_client(lambda request: httpx.Response(404))

    with pytest.raises(mal.MalSourceError, match="Téléchargement MAL impossible"):
        mal.download_snapshot(COMMIT, config, tmp_path, client)

    assert not list(tmp_path.rglob("*.csv"))


def test_empty_file_is_rejected(config, tmp_path):
    client = make_client(lambda request: httpx.Response(200, content=b"  \n"))

    with pytest.raises(mal.MalSourceError, match="vide"):
        mal.download_snapshot(COMMIT, config, tmp_path, client)


def test_load_snapshot_returns_raw_dataframe(config, tmp_path):
    client = make_client(lambda request: httpx.Response(200, content=CSV))
    snapshot = mal.download_snapshot(COMMIT, config, tmp_path, client)

    data = mal.load_snapshot(snapshot)

    assert list(data.columns) == ["id", "title", "mean"]
    assert len(data) == 2


def test_resolve_latest_commit(config):
    def handler(request):
        assert request.url.params["path"] == "data/anime.csv"
        return httpx.Response(200, json=[{"sha": COMMIT}])

    assert mal.resolve_latest_commit(config, make_client(handler)) == COMMIT


def test_resolve_latest_commit_fails_explicitly(config):
    client = make_client(lambda request: httpx.Response(200, json=[]))

    with pytest.raises(mal.MalSourceError, match="dernier commit"):
        mal.resolve_latest_commit(config, client)
