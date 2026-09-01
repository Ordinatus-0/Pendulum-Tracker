import hashlib

from scripts.download_pretrained_bob_weights import SHA256, download


def test_download_rejects_a_payload_with_an_unexpected_checksum(monkeypatch, tmp_path):
    class Response:
        def read(self):
            return b"not a YOLO checkpoint"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr("scripts.download_pretrained_bob_weights.urlopen", lambda *_args, **_kwargs: Response())
    try:
        download(tmp_path / "weights.pt")
    except ValueError as error:
        assert SHA256 in str(error)
    else:
        raise AssertionError("An unverified checkpoint was accepted")


def test_checksum_is_a_sha256_digest():
    assert len(SHA256) == hashlib.sha256().digest_size * 2
