"""H0c: every benchmark file's sha256 matches DESIGN.labkit.md's locked table."""

from overlap_bench.dataset_hashes import LOCKED_SHA256, verify_locked_dataset_hashes


def test_h0c_all_nine_files_match_locked_sha256():
    actual = verify_locked_dataset_hashes()
    assert set(actual) == set(LOCKED_SHA256)
