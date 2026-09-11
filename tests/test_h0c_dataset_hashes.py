"""H0c: every benchmark file's sha256 matches DESIGN.labkit.md's locked table."""

import pytest

from overlap_bench.dataset_hashes import LOCKED_SHA256, verify_locked_dataset_hashes


def test_h0c_all_nine_files_match_locked_sha256():
    actual = verify_locked_dataset_hashes()
    assert set(actual) == set(LOCKED_SHA256)


def test_narrowing_to_an_unlocked_name_is_an_error_not_a_skipped_check():
    """The failure mode `only` must not have: quietly checking nothing.

    A narrowed check that treats an unrecognised filename as "no such entry,
    nothing to verify" reports success having verified less than the caller
    asked for. Every plausible loose implementation - dict.get with a default,
    intersecting with the locked keys - passes the happy path and fails here.
    """
    with pytest.raises(ValueError, match="not in the locked table"):
        verify_locked_dataset_hashes(only=["2shapes_trian.npz"])


def test_narrowing_verifies_exactly_the_named_files():
    actual = verify_locked_dataset_hashes(only=["2shapes_val.npz"])
    assert actual == {"2shapes_val.npz": LOCKED_SHA256["2shapes_val.npz"]}
