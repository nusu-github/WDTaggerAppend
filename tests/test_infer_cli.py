from inspect import signature

from wd_tagger_append import infer


def test_infer_help_shows_tag_retention_flag() -> None:
    params = signature(infer.infer).parameters
    assert "discard_existing_tags" in params
