from collections.abc import Callable

from datasets import Dataset
from wd_tagger_append.prepare_dataset import PreparationSettings, SplitPlanner
from wd_tagger_append.stratification import build_label_matrix


def _make_dataset() -> Dataset:
    records = [
        {
            "md5": "1",
            "source": "",
            "rating": "general",
            "tags": {"general": ["cat"], "character": []},
            "score": 0,
        },
        {
            "md5": "2",
            "source": "",
            "rating": "general",
            "tags": {"general": ["dog"], "character": []},
            "score": 0,
        },
        {
            "md5": "3",
            "source": "",
            "rating": "sensitive",
            "tags": {"general": ["dog"], "character": ["alice"]},
            "score": 0,
        },
        {
            "md5": "4",
            "source": "",
            "rating": "general",
            "tags": {"general": ["cat"], "character": ["bob"]},
            "score": 0,
        },
        {
            "md5": "5",
            "source": "",
            "rating": "questionable",
            "tags": {"general": ["bird"], "character": []},
            "score": 0,
        },
        {
            "md5": "6",
            "source": "",
            "rating": "general",
            "tags": {"general": ["bird"], "character": ["alice"]},
            "score": 0,
        },
    ]
    return Dataset.from_list(records)


def _silent_reporter() -> Callable[[str], None]:
    return lambda _: None


def test_build_label_matrix_contains_expected_labels() -> None:
    dataset = _make_dataset()
    result = build_label_matrix(dataset)

    assert result.matrix.shape[0] == len(dataset)
    assert result.matrix.shape[1] == len(result.mapping)

    cat_index = result.mapping.label_to_id["cat"]
    alice_index = result.mapping.label_to_id["alice"]
    rating_index = result.mapping.label_to_id["general"]

    assert result.matrix[0, cat_index] == 1
    assert result.matrix[0, alice_index] == 0
    assert result.matrix[0, rating_index] == 1

    sensitive_index = result.mapping.label_to_id["sensitive"]
    assert result.matrix[2, sensitive_index] == 1
    assert result.matrix[1, sensitive_index] == 0


def test_split_planner_adds_label_matrix_and_metadata() -> None:
    dataset = _make_dataset()
    settings = PreparationSettings(train_ratio=0.5, multilabel_stratification=True)
    planner = SplitPlanner(settings, _silent_reporter())

    split_result = planner.split(dataset)
    assert split_result.metadata is not None
    assert split_result.label_mapping is not None

    label_columns = split_result.metadata["label_matrix"]["columns"]
    assert label_columns == len(split_result.label_mapping)

    for split_name in ("train", "validation"):
        split_dataset = split_result.dataset_dict[split_name]
        assert "label_matrix" in split_dataset.column_names
        for row in split_dataset["label_matrix"]:
            assert len(row) == label_columns


def test_split_planner_can_disable_multilabel_stratification() -> None:
    dataset = _make_dataset()
    settings = PreparationSettings(train_ratio=0.5, multilabel_stratification=False)
    planner = SplitPlanner(settings, _silent_reporter())

    split_result = planner.split(dataset)
    assert split_result.metadata is None
    assert split_result.label_mapping is None

    train_dataset = split_result.dataset_dict["train"]
    assert "label_matrix" not in train_dataset.column_names
