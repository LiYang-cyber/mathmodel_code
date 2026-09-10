import pandas as pd
import pytest

from mathmodel.common import inspect_frame
from mathmodel.io import load_table


def test_load_and_inspect_csv(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 1, None], "target": [0, 0, 1]}).to_csv(path, index=False)
    frame = load_table(path)
    result = inspect_frame(frame, "target")
    assert result["rows"] == 3
    assert result["missing"]["x"] == 1
    assert result["duplicate_rows"] == 1


def test_unsupported_format(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("x")
    with pytest.raises(ValueError):
        load_table(path)

