import matplotlib.pyplot as plt

from mathmodel.common import save_figure


def test_save_figure_exports_raster_and_vector_formats(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    save_figure(fig, tmp_path, "result")
    assert (tmp_path / "result.png").is_file()
    assert (tmp_path / "result.svg").is_file()
    assert (tmp_path / "result.pdf").is_file()
    assert (tmp_path / "result.svg").read_text(encoding="utf-8").lstrip().startswith("<?xml")
    assert (tmp_path / "result.pdf").read_bytes().startswith(b"%PDF")
