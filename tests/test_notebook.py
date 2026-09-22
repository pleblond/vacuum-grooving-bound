"""Execute the Lee2026 notebook's code cells headlessly (matplotlib Agg)."""

import json
from pathlib import Path


def test_notebook_cells_execute():
    nb_path = Path(__file__).resolve().parents[1] / "examples" / "Lee2026_Fig3b.ipynb"
    nb = json.loads(nb_path.read_text())
    assert nb["nbformat"] == 4
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) >= 4
    namespace: dict = {}
    for i, cell in enumerate(code):
        src = "".join(cell["source"])
        exec(compile(src, f"<notebook-cell-{i}>", "exec"), namespace)
    assert "df" in namespace and len(namespace["df"]) > 100
