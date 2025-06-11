import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, 'tex-src/scripts')

spec = importlib.util.spec_from_file_location('fund', 'tex-src/scripts/csv_to_fundamentals_diff.py')
fund = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fund)


def test_make_table_and_process(tmp_path):
    events = Path('tests/fixtures/events.csv')
    out = tmp_path / 'fund.tex'
    fund.process_all(events_csv=events, out_file=out, prices_dir=Path('tests/fixtures/prices'))
    assert out.exists()
    text = out.read_text()
    assert '\\documentclass' in text
    assert '\\begin{tabular}' in text
    assert '1321' in text
    assert any(suffix in text for suffix in ['_J_d', '_T_d', '_U_d-1'])
