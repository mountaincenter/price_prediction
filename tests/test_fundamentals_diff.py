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
    csv_out = tmp_path / 'fund.csv'
    fund.process_all(events_csv=events, out_file=out, prices_dir=Path('tests/fixtures/prices'), csv_file=csv_out)
    assert out.exists()
    text = out.read_text()
    assert '\\begin{longtable}' in text
    assert '1321' in text
    assert 'Total' in text
    assert any(tag in text for tag in ['\\_J\\_d', '\\_T\\_d', '\\_U\\_d', '\\_U\\_d-1', '\\_U\\_d+1'])
    assert csv_out.exists()
