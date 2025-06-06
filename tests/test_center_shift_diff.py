import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, 'tex-src/scripts')

spec = importlib.util.spec_from_file_location('diff', 'tex-src/scripts/csv_to_center_shift_diff.py')
diff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diff)


def test_calc_center_shift_phase2():
    csv = Path('tex-src/data/prices/1321.csv')
    df = diff.calc_center_shift(diff.read_prices(csv), phase=2)
    assert {'MAE_5d', 'HitRate_20d', 'RelMAE'}.issubset(df.columns)
    assert 0 <= df['HitRate_20d'].iloc[-1] <= 100


def test_process_one(tmp_path):
    csv = Path('tex-src/data/prices/1321.csv')
    out = diff.process_one(csv, out_dir=tmp_path)
    assert out.exists()
    text = out.read_text()
    assert text.strip() != ''
    assert 'Median' in text

