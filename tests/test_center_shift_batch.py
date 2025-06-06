import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, 'tex-src/scripts')

spec = importlib.util.spec_from_file_location('batch', 'tex-src/scripts/csv_to_center_shift_batch.py')
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def test_compute_metrics():
    csv = Path('tex-src/data/prices/1321.csv')
    raw = batch.read_prices(csv)
    df = batch.calc_center_shift(raw, phase=2)
    mae, rmae, hit = batch.compute_metrics(df)
    assert mae == df['MAE_5d'].iloc[-1]
    assert rmae == df['RelMAE'].iloc[-1]
    assert 0 <= hit <= 100

def test_make_summary_contains_median():
    rows = [('1321', 100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)]
    tex = batch.make_summary(rows.copy())
    assert 'Median' in tex
