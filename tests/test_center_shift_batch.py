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