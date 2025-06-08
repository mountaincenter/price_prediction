import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, 'tex-src/scripts')

spec = importlib.util.spec_from_file_location('batch', 'tex-src/scripts/csv_to_event_batch.py')
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def test_compute_metrics():
    csv = Path('tex-src/data/prices/1321.csv')
    raw = batch.read_prices(csv)
    df = batch.calc_event_beta(raw)
    mae, rmae, hit = batch.compute_metrics(df)
    assert mae == df['MAE_5d'].iloc[-1]
    assert rmae == df['RelMAE'].iloc[-1]
    assert 0 <= hit <= 100


def test_make_summary_contains_median():
    rows = [('1321', 100.0, 1.0, 2.0, 50.0)]
    tex = batch.make_summary(rows.copy())
    assert 'Median' in tex


def test_compute_metrics_custom():
    csv = Path('tex-src/data/prices/1321.csv')
    raw = batch.read_prices(csv)
    df = batch.calc_event_beta(
        raw,
        eta=0.02, l_init=0.95, l_min=0.91, l_max=0.99
    )
    mae, rmae, hit = batch.compute_metrics(df)
    assert mae == df['MAE_5d'].iloc[-1]
    assert rmae == df['RelMAE'].iloc[-1]
    assert 0 <= hit <= 100
