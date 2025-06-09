import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, 'tex-src/scripts')

spec = importlib.util.spec_from_file_location('diff', 'tex-src/scripts/csv_to_event_diff.py')
diff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diff)


def test_beta_weekday_range():
    csv = Path('tex-src/data/prices/1321.csv')
    df = diff.read_prices(csv)
    beta = diff.compute_beta_weekday(df['Date'])
    assert beta.min() >= 0.8
    assert beta.max() <= 1.2


def test_beta_earn_range():
    csv = Path('tex-src/data/prices/1321.csv')
    df = diff.read_prices(csv)
    earn = diff.load_earn_dates('1321')
    beta = diff.compute_beta_earn(df['Date'], earn)
    assert beta.min() >= 0.8
    assert beta.max() <= 1.5


def test_beta_market_range():
    csv = Path('tex-src/data/prices/1321.csv')
    df = diff.read_prices(csv)
    idx = {
        'topix': diff.load_index('topix'),
        'sp500': diff.load_index('sp500'),
        'usd_jpy': diff.load_index('usd_jpy'),
        'nikkei225_vi': diff.load_index('nikkei225_vi'),
    }
    beta = diff.compute_beta_market(df['Date'], df['Close'], idx)
    assert beta.min() >= 0.8
    assert beta.max() <= 1.2
