import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, 'tex-src/scripts')

spec = importlib.util.spec_from_file_location('diff', 'tex-src/scripts/csv_to_open_price_diff.py')
diff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diff)


def test_calc_open_price_phase2():
    csv = Path('tex-src/data/prices/1321.csv')
    df = diff.calc_open_price(diff.read_prices(csv), phase=5)
    required = {
        'MAE_5d', 'HitRate_20d', 'RelMAE',
        'G_phase0', 'G_phase1', 'G_phase2', 'G_final'
    }
    assert required.issubset(df.columns)
    assert 0 <= df['HitRate_20d'].iloc[-1] <= 100


def test_process_one(tmp_path):
    csv = Path('tex-src/data/prices/1321.csv')
    out = diff.process_one(csv, out_dir=tmp_path)
    assert out.exists()
    text = out.read_text()
    assert text.strip() != ''
    assert text.count('\\begin{threeparttable}') == 3
    assert text.count('\\clearpage') == 2


def test_custom_params():
    csv = Path('tex-src/data/prices/1321.csv')
    df = diff.calc_open_price(
        diff.read_prices(csv), phase=5,
        eta=0.02, l_init=0.95, l_min=0.91, l_max=0.99
    )
    assert df[r'$\lambda_{\text{shift}}$'].iloc[0] == 0.95


def test_make_table_newline():
    csv = Path('tex-src/data/prices/1321.csv')
    df = diff.calc_open_price(diff.read_prices(csv), phase=5)
    tex = diff.make_table(df, title='code:1321')
    lines = tex.splitlines()
    assert lines[0].startswith('\\noindent')
    assert lines[0].endswith('\\')
    assert lines[1] == '\\begingroup'
    assert tex.endswith('\\endgroup\n')
