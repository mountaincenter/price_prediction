from pathlib import Path
import importlib.util
import sys

sys.path.insert(0, 'tex-src/scripts')

spec = importlib.util.spec_from_file_location('diff', 'tex-src/scripts/csv_to_range_diff.py')
diff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diff)


def test_calc_range():
    csv = Path('tex-src/data/prices/1321.csv')
    df = diff.calc_range(diff.read_prices(csv))
    required = {'MAE_5d', 'HitRate_20d', 'RelMAE', 'B_final'}
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
    df = diff.calc_range(
        diff.read_prices(csv),
        eta=0.02, l_init=0.85, l_min=0.80, l_max=0.99
    )
    assert df['B_final'].iloc[0] >= 0


def test_make_table_newline():
    csv = Path('tex-src/data/prices/1321.csv')
    df = diff.calc_range(diff.read_prices(csv))
    tex = diff.make_table(df, title='code:1321')
    lines = tex.splitlines()
    assert lines[0].startswith('\\noindent')
    assert lines[0].endswith('\\')
    assert lines[1] == '\\begingroup'
    assert tex.endswith('\\endgroup\n')
