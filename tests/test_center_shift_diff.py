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
    assert {
        'Outlier', 'C_ratio', 'S_t,p', 'S_r', 'S_verification',
        r'$\lambda_{\text{shift}}$', r'$\Delta\alpha_t$'
    }.issubset(df.columns)
    assert set(df['Outlier'].unique()) <= set(range(10))
    assert df['C_ratio'].notna().any()
    assert df['S_t,p'].isin([-1, 0, 1]).all()
    assert df['S_verification'].isin([0, 1]).all()
    mask = df['C_ratio'].abs() >= 0.01
    if mask.any():
        assert set(df.loc[mask, 'Outlier'].unique()) <= set(range(2, 10))


def test_event_outlier():
    csv = Path('tex-src/data/prices/1321.csv')
    events = Path('tests/fixtures/events.csv')
    df = diff.calc_center_shift(diff.read_prices(csv), phase=2, events_csv=events)
    assert set(df['Outlier'].unique()) != {0}


def test_process_one(tmp_path):
    csv = Path('tex-src/data/prices/1321.csv')
    out = diff.process_one(csv, out_dir=tmp_path)
    assert out.exists()
    text = out.read_text()
    assert text.strip() != ''
    assert text.count('\\begin{threeparttable}') == 3
    assert text.count('\\clearpage') == 2
    assert 'C_\\Delta/C_r' in text and '\\mathrm{Out}' in text
    assert 'λ = 0.90' in text and 'λ = 0.94' in text and 'λ = 0.98' in text


def test_custom_params():
    csv = Path('tex-src/data/prices/1321.csv')
    df = diff.calc_center_shift(
        diff.read_prices(csv), phase=2,
        eta=0.02, l_init=0.95, l_min=0.91, l_max=0.99
    )
    assert df[r'$\lambda_{\text{shift}}$'].iloc[0] == 0.95




def test_make_table_newline():
    csv = Path('tex-src/data/prices/1321.csv')
    df = diff.calc_center_shift(diff.read_prices(csv), phase=2)
    tex = diff.make_table(df, title='code:1321')
    lines = tex.splitlines()
    assert lines[0].startswith('\\noindent')
    assert lines[0].endswith('\\')
    assert lines[1] == '\\begingroup'
    assert tex.endswith('\\endgroup\n')


def test_calc_center_shift_phase5_ma():
    csv = Path('tex-src/data/prices/1321.csv')
    df = diff.calc_center_shift(diff.read_prices(csv), phase=5)
    assert {'B_ma5', 'B_ma10'}.issubset(df.columns)
    assert df['B_ma5'].notna().any()
    assert df['B_ma10'].notna().any()


def test_calc_center_shift_phase6_base10():
    csv = Path('tex-src/data/prices/1321.csv')
    df = diff.calc_center_shift(diff.read_prices(csv), phase=6)
    idx = df['C_pred'].first_valid_index()
    if idx is not None:
        base = df['B_ma10'].iloc[idx]
        expect = base * (
            1
            + df[r'$\alpha_t$'].iloc[idx]
            * df[r'$\sigma_t^{\mathrm{shift}}$'].iloc[idx]
        )
        assert abs(df['C_pred'].iloc[idx] - expect) < 1e-6

