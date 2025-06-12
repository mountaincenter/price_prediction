import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, 'tex-src/scripts')

spec = importlib.util.spec_from_file_location('integ', 'tex-src/scripts/csv_to_integrate.py')
integ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integ)


def test_calc_integrate_columns():
    csv = Path('tex-src/data/prices/1321.csv')
    df = integ.calc_integrate(integ.read_prices(csv), code='1321')
    required = {
        'O_p', 'O_r', 'O_diff', 'O_diff/O_r',
        'H_p', 'H_r', 'H_diff', 'H_diff/H_r',
        'L_p', 'L_r', 'L_diff', 'L_diff/Lr',
        'Cl_p', 'Cl_r', 'Cl_diff', 'Cl_diff/Cl_r'
    }
    assert required.issubset(df.columns)
    assert len(df) > 0


def test_make_table_contains_median():
    csv = Path('tex-src/data/prices/1321.csv')
    df = integ.calc_integrate(integ.read_prices(csv), code='1321')
    tex = integ.make_table(df)
    assert 'Median' in tex
