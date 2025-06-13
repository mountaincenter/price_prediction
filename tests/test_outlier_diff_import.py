import importlib.util
spec = importlib.util.spec_from_file_location('odiff', 'tex-src/scripts/csv_to_outlier_diff.py')
odiff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(odiff)

def test_import():
    assert hasattr(odiff, 'collect_outliers')
