# price_prediction

本リポジトリには株価 CSV を処理して各種 LaTeX テーブルを生成する
スクリプト群が含まれています。以下では依存関係の導入方法と、各スクリ
プトの使い方をまとめます。

## Setup

依存パッケージは `requirements.txt` で管理しています。仮想環境を作成
してインストールしてください。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Input CSV

`tex-src/data/prices/` 配下に銘柄コード名の CSV を置きます
（例: `8801.csv`）。各スクリプトは引数で明示しない場合、このディレク
トリを自動的に参照します。

`tex-src/data/earn/perpbr/` には東証が公開する月次の PER/PBR Excel (`perpbrYYYYMM.xlsx`) を配置します。`xlsx_to_csv_perpbr.py` を実行すると、未処理分をまとめて `tex-src/data/earn/perpbr.csv` に追記します。

```bash
python tex-src/scripts/xlsx_to_csv_perpbr.py
```


## Scripts

### `csv_to_tex.py`

指定した CSV を読み込み、Open/High/Low/Close 等を LaTeX の `tabular`
形式に変換します。

```bash
python tex-src/scripts/csv_to_tex.py tex-src/data/prices/8801.csv
```

`--out/-o` で出力 `.tex` ファイルを指定できます。省略時は同名 `.tex`
を生成します。

### `csv_to_basic_form_tex.py`

基本形 Phase‑0 の予測結果を含むテーブルを作成します。出力は
`tex-src/data/analysis/basic_form/` に `<code>.tex` として保存されます。

```bash
python tex-src/scripts/csv_to_basic_form_tex.py 8801.csv
```

### `csv_to_center_shift.py`

終値を用いた Center Shift 指標を計算し、直近 63 営業日のテーブルを
`tex-src/data/analysis/center_shift/` に出力します。

```bash
python tex-src/scripts/csv_to_center_shift.py 8801.csv
```

### `csv_to_center_shift_diff.py`

Center Shift から派生する差分テーブル (30 日 + 平均) を生成します。
CSV を省略すると `data/prices` 内の全銘柄を処理します。

```bash
python tex-src/scripts/csv_to_center_shift_diff.py 8801.csv
```

### `csv_to_center_shift_batch.py`

`data/prices` 内の全 CSV を対象に `csv_to_center_shift_diff.py` を実行
し、指標サマリ `summary.tex` も合わせて生成します。

```bash
python tex-src/scripts/csv_to_center_shift_batch.py
```


### `csv_to_open_price_diff.py`

始値ギャップを用いた差分テーブルを生成します。出力先は
`tex-src/data/analysis/open_price/` です。

```bash
python tex-src/scripts/csv_to_open_price_diff.py 8801.csv
```

### `csv_to_open_price_batch.py`

`data/prices` 内の全 CSV について始値版 diff テーブルを作成し、
`summary.tex` を同ディレクトリに保存します。

```bash
python tex-src/scripts/csv_to_open_price_batch.py
```

### `backtest_open_price.tex`

生成した diff テーブルをまとめたバックテスト用 LaTeX です。

```bash
pdflatex tex-src/backtest_open_price.tex
```

## Testing

依存パッケージをインストールした上で `pytest` を実行します。

```bash
pip install -r requirements.txt
pytest -q
```

