# RULES: Data Ingestion & DB Integration  (Codex版 2025-06-02)
この `rules.md` が開発・回答に関する唯一のルールソースです。Assistant は常に本ファイルとユーザー指示を照合し、最優先で遵守すること。

## 1. MUST NOT
- `app.db` からの import は禁止。必ず `ultimate_alchemy` で Base & Session を取得。
- ハードコードされたパスは使用しない。必ず utils のヘルパを経由する。
- ループ内でグローバル state を変更しない。
- `if __name__ == '__main__':` 以外で print やデバッグ出力を行わない。

## 2. MUST
1. SQLAlchemy v2 スタイルの API を使用する。
2. Bulk-insert / upsert は `date DESC` で並べ、新しい行を先に挿入する。
3. 1トランザクションは 500 行以下に分割する。
4. CLI 出力には可能であれば `rich` を使用し、利用できない場合は logger で簡易出力。
5. ログは `app.logger` を使用（print 禁止）。
6. CSV の各列は `ColumnMap` でバリデーションしてから DB に入れる。
7. 単体テストの実行は `pytest --sqlite-memory` を推奨。実行できない場合は最小限のユニットテストを用意し、実行方法をドキュメント化する。

### 2.1 Function Contracts
| Function | Responsibility | Returns |
|---------|------------------------------------------------------|--------|
| `upsert_market_data(code, path)` | 価格 CSV を解析し `market_prices` に bulk‑upsert | `int` rows |

### 2.2 Directory & Naming
- 実行例：`python -m ingest.push_market_data 1332 ~/foo.csv`
- Python ファイル・変数は snake_case、DB カラムは camelCase、環境変数は UPPER_SNAKE

### 2.3 Example API Use
```python
from ultimate_alchemy import get_session
from sqlalchemy import insert
from app.models import MarketPrice

with get_session() as sess:
    stmt = (
        insert(MarketPrice)
        .values(rows)
        .on_conflict_do_update(
            index_elements=['date', 'stockCode'],
            set_={c.name: c for c in stmt.excluded if c.name not in ['date', 'stockCode']}
        )
    )
    sess.execute(stmt)
```

## 3. File Header Convention
```
#!/usr/bin/env python3
""" <相対パス>/<file_name>.py
  v<MAJOR>.<MINOR>  (<Created>)
────────────────────────────────────────────────────────
CHANGELOG:
- YYYY-MM-DD  <直近の変更概要>
- YYYY-MM-DD  <過去の変更>
"""
```
- Created は初回作成日から変更しない。
- CHANGELOG は更新日当日のみ。未来・過去日は禁止。
- `vX.Y`：機能追加なら X++、修正なら Y++。
- 既存エントリの改ざん禁止。追記は常に最上段に書く。
- Docstring 以外へ同じ情報を重複させない。

## 4. ChatGPT Customization Rules
- ユーザー指示と本 `rules.md` を最上位に置く。
- 未指定のファイル・関数・変数を変更しない。
- TypeScript では `any` を極力使わない。
- コード提示は基本的に全文を示す。差分提示はユーザーからの要求時のみ。
- 不足情報があれば具体的に列挙してユーザーに確認する。
- 回答は日本語。「はい」「いいえ」で明確に答える。
- 以後、Assistant は本ルールを破らない。

## 5. チェックリスト
1. 最新 `rules.md` とユーザー指示の整合を確認
2. 指示を削らず追加せず反映したか
3. 未来・過去日、自動生成 CHANGELOG を混入していないか
4. 不要な独自判断をしていないか
5. 出力形式が全文か差分かを確認
6. Yes/No 回答が厳密か
7. 日本語で回答しているか

## 6. 運用フロー
1. ルール準拠を確認する
2. 回答形式を決定する（全文 or diff）
3. ルールを守って回答を生成する
4. チェックリストを再確認する
5. 日本語で送信する

## 7. 特に厳守すべき追加ルール
1. 指定されていない箇所の修正・削除は禁止
2. 修正時は必ず全文 or diff で提示
3. 矛盾を検出した場合はユーザーに確認する
4. 本 `rules.md` が最上位判断軸となることを再確認
5. 指示から逸脱しない
