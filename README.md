# horse_racing_ai

競馬の予想候補を提示するための個人用プロジェクトです。自動購入は行わず、回収率と安定性を重視した「買い候補」の判断材料を作ることを目的にしています。

現状は、netkeibaのレース結果ページとオッズAPIから過去レース情報を取得し、JSONとして保存したうえでSQLiteへ正規化して投入するデータ収集基盤が中心です。

## 目的

- 複勝、ワイド、3連複の順に予測・推奨対象を広げる
- まずは各馬の3着内確率を扱う
- 最終的には日付単位で複数レースを予測する
- 出力は人が確認しやすい形式にする
- 自動購入はしない

## 現在できること

- レース一覧ページから`race_id`を抽出
- 各レースの結果ページからレース情報と出走馬情報を取得
- 複勝、ワイド、3連複のオッズをJSONP APIから取得
- 取得データを外部SSDの`D:\horse_racing_ai\data\raw\<race_id>.json`へ保存
- 正規化したデータをSQLiteの`D:\horse_racing_ai\data\hr.db`へUPSERT
- ログを`logs/scrape_to_db.log`へ出力

## ディレクトリ構成

```text
horse_racing_ai/
  logs/
  scripts/
    scrape_to_db.py
  src/
    common/
    data/
    pipelines/
    preprocess/
    scrape/
  schema.sql

D:\horse_racing_ai\data\
  hr.db
  raw/
  feature/
  model/
```

## データベース

DBはSQLiteです。スキーマは`schema.sql`に定義されています。

主なテーブル:

- `race`: レース単位の基本情報
- `runner`: 各レースの出走馬と結果
- `place_odds`: 複勝オッズ
- `wide_odds`: ワイドオッズ
- `trio_odds`: 3連複オッズ

主な正規化内容:

- 競馬場、馬場、天候、性別、所属などをID化
- 発走時刻を分単位に変換
- 走破タイムを秒に変換
- 着差、コーナー通過順、馬体重増減を数値化

## セットアップ

Python環境を用意し、実行に必要なライブラリをインストールします。依存関係ファイルはまだありません。

現在コード上で使われている主な外部ライブラリ:

- `requests`
- `beautifulsoup4`
- `lxml`

例:

```bash
pip install requests beautifulsoup4 lxml
```

DBを初期化する場合:

Windows:

```powershell
sqlite3 D:\horse_racing_ai\data\hr.db ".read schema.sql"
```

WSL:

```bash
sqlite3 /mnt/d/horse_racing_ai/data/hr.db < schema.sql
```

データ保存先を変更したい場合は、環境変数`HORSE_RACING_DATA_ROOT`で上書きできます。

## 使い方

レース一覧ページのURLを指定してスクレイピングします。

```bash
python scripts/scrape_to_db.py --url "<netkeibaのレース一覧URL>"
```

自動で次ページへ進める場合:

```bash
python scripts/scrape_to_db.py --url "<netkeibaのレース一覧URL>" --mode auto
```

DB内のレース数が指定値以上になったら開始せず終了する場合:

```bash
python scripts/scrape_to_db.py --url "<netkeibaのレース一覧URL>" --limit 1000
```

## 実装メモ

- パス定義は`src/data/paths.py`にあります
- データ保存先のデフォルトはWindowsでは`D:\horse_racing_ai\data`、WSL/Linuxでは`/mnt/d/horse_racing_ai/data`です
- スクレイピング処理は`src/scrape/`配下に分かれています
- 正規化処理は`src/preprocess/normalizers.py`に集約されています
- SQLiteへの投入処理は`src/data/database.py`にあります
- `src/pipelines/scrape_to_db.py`は現時点では空で、実行入口は`scripts/scrape_to_db.py`です

## 今後の予定

- 予測用特徴量の作成
- 学習・評価パイプラインの再構築
- 複勝の3着内確率予測
- ワイド、3連複への拡張
- EARLY/LATEモードの整理
- 日付単位の予測出力
- 推奨結果のファイル出力

想定する予測出力項目:

- `race_id`
- `race_date`
- `track`
- `race_number`
- `mode`
- `horse_number`
- `horse_name`
- `odds`
- `predicted_probability`

## 注意

スクレイピング対象サイトの仕様変更により、HTMLのクラス名やAPIレスポンス形式が変わると取得処理が失敗する可能性があります。アクセス間隔を空け、過度なリクエストにならないように運用してください。
