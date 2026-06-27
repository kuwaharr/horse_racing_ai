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
- 単勝、複勝、ワイド、3連複のオッズをJSONP APIから取得
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
- `win_odds`: 単勝オッズ
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
- `pandas`
- `pyarrow` または `fastparquet`
- `lightgbm`
- `catboost`

Windows TerminalのPowerShellで実行する例:

```powershell
pip install requests beautifulsoup4 lxml pandas pyarrow lightgbm catboost
```

DBを初期化する場合:

```powershell
sqlite3 D:\horse_racing_ai\data\hr.db ".read schema.sql"
```

データ保存先を変更したい場合は、環境変数`HORSE_RACING_DATA_ROOT`で上書きできます。

```powershell
$env:HORSE_RACING_DATA_ROOT = "D:\horse_racing_ai\data"
```

## 使い方

レース一覧ページのURLを指定してスクレイピングします。

```powershell
python scripts\scrape_to_db.py --url "<netkeibaのレース一覧URL>"
```

自動で次ページへ進める場合:

```powershell
python scripts\scrape_to_db.py --url "<netkeibaのレース一覧URL>" --mode auto
```

DB内のレース数が指定値以上になったら開始せず終了する場合:

```powershell
python scripts\scrape_to_db.py --url "<netkeibaのレース一覧URL>" --limit 1000
```

DBの収集状況と品質を確認する場合:

```powershell
python scripts\check_db_quality.py
```

別のDBファイルを確認する場合:

```powershell
python scripts\check_db_quality.py --db "D:\horse_racing_ai\data\hr.db"
```

複勝3着内予測用の学習データセットをParquetで作成する場合:

```powershell
python scripts\build_place_top3_dataset.py --kind training --engine fastparquet
```

評価用の人気・複勝オッズデータをParquetで作成する場合:

```powershell
python scripts\build_place_top3_dataset.py --kind eval-odds --engine fastparquet
```

学習データセットのデフォルト出力先は`D:\horse_racing_ai\data\feature\place_top3_dataset.parquet`です。
評価用オッズデータのデフォルト出力先は`D:\horse_racing_ai\data\feature\place_top3_eval_odds.parquet`です。
過去成績特徴量には、馬・騎手・調教師の過去成績、馬や騎手の競馬場/芝ダート/距離帯別成績、直近フォーム、前走からの日数・距離変化、過去平均との差による距離・斤量適性、同一レース内での過去成績順位や平均との差分が含まれます。

出力先を指定する場合:

```powershell
python scripts\build_place_top3_dataset.py --kind training --engine fastparquet --output "D:\horse_racing_ai\data\feature\place_top3_dataset.parquet"
```

学習データセットには人気、オッズ、馬体重など取得タイミングが遅い情報を含めません。複勝オッズや人気は、ベースライン評価と回収率計算用の`eval-odds`データセットに分離します。

人気順・複勝オッズ順のベースラインを評価する場合:

```powershell
python scripts\evaluate_baselines.py --engine fastparquet
```

このベースライン評価は、人気と複勝オッズを使うため評価用オッズデータセット向けです。

リークのない学習データセットでLightGBMを学習し、評価時だけ複勝オッズを使って回収率を確認する場合:

```powershell
python scripts\evaluate_place_top3_lgbm.py --engine fastparquet
```

このLightGBM評価では、学習特徴量に人気、オッズ、馬体重を含めません。複勝オッズはテスト期間の回収率計算にだけ使います。
出力には上位選択、しきい値、予測確率帯、複勝オッズ帯、期待値帯ごとの評価が含まれます。

買い条件のしきい値を指定する場合:

```powershell
python scripts\evaluate_place_top3_lgbm.py --engine fastparquet --pred-thresholds "0.3,0.4,0.5" --ev-thresholds "1.0,1.1,1.2"
```

予測確率と複勝オッズ帯の交差条件を評価する場合、少なすぎる買い目を除外する最低件数を指定できます。

```powershell
python scripts\evaluate_place_top3_lgbm.py --engine fastparquet --min-rule-selections 100
```

複数の時系列foldでルールの安定性を確認する場合:

```powershell
python scripts\evaluate_place_top3_walk_forward.py --engine fastparquet
```

walk-forward評価では、ルール全体の安定性に加えて、開催場、芝/ダート、距離帯、頭数帯ごとの成績も確認できます。

CatBoostで同じwalk-forward評価を行う場合:

```powershell
python scripts\evaluate_place_top3_catboost.py --engine fastparquet
```

CatBoostの有望な固定ルールを詳細評価する場合:

```powershell
python scripts\evaluate_fixed_place_top3_catboost_rule.py --engine fastparquet --pred-min 0.40
```

現時点の代表結果は、CatBoostで`pred_top3>=0.40`、複勝オッズ中間値`[3.0,5.0)`、距離`[1800,2200)`を買う条件です。walk-forward 4 foldで129レース、149点、63的中、的中率42.28%、複勝オッズ中間値ベース回収率154.53%でした。

CatBoostのwalk-forward予測を保存し、重い再学習を避けて買い条件だけを高速に検証する場合:

```powershell
python scripts\generate_catboost_predictions.py --engine fastparquet
python scripts\evaluate_fixed_rule_from_predictions.py --engine fastparquet --pred-min 0.40
```

保存済み予測のデフォルト出力先は`D:\horse_racing_ai\data\model\catboost_place_top3_predictions.parquet`です。例えば開催場`3,7,10`を除外する条件は次のように確認できます。

```powershell
python scripts\evaluate_fixed_rule_from_predictions.py --engine fastparquet --pred-min 0.40 --exclude-track-ids "3,7,10"
```

この条件はwalk-forward 4 foldで119レース、138点、60的中、的中率43.48%、複勝オッズ中間値ベース回収率159.17%でした。より絞るなら開催場`3,7,8,10`除外で82レース、93点、43的中、回収率168.28%でしたが、買い点数が少なくなるため主候補は`3,7,10`除外とします。

保存済み予測から、最低買い点数を指定して候補ルールを一括探索する場合:

```powershell
python scripts\search_rules_from_predictions.py --engine fastparquet --min-selections 120 --min-fold-selections 20
```

この設定では、上記の`pred_top3>=0.40`、複勝オッズ中間値`[3.0,5.0)`、距離`[1800,2200)`、開催場`3,7,10`除外が候補内トップでした。

買い点数を増やす場合は、`pred_top3>=0.35`、複勝オッズ中間値`[3.0,5.0)`、距離`[1800,2200)`、開催場`3,7,10`除外が候補です。walk-forward 4 foldで180レース、220点、83的中、的中率37.73%、回収率139.48%でした。

有望な固定ルールを詳細評価する場合:

```powershell
python scripts\evaluate_fixed_place_top3_rule.py --engine fastparquet
```

弱い開催場を除外して評価する場合:

```powershell
python scripts\evaluate_fixed_place_top3_rule.py --engine fastparquet --exclude-track-ids "3,6,10"
```

学習対象の距離帯だけを絞って、特定条件向けの専門モデルを検証する場合:

```powershell
python scripts\evaluate_fixed_place_top3_rule.py --engine fastparquet --pred-min 0.40 --train-distance-min 1800 --train-distance-max 2200
```

## 実装メモ

- パス定義は`src/data/paths.py`にあります
- データ保存先のデフォルトはWindowsでは`D:\horse_racing_ai\data`、WSL/Linuxでは`/mnt/d/horse_racing_ai/data`です
- スクレイピング処理は`src/scrape/`配下に分かれています
- 正規化処理は`src/preprocess/normalizers.py`に集約されています
- SQLiteへの投入処理は`src/data/database.py`にあります
- 学習用データセット作成処理は`src/features/`配下にあります
- 評価処理は`src/evaluate/`配下にあります
- モデル評価処理は`src/models/`配下にあります
- `src/pipelines/scrape_to_db.py`にスクレイピングからDB投入までの処理本体があります
- `scripts/scrape_to_db.py`はCLI用の薄い入口です

## 今後の予定

- 予測用特徴量の作成
- 学習・評価パイプラインの再構築
- 複勝の3着内確率予測
- ワイド、3連複への拡張
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
