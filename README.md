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
- `horse`: 馬単位の基本情報と血統取得状態

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

既存DBに検索用インデックスを追加する場合。DB書き込みを伴うため、スクレイピングを止めた状態で実行してください:

```powershell
python scripts\ensure_db_indexes.py
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

特定の1レースだけ取得する場合:

```powershell
python scripts\scrape_to_db.py --race-id 202403030106
```

自動で次ページへ進める場合:

```powershell
python scripts\scrape_to_db.py --url "<netkeibaのレース一覧URL>" --mode auto
```

autoモード中に現在ページの処理が終わったところで停止したい場合は、同じターミナルで`stop`と入力してEnterを押します。別の停止文字列を使う場合:

```powershell
python scripts\scrape_to_db.py --url "<netkeibaのレース一覧URL>" --mode auto --auto-stop-command quit
```

レース一覧をページ単位で取得し、各ページ完了後に血統pending数が一定以上ならpendingがなくなるまで血統取得も実行する場合:

```powershell
python scripts\scrape_with_pedigree_backfill.py --url "<netkeibaのレース一覧URL>" --pedigree-threshold 100
```

このスクリプトでレース取得中に`stop`と入力してEnterを押すと、現在処理中のレースを最後まで取得し、血統pending数を確認します。pendingが閾値以上なら血統取得を実行し、閾値未満なら血統取得をせず停止します。血統取得中に`stop`を受け付けた場合は、pendingがなくなるまで血統取得を続けてから停止します。停止文字列を変える場合:

```powershell
python scripts\scrape_with_pedigree_backfill.py --url "<netkeibaのレース一覧URL>" --stop-command quit
```

DB内のレース数が指定値以上になったら開始せず終了する場合:

```powershell
python scripts\scrape_to_db.py --url "<netkeibaのレース一覧URL>" --limit 1000
```

発走前オッズのスナップショットを取得する場合:

```powershell
python scripts\collect_pre_race_odds.py --sweep --date 2026-07-04
python scripts\collect_pre_race_odds.py --watch --today
python scripts\collect_pre_race_odds.py --sweep --weekend
python scripts\collect_pre_race_odds.py --watch --weekend
```

対象は中央競馬の単勝・複勝・ワイド・3連複です。`--sweep`はその時点で取得できる対象レースのオッズを一括取得します。`--watch`は常駐し、各レースの発走時刻に対して`over_120`、`pre_60_120`、`pre_30_60`、`pre_15_30`、`pre_5_15`、`pre_2_5`、`pre_0_2`の各bucket終端に近いタイミングで1回ずつ取得します。`--weekend`は土日月の固定計算ではなく、netkeibaのレース一覧に掲載されている開催日を対象にし、発走済みレースは除外します。未表示のオッズは標準ではDBに保存せずスキップします。取得済みオッズは`pre_race_odds_snapshot`と券種別の`pre_race_win_odds`、`pre_race_place_odds`、`pre_race_wide_odds`、`pre_race_trio_odds`へ追記保存します。
APIレスポンスのraw JSONPはデフォルトで`D:\horse_racing_ai\data\raw_odds_snapshot`へ保存します。保存先を変える場合は`--raw-dir`を指定してください。

DBの収集状況と品質を確認する場合:

```powershell
python scripts\check_db_quality.py
```

別のDBファイルを確認する場合:

```powershell
python scripts\check_db_quality.py --db "D:\horse_racing_ai\data\hr.db"
```

既存の`runner`テーブルから`horse`テーブルへ馬IDを投入する場合:

```powershell
python scripts\backfill_horses_from_runners.py
```

未取得の血統情報を取得する場合:

```powershell
python scripts\fetch_pending_horse_pedigrees.py --limit 100
```

血統取得は`https://db.netkeiba.com/horse/ped/<horse_id>/`から取得します。`--limit`は1回の実行で取得する最大頭数です。1頭ごとの待機は常に1.5秒から2.0秒のランダム値です。

血統取得をどこまで進めると出走馬行カバー率が何%になるか確認する場合:

```powershell
python scripts\estimate_pedigree_fetch_coverage.py
```

`pending`が0になるまで取得し続ける場合:

```powershell
python scripts\fetch_pending_horse_pedigrees.py --until-empty
```

モデル用データの血統カバー率を早く上げたい場合は、出走回数が多い馬から優先して取得します:

```powershell
python scripts\fetch_pending_horse_pedigrees.py --limit 100 --order-by runner_count
```

`--order-by runner_count`は`runner.horse_id`を集計するため、既存DBでは先に`python scripts\ensure_db_indexes.py`を実行しておくと高速です。
SQLiteは同時書き込みが1つに制限されます。別のスクレイピング処理と同時に動かす場合は、血統取得側がロック解除を待ってDB更新を再試行します。待ち時間を長めにする場合:

```powershell
python scripts\fetch_pending_horse_pedigrees.py --limit 100 --order-by runner_count --db-retries 10 --db-retry-sleep 3
```

レース取得側のDB保存もロック解除を待って再試行します。それでも`database is locked`が続く場合は、同時に動かすDB書き込みプロセスを1つに絞ってください。

失敗済みの馬も再試行する場合:

```powershell
python scripts\fetch_pending_horse_pedigrees.py --limit 100 --include-failed
```

特定の1頭だけ取得し直す場合:

```powershell
python scripts\fetch_pending_horse_pedigrees.py --horse-id 2014104716
```

複勝3着内予測用の学習データセットをParquetで作成する場合:

```powershell
python scripts\build_place_top3_dataset.py --kind training --engine fastparquet
```

評価用の人気・複勝オッズデータをParquetで作成する場合:

```powershell
python scripts\build_place_top3_dataset.py --kind eval-odds --engine fastparquet
```

単勝1着予測を既存のCatBoost/探索パイプラインで扱う互換Parquetを作成する場合:

```powershell
python scripts\build_place_top3_dataset.py --kind win-training --engine fastparquet
python scripts\build_place_top3_dataset.py --kind win-eval-odds --engine fastparquet
```

単勝互換データは、既存モデルコードを再利用するため内部列名として`target_top3`と`place_odds_min`/`place_odds_max`を使います。中身はそれぞれ「1着かどうか」と単勝オッズです。

単勝互換データでCatBoost予測を作成し、単勝向けの低確率・低オッズ帯を探索する場合:

```powershell
python scripts\generate_catboost_predictions.py --engine fastparquet --training-dataset "D:\horse_racing_ai\data\feature\win_top1_dataset.parquet" --odds-dataset "D:\horse_racing_ai\data\feature\win_top1_eval_odds.parquet" --output "D:\horse_racing_ai\data\model\catboost_win_top1_predictions.parquet"
python scripts\search_rules_from_predictions.py --engine fastparquet --profile win --predictions "D:\horse_racing_ai\data\model\catboost_win_top1_predictions.parquet" --min-buy-rate 18 --max-buy-rate 22 --min-selections 70 --include-rank-ev-filters
```

現時点の単勝20%前後探索では、最高候補でも回収率は90%未満です。ROI優先の最新候補は、`pred_top3>=0.15`、単勝オッズ`[3.0,10.0)`、距離`1600m以上`、開催場`4,5,6,8,9`、芝、同一レース内の予測順位`3位以内`です。walk-forward 4 foldで534レース、759点、144的中、的中率18.97%、購入率19.50%、単勝回収率89.06%でした。ただし最低fold回収率は73.17%まで落ちます。安定性寄りの候補は、`pred_top3>=0.15`、単勝オッズ`[1.2,3.5)`、距離`1600m以上`、開催場`4,5,6,8,9`、芝、同一レース内の予測順位`3位以内`で、562レース、712点、276的中、的中率38.76%、購入率20.53%、単勝回収率88.30%、最低fold回収率86.05%でした。

改善候補を固定評価する場合:

```powershell
python scripts\evaluate_fixed_rule_from_predictions.py --engine fastparquet --predictions "D:\horse_racing_ai\data\model\catboost_win_top1_predictions.parquet" --pred-min 0.15 --odds-min 1.2 --odds-max 3.5 --distance-min 1600 --distance-max none --include-track-ids "4,5,6,8,9" --surface-id 0 --pred-rank-max 3
```

保存済みの複勝3着内予測からワイド候補を作り、購入レース割合20%前後のルールを探索する場合:

```powershell
python scripts\search_wide_rules_from_predictions.py --engine fastparquet --predictions "D:\horse_racing_ai\data\model\catboost_place_top3_predictions_affinity_lift_no_horse_id.parquet" --min-fold-return-mid 80 --output "D:\horse_racing_ai\data\model\wide_rule_search_results_affinity_lift_no_horse_id_minfold80.csv" --selections-output "D:\horse_racing_ai\data\model\wide_rule_selections_affinity_lift_no_horse_id_minfold80.csv"
```

現時点のワイド探索では、`catboost_place_top3_predictions_affinity_lift_no_horse_id.parquet`を使った候補が最上位です。条件は、同一レース内のペア予測スコア上位5点、2頭の予測順位がともに3位以内、ペア予測スコア`0.10`以上、2頭の予測確率の小さい方が`0.25`以上、ワイドオッズ中間値`[10.0,100.0)`、開催場`1,2,3,7,10`除外です。walk-forward 4 foldで379レース、520点、37的中、的中率7.12%、購入率18.66%、ワイドオッズ中間値ベース回収率114.36%、最低fold回収率100.63%でした。最新のデフォルト複勝予測`catboost_place_top3_predictions.parquet`で同条件の探索候補を更新すると、トップは距離`2000m以上`、ペア予測スコア上位5点、2頭の予測順位がともに3位以内、ワイドオッズ中間値`[2.0,30.0)`で、535レース、1,289点、291的中、購入率19.05%、回収率102.63%でした。

保存済みの複勝3着内予測から3連複候補を作り、購入レース割合20%前後のルールを探索する場合:

```powershell
python scripts\search_trio_rules_from_predictions.py --engine fastparquet --predictions "D:\horse_racing_ai\data\model\catboost_place_top3_predictions_affinity_lift_no_horse_id.parquet" --min-fold-return-mid 80 --output "D:\horse_racing_ai\data\model\trio_rule_search_results_affinity_lift_no_horse_id_minfold80.csv" --selections-output "D:\horse_racing_ai\data\model\trio_rule_selections_affinity_lift_no_horse_id_minfold80.csv"
```

現時点の3連複探索では、`catboost_place_top3_predictions_affinity_lift_no_horse_id.parquet`を使った候補が上位です。条件は、同一レース内の3頭組予測スコア上位10点、3頭の予測順位がすべて6位以内、3連複オッズ`[100.0,1000.0)`、芝、開催場`1,2,3,7,10`除外です。walk-forward 4 foldで385レース、939点、9的中、的中率0.96%、購入率18.96%、3連複回収率142.94%、最低fold回収率93.92%でした。最新のデフォルト複勝予測`catboost_place_top3_predictions.parquet`では、最低fold回収率80%以上のトップが606レース、1,217点、15的中、購入率21.58%、回収率115.82%でした。

単勝・複勝・ワイド・3連複を同じ100円単位で組み合わせたポートフォリオとして評価する場合:

```powershell
python scripts\evaluate_bet_portfolios.py --engine fastparquet --output "D:\horse_racing_ai\data\model\bet_portfolio_results.csv"
```

現時点では、券種ごとに広狭を変えた`best_mixed_width`が上位です。単勝は安定狭め、複勝は少し広め、ワイドと3連複は高ROI狭めで組み合わせます。結果は1,294レース、2,807点、526的中、購入率46.08%、的中率18.74%、ROI121.60%、最低fold ROI105.71%でした。的中率寄りの3連複を使う`balanced_all`は、1,229レース、3,183点、598的中、購入率43.77%、的中率18.79%、ROI113.79%、最低fold ROI102.23%でした。買い点数を増やす`volume_plus_roi`は購入率45.83%、3,942点まで増えますが、ROIは109.03%、最低fold ROI98.63%まで下がります。

購入率を30%前後まで落とす場合は、全券種ではなく2券種中心が上位です。複勝広めと3連複高ROIを組み合わせる`place_trio_roi`は、845レース、1,575点、213的中、購入率30.09%、的中率13.52%、ROI139.05%、最低fold ROI115.54%でした。ワイド高ROIと3連複高ROIだけの`exotic_roi`は、622レース、1,459点、46的中、購入率30.63%、的中率3.15%、ROI132.75%、最低fold ROI96.28%でした。複勝広めとワイド高ROIの`place_wide`は、839レース、1,156点、241的中、購入率29.88%、的中率20.85%、ROI124.78%、最低fold ROI102.61%でした。

血統特徴量あり/なしを比較し、購入率20%前後の合議ルール候補を探す場合:

```powershell
python scripts\run_pedigree_place_top3_experiment.py --engine fastparquet
```

実験結果のサマリーCSVはデフォルトで次の場所に出力されます。

- `D:\horse_racing_ai\data\model\pedigree_place_top3_portfolios.csv`
- `D:\horse_racing_ai\data\model\pedigree_place_top3_rules.csv`

出力先を変える場合:

```powershell
python scripts\run_pedigree_place_top3_experiment.py --engine fastparquet --portfolio-output "D:\horse_racing_ai\data\model\portfolios.csv" --rule-output "D:\horse_racing_ai\data\model\rules.csv"
```

この実験は、学習データセット内の血統行カバー率がデフォルトで20%未満の場合は停止します。低カバー率でも動作確認だけ行う場合:

```powershell
python scripts\run_pedigree_place_top3_experiment.py --engine fastparquet --force-low-coverage
```

既に予測Parquetを作成済みで、評価だけやり直す場合:

```powershell
python scripts\run_pedigree_place_top3_experiment.py --engine fastparquet --skip-predictions
```

学習データセットのデフォルト出力先は`D:\horse_racing_ai\data\feature\place_top3_dataset.parquet`です。
評価用オッズデータのデフォルト出力先は`D:\horse_racing_ai\data\feature\place_top3_eval_odds.parquet`です。
過去成績特徴量には、馬・騎手・調教師の過去成績、馬や騎手の競馬場/芝ダート/距離帯別成績、直近フォーム、前走からの日数・距離変化、過去平均との差による距離・斤量適性、同一レース内での過去成績順位や平均との差分が含まれます。競馬場、芝ダート、距離帯、コース条件については、対象条件での経験割合と通常3着内率からの上振れ/下振れも特徴量化します。
血統情報が取得済みの場合は、父・母・母父の過去3着内率、平均着順、父/母父の競馬場・芝ダート・距離帯別成績もリークしない時系列特徴量として追加します。

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

特定の特徴量グループを一時的に除外してCatBoostを評価する場合:

```powershell
python scripts\evaluate_fixed_place_top3_catboost_rule.py --engine fastparquet --drop-feature-patterns "horse_jockey_,jockey_trainer_,horse_trainer_"
```

芝やダートだけで学習した専門モデルを検証する場合:

```powershell
python scripts\evaluate_fixed_place_top3_catboost_rule.py --engine fastparquet --train-surface-id 0 --surface-id 0
```

現時点の代表確認は、保存済みCatBoost予測から購入率20%前後の候補を探索する流れです。学習データにはレース後に確定する`popularity`、単勝オッズ、複勝オッズを含めません。これらは予測後の回収率評価と買い基準検証にだけ使います。最新DB反映後のデフォルト予測では、`pred_top3>=0.37`、複勝オッズ中間値`[3.2,6.0)`、距離`1200m以上`、同一レース内の予測順位`5位以内`、`pred_top3 * 複勝オッズ中間値 >= 1.5`が上位候補です。walk-forward 4 foldで521レース、600点、189的中、的中率31.50%、購入率18.55%、複勝オッズ中間値ベース回収率133.62%、最低fold回収率92.34%でした。更新前の代表候補は、`pred_top3>=0.34`、複勝オッズ中間値`[3.2,6.0)`、距離`1200m以上`、開催場`3,7,10`除外、同一レース内の予測順位`5位以内`、`pred_top3 * 複勝オッズ中間値 >= 1.4`で、571レース、671点、212的中、購入率20.59%、回収率133.09%でした。

CatBoostのwalk-forward予測を保存し、重い再学習を避けて買い条件だけを高速に検証する場合:

```powershell
python scripts\generate_catboost_predictions.py --engine fastparquet
python scripts\evaluate_fixed_rule_from_predictions.py --engine fastparquet --pred-min 0.40
```

購入率20%前後の改善候補を再評価する場合:

```powershell
python scripts\evaluate_fixed_rule_from_predictions.py --engine fastparquet --pred-min 0.37 --odds-min 3.2 --odds-max 6.0 --distance-min 1200 --distance-max none --pred-rank-max 5 --ev-mid-min 1.5
```

本番予測用にCatBoostモデルを保存する場合。以後の予測実行ではこの保存済みモデルを読み込むため、毎回学習し直しません。

```powershell
python scripts\train_catboost_model.py --profile place --engine fastparquet
python scripts\train_catboost_model.py --profile win --engine fastparquet
```

デフォルトでは次のファイルへ出力します。

- `local_models\catboost_place_top3_model.cbm`
- `local_models\catboost_place_top3_model_metadata.json`
- `local_models\catboost_win_top1_model.cbm`
- `local_models\catboost_win_top1_model_metadata.json`

`local_models\`はgit追跡対象外です。

出走表ページから当日以降のレースを予測する場合:

```powershell
python scripts\predict_upcoming_races.py --profile place --race-id 202605030101
python scripts\predict_upcoming_races.py --profile place --date 2026-07-04
python scripts\predict_upcoming_races.py --profile win --date 2026-07-04
```

`--date`はnetkeibaの開催一覧`https://race.netkeiba.com/top/race_list.html?kaisai_date=YYYYMMDD`から中央競馬の`race_id`を抽出します。予測CSVはデフォルトで`local_models\live_predictions.csv`へ出力し、端末には現在の固定ルールに合う買い候補だけを表示します。発走前オッズを先に`collect_pre_race_odds.py`で保存していれば、最新スナップショットの単勝または複勝オッズを使って買い候補を絞ります。

買い候補ルールはJSONでまとめて指定できます。JSONに書けるキーは`pred_min`、`odds_min`、`odds_max`、`distance_min`、`distance_max`、`include_track_ids`、`exclude_track_ids`、`surface_id`、`pred_rank_max`、`ev_mid_min`です。

```json
{
  "pred_min": 0.34,
  "odds_min": 3.2,
  "odds_max": 6.0,
  "distance_min": 1200,
  "exclude_track_ids": [3, 7, 10],
  "pred_rank_max": 5,
  "ev_mid_min": 1.4
}
```

```powershell
python scripts\predict_upcoming_races.py --profile place --date 2026-07-04 --rule-json local_models\place_rule.json
```

一部だけ変える場合は個別オプションでも指定できます。

```powershell
python scripts\predict_upcoming_races.py --profile place --date 2026-07-04 --rule-pred-min 0.36 --rule-odds-min 3.0 --rule-exclude-track-ids "3,7,10"
```

特徴量グループを除外した予測キャッシュを作る場合:

```powershell
python scripts\generate_catboost_predictions.py --engine fastparquet --drop-feature-patterns "horse_jockey_,jockey_trainer_,horse_trainer_" --output "D:\horse_racing_ai\data\model\catboost_place_top3_predictions_no_connection.parquet"
```

芝やダートだけで学習した予測キャッシュを作る場合:

```powershell
python scripts\generate_catboost_predictions.py --engine fastparquet --train-surface-id 0 --output "D:\horse_racing_ai\data\model\catboost_place_top3_predictions_turf_train.parquet"
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

購入率20%前後で探索する場合:

```powershell
python scripts\search_rules_from_predictions.py --engine fastparquet --min-buy-rate 18 --max-buy-rate 22 --min-selections 70
```

同一レース内の予測順位と、予測確率に複勝オッズ中間値を掛けた期待値条件も含めて探索する場合:

```powershell
python scripts\search_rules_from_predictions.py --engine fastparquet --min-buy-rate 18 --max-buy-rate 22 --min-selections 70 --min-fold-selections 10 --include-rank-ev-filters
```

各foldの回収率下限も指定して、不安定な候補を除外する場合:

```powershell
python scripts\search_rules_from_predictions.py --engine fastparquet --min-selections 120 --min-fold-selections 20 --min-fold-return-mid 100
```

血統あり、血統なし、血統あり/なし合議の代表候補を同じ条件で比較する場合:

```powershell
python scripts\compare_place_top3_prediction_candidates.py --engine fastparquet
```

この設定では、上記の`pred_top3>=0.40`、複勝オッズ中間値`[3.0,5.0)`、距離`[1800,2200)`、開催場`3,7,10`除外が候補内トップでした。
探索候補には複勝オッズ中間値の`[2.5,5.0)`、`[3.0,4.0)`、`[4.0,5.0)`、`[3.0,6.0)`なども含めています。

買い点数を増やす場合は、`pred_top3>=0.35`、複勝オッズ中間値`[3.0,5.0)`、距離`[1800,2200)`、開催場`3,7,10`除外が候補です。walk-forward 4 foldで180レース、220点、83的中、的中率37.73%、回収率139.48%でした。各foldの最低回収率を100%以上にした探索でも、この候補は最低116.55%で残ります。

経験割合と通常3着内率からの上振れ/下振れ特徴量を追加した試験では、買い点数多めの候補が改善しました。`pred_top3>=0.35`、複勝オッズ中間値`[3.0,5.0)`、距離`[1800,2200)`、芝に限定した条件で98レース、121点、52的中、的中率42.98%、回収率159.63%でした。同じ芝限定条件の既存特徴量モデルは97レース、119点、44的中、回収率137.31%でした。ただし月別では弱い月もあるため、主候補は固定せず、保存済み予測からのルール探索で比較します。

経験割合と上振れ/下振れ特徴量を追加した予測キャッシュでは、買い点数別に次の候補を比較します。

- 高回収寄り: `pred_top3>=0.45`、複勝オッズ中間値`[3.0,6.0)`、距離`[1800,2200)`で92レース、103点、50的中、回収率180.97%、最低fold回収率131.50%
- 中間: `pred_top3>=0.35`、複勝オッズ中間値`[3.0,5.0)`、距離`[1800,2200)`、芝限定で98レース、121点、52的中、回収率159.63%、最低fold回収率107.50%
- 買い目多め: `pred_top3>=0.35`、複勝オッズ中間値`[3.0,5.0)`、距離`[1800,2200)`、開催場`4,5,6,8,9`で184レース、225点、88的中、回収率146.24%、最低fold回収率113.06%

この3候補をまとめて再評価する場合:

```powershell
python scripts\evaluate_rule_tiers_from_predictions.py --engine fastparquet --predictions "D:\horse_racing_ai\data\model\catboost_place_top3_predictions_affinity_lift_trial.parquet"
```

3候補に該当した馬をCSVへ出力する場合:

```powershell
python scripts\export_rule_tier_selections.py --engine fastparquet --predictions "D:\horse_racing_ai\data\model\catboost_place_top3_predictions_affinity_lift_trial.parquet" --output "D:\horse_racing_ai\data\model\rule_tier_selections_affinity_lift_trial.csv"
```

`horse_id`を学習から除外したモデルは、高回収寄りだけ派生候補として比較します。中間・買い目多めでは悪化したため、標準の3候補とは分けて扱います。

```powershell
python scripts\evaluate_rule_tiers_from_predictions.py --engine fastparquet --tier-set no_horse_id_high_return --predictions "D:\horse_racing_ai\data\model\catboost_place_top3_predictions_affinity_lift_no_horse_id.parquet"
```

この派生候補は、`pred_top3>=0.45`、複勝オッズ中間値`[3.0,6.0)`、距離`[1800,2200)`、開催場`3,7,10`除外で79レース、94点、46的中、回収率187.18%、最低fold回収率132.92%でした。

特徴量グループごとの影響を確認する場合:

```powershell
python scripts\evaluate_feature_ablation.py --engine fastparquet
```

CatBoostのfold平均特徴量重要度をCSVへ出力する場合:

```powershell
python scripts\export_catboost_feature_importance.py --engine fastparquet --training-dataset "D:\horse_racing_ai\data\feature\place_top3_dataset_affinity_lift_trial.parquet" --output "D:\horse_racing_ai\data\model\catboost_feature_importance_affinity_lift_trial.csv"
```

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
