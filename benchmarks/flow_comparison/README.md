# 光フロー比較の準備：EV-LayerSegNet / SNN / optimization

DSEC-Flowで3手法の精度・速度を比較するための、モデルサイズ調整と動作確認用コードです。
2つのニューラルネットワークのパラメータ計数・合成入力での前向き/逆向き計算に加え、
DSECの読み込み・GTマスク・時刻合わせ・実イベント入力の確認、共通条件での短時間学習、
保存済みcheckpointのゼロフロー比較とモデル部分の推論時間計測を実装しています。
**本学習による3手法の精度・速度比較、optimizationの実装はまだ含みません。**

## モデルサイズ

| 構成 | 学習可能パラメータ数 |
| --- | ---: |
| EV-LayerSegNet、基本幅32（調整前） | 28,432,750 |
| SpikingRecEVFlowNet、基本幅32（調整前） | 20,400,840 |
| EV-LayerSegNet、基本幅32＋フロー分岐の中間幅548 | 28,764,562 |
| SpikingRecEVFlowNet、基本幅38 | 28,765,020 |

調整後の差は458個、EV-LayerSegNetに対して約0.001592%です。許容差は1%に設定しています。
ヘッドを含む全モデルと、SNNの学習可能な閾値・減衰係数を計数します。
調整前の行はコード上の比較用設定であり、論文の全実験条件の再現を意味しません。

segmentationでは、アフィンフロー分岐の最初の2畳み込みを
`512 → 512 → 512`から`512 → 548 → 512`に変更します。
活性化関数、空間stride、出力、skip接続は同じです。追加した重みも実際に計算に使われ、
合成入力のチェックでは非ゼロの勾配を確認します。

パラメータ数を合わせても、演算量、SNNの状態メモリ、実行時間は同じにはなりません。
これらは本評価で別々に測定する必要があります。

## GitHub上で実行するチェック

`Flow comparison preflight`ワークフローがCPUで以下を実行します。

1. PyTorch `2.5.1+cpu`とNumPy `1.26.4`をインストール。
2. 固定コミットから必要なモデル定義だけを取得し、SHA-256を検証。
3. 調整前後のパラメータ数を計測し、調整後の差が1%以内であることを確認。
4. 調整後のモデルを64×80の合成入力で動かし、出力形状、有限な勾配、
   全学習パラメータから損失への接続を確認。
5. JSONレポートをActionsのartifactとして保存。

小さな人工データでDSECのRGB/BGR順序、時刻オフセット、半開区間、空区間、
座標補正・補間、GTマスク、評価指標を確認するテストも実行します。

SNNのチェックは内部状態を使う経路の確認のため2ステップ、segmentationは1ステップです。
これは実装の動作確認であり、公平な速度測定や精度評価ではありません。

ローカルで同じ確認を行う場合：

```bash
cd benchmarks/flow_comparison
python scripts/fetch_sources.py
python scripts/model_preflight.py --device cpu --output results/cpu-preflight.json
```

## IzarでGPU確認を実行

既存のIzar用スクリプトと同じ`gcc/11.3.0`、`python/3.10.4`を使い、
Python標準の`python -m venv`で専用環境を作り、PyTorch `2.5.1+cu121`とNumPy `1.26.4`をインストールします。
Izarにない`py-virtualenv`モジュールや、外部の`virtualenv`コマンドは必要ありません。
環境作成直後にpipを`25.0.1`に揃え、古いpipによる依存パッケージ名の解釈エラーを避けます。
システムのCUDA Toolkitやドライバの変更は行いません。

Izar上で、このリポジトリのルートから実行します。

```bash
bash benchmarks/flow_comparison/cluster/setup_izar.sh
```

セットアップに失敗してコードを更新した場合は、取得済みリポジトリ内で
`git pull --ff-only`を実行し、上記のセットアップを再実行してください。
モジュール読み込みや保存先ディレクトリ作成の段階で停止した場合、cloneや環境の削除は不要です。

環境の既定の保存先は`$HOME/venvs/event-flow-torch251-cu121`です。
`/scratch/$USER`はアカウントによって作成権限がないため、自動では使いません。
書き込み権限と空き容量のある別の保存先を使う場合は、`EVENT_FLOW_VENV`に絶対パスを指定してください。
指定先はログインノードと計算ノードの両方からアクセスできる必要があります。
仮想環境には数GBの空き容量が必要です。DSECデータの保存先は別途設定します。
取得したモデル定義はこのディレクトリ内の`.sources/`に保存されます。
既存ファイルのハッシュが異なる場合は上書きせず停止します。

セットアップ後、`gpu`パーティション・`debug` QoS・GPU 1枚・最大20分のジョブを投入し、
ジョブIDを表示します。Slurm accountはユーザーの既定値を使います。
GPU実行の前に`results/`を作成するため、Slurmログの保存先も用意されます。

```bash
squeue -u "$USER"
cat benchmarks/flow_comparison/results/model-preflight-<JOB_ID>.log
cat benchmarks/flow_comparison/results/model-preflight-<JOB_ID>.json
```

GPUの型番、VRAM、CUDA runtime、合成入力のピークメモリはJSONに記録します。
このピークメモリをDSECの640×480入力や長いBPTT系列の必要メモリとして扱わないでください。
GitHub ActionsからIzarへログインする処理や、SSH認証情報の登録は含みません。

## 次の段階：既存DSECデータで確認

Izar上のリポジトリのルートで実行します。以下は今回の既存データ保存先です。
データのダウンロード・変換・上書きは行いません。

```bash
git pull --ff-only
bash benchmarks/flow_comparison/cluster/submit_dsec_preflight.sh \
  /home/umetsu/projects/event_segmentation/dataset/dsec_flow_matched
```

専用venvにデータ読み込み用の依存パッケージを追加し、Slurmジョブを投入します。
ジョブIDを表示後、終了まで待ってJSONを表示します。失敗時も診断情報を保存します。
結果は`benchmarks/flow_comparison/results/dsec-preflight-<JOB_ID>.json`と同名の`.log`です。
対象シーケンスを指定する場合は保存先の後にシーケンス名を追加してください。
指定しない場合は、見つかったGT付きシーケンスの辞書順で最初のものを検査します。

確認内容：

- 元のDSECの整数マイクロ秒時刻に`t_offset`を加え、各GTの開始・終了時刻に一致する
  `[start, end)`のイベントだけを選択。固定100 msやファイル番号から時刻を推定しません。
- 左カメラの`rectify_map[y, x]`で補正し、画面外になるイベント数を記録。
- 16-bit PNGをRGB順に戻し、`(value - 32768) / 128`で前向きの画素変位に変換。
  第3チャンネルの有効マスクをそのまま使い、**有効なゼロフローも評価対象に残します**。
- シーケンスの先頭・中間・末尾の最大3区間を検査。GT自身とのEPEが0であることを確認し、
  ゼロフロー予測のEPE・1/2/3-pixel error（厳密な`> N`、百分率）を参考値として記録。
  指標は各サンプル内の有効画素平均で、DSEC公式ランキングの集計結果ではありません。
- 両NNは同じ区間の全イベントを使用。segmentationは5-bin signed voxel、
  SNNはその区間を5等分した正負2チャンネルのカウント列を使用します。
  空間方向は双線形補間、voxelの時間方向は線形補間。正規化・GTによる入力マスクは使いません。
- 最初の区間を元解像度480×640、batch=1で両NNに入力し、有限な出力と形状を確認。
  SNN状態は区間開始時にresetし、5ステップを実行します。
  **重みはランダム初期値で、学習・速度計測・モデル精度評価は行いません。**

対応する元データの配置例：

```text
<root>/[train/]<sequence>/
  events/left/events.h5
  events/left/rectify_map.h5
  flow/forward/*.png
  flow/forward_timestamps.txt
```

`events/events.h5`、`events_left/events.h5`またはシーケンス直下の`events.h5`と、
`optical_flow_forward/`、`optical_flow_forward_event/`または`flow/`直下のPNGに分けた配置も対応します。
時刻ファイルは`flow/forward_timestamps.txt`、フローPNGディレクトリ内の`timestamps.txt`、
またはシーケンス直下の`[<sequence>_]optical_flow_forward_timestamps.txt`、
`flow_timestamps.txt`、`forward_timestamps.txt`を認識します。
`flow/`直下のPNGを使う場合も、前向きの時刻区間・PNG/時刻行数・GT形式を検証します。
PNG数と時刻行数の不一致、GTの欠損、未知の変換済み形式では評価せず停止します。
配置が見つからない場合は`sequence_discovery`にPNGや時刻ファイルの場所を記録します。
配置の不一致だけを理由に、GTや有効マスクが欠けているとは判定しません。

既存の`data/dsec_to_h5.py`と`data/h5_packager.py`による変換では、
GTの無効画素がゼロに置き換わり、元の有効マスクが保存されません。
この形式の場合はHDF5構造をJSONに記録し、元PNGとマスクを保持した形式への対応が必要と報告します。
元データなしにゼロ値からマスクを復元することはできません。

CPUまたはデータだけの確認：

```bash
cd benchmarks/flow_comparison
python -m pip install -r requirements-data.txt
python -m unittest discover -s tests -v
python scripts/dsec_preflight.py --root /path/to/DSEC --device none --output results/dsec-data.json
# 同じチェックに両モデルのCPU推論を追加する場合は --device cpu
```

## 両NNを共通条件で20更新する

既定設定は`configs/short_train.json`です。これは学習・保存・再開の動作確認であり、
論文の学習設定の再現や、収束後の精度・速度比較ではありません。

```bash
# Izarのリポジトリルートから
bash benchmarks/flow_comparison/cluster/submit_short_train.sh \
  /home/umetsu/projects/event_segmentation/dataset/dsec_flow_matched
```

1 GPU・最大20分で、segmentation、SNNの順に実行し、終了後に結果を表示します。
各モデルは10更新でプロセスを終了して保存し、別プロセスで再開して合計20更新します。
データの準備も計算ノードで実行します。元データは読み取り専用です。
設定を変える場合はコピーしたJSONのパスを第2引数に指定してください。

固定した内部分割（いずれも公式training set内）：

| 用途 | 撮影グループと区間 |
| --- | --- |
| 学習 | `zurich_city_01_a`、`zurich_city_02_{a,c,d,e}` |
| 検証 | `zurich_city_10_{a,b}` |
| 内部テスト・今回は使用しない | `zurich_city_11_{a,b,c}` |

末尾の区間名だけが違う同一撮影グループは、異なる分割へ入れられません。
必要なシーケンスがなければ停止し、自動で別データへ置き換えません。
全DSECを使った分割でも、公式test setによる評価でもありません。

両モデルで共通にする条件：

- 各学習シーケンスの全GT時刻範囲から等間隔に4区間を選び、計20区間を使用。
  検証は各2区間で計4区間。GTの**時刻情報のみ**で入力区間を揃え、学習用PNGは開きません。
- 中央256×256 crop、batch=1、FP32、拡張なし、seed=7、20 optimizer updates。
  cropは座標原点だけを変え、GT変位の大きさは変更しません。
- Adam、learning rate=`1e-5`、gradient norm clip=`1.0`、schedulerなし。
  両モデルに同じ区間と順序を渡します。入力キャッシュとSHA-256を共有manifestに固定します。
- segmentationは5-bin voxelを1回、SNNは5分割の正負カウントを5ステップ入力。
  全イベントを入力表現に使い、各区間の開始時にSNN状態をreset。
  5ステップを通じて逆伝播し、区間をまたいだ状態や勾配は保持しません。
- 損失は両者とも**区間末の最終解像度のフローだけ**で計算。
  SNNの途中解像度の予測は後続decoderへ接続されており、全学習パラメータの勾配接続も検査します。

共通損失は前後2方向の平均タイムスタンプ画像の二乗和に、空間平滑化と画面外ワープの罰則を加えたものです。
`scripts/flow_objective.py`に実装しています。GTフローやGTマスクは損失関数へ渡しません。
計算量を制限するため、損失に使うイベントだけを最大50,000個に等間隔抽出します。
抽出結果は両モデルで同一です。境界罰則はイベントを画面外へ逃がす退化を抑える補助項であり、
フローの正確さやevent collapseを防げることを保証するものではありません。
損失の内訳と、ワープ後に画面内に残るイベント率を記録します。

フローの単位は、両モデルの出力に`max(H, W)`（今回は256）を掛けた区間全体の画素変位です。
SNNのステップ数を追加で掛けません。これは短時間学習用に統一した出力の解釈です。
これらのcrop用checkpointをそのまま640×480で評価する手順は含めていません。

結果は`benchmarks/flow_comparison/results/short-train-<JOB_ID>/`に保存します。

- `summary.json`：両モデルの更新回数、再開回数、固定学習サンプルの損失変化、検証損失とEPE等。
- `data/manifest.json`：設定、分割、選択区間、キャッシュのハッシュ。内部テストのラベルは読みません。
- `<model>/last.pt`：重み、Adam状態、更新回数、RNG状態、学習履歴。
- `<model>/summary.json`：詳細な勾配・損失履歴、検証値、設定・上流ソース・実行環境。

`short_comparison_completed`は実行が完了したことを表します。
`train_probe_loss_decreased`と`validation_loss_decreased`で改善の有無を別途確認してください。
検証EPEは小さなcropと4区間だけの値です。損失が減ることと精度向上は同義ではありません。
学習量をそろえてもMAC数・実行時間は同じにならず、速度測定は別途必要です。

CPU CIでは人工イベントで両モデルを学習し、連続4更新と「2更新＋保存・再開＋2更新」の
重み・損失・サンプル順序が完全一致することを検査します。
CUDAのscatter加算は非決定的になり得るため、GPU上のビット単位一致は保証しません。
再開時はデータ・設定・コード・モデルが一致するかを照合します。

途中から手動で続ける場合（合計更新数は設定の上限以内）：

```bash
cd benchmarks/flow_comparison
python scripts/short_train.py \
  --manifest results/short-train-<JOB_ID>/data/manifest.json \
  --model snn --device cuda --output results/short-train-<JOB_ID>/snn \
  --until 20 --resume
```

このGPUコマンドもSlurmの割り当て内で実行してください。
無関係な実験のcheckpointを読み込んだり、同じ出力先で最初から上書きしたりしないよう検査しています。

## 20更新後の結果と追加診断

ユーザー提供のIzarジョブ`3256311`の出力（コード`3f72649`）では、両モデルの20更新、
重みの変更、10更新からの再開が完了しました。256×256の同じ4検証区間での結果です。

| モデル | 学習前EPE [px] | 20更新後EPE [px] | 検証自己教師あり損失：前 → 後 |
| --- | ---: | ---: | --- |
| segmentation | 15.7007 | 15.1512 | 0.515554 → 0.514171 |
| SNN | 4.4668 | 5.0086 | 0.515748 → 0.513587 |

SNNでは損失が減ってもEPEが増えています。固定学習サンプルの損失もsegmentationでは微増しました。
初期値、出力形式、短い学習量の影響があるため、これだけでモデルの優劣や学習の収束を判断しません。

既存の重み・検証キャッシュを使って、再学習せずに診断できます。

```bash
# Izarのリポジトリルートから。引数は既存の学習結果ディレクトリ
bash benchmarks/flow_comparison/cluster/submit_diagnostics.sh \
  benchmarks/flow_comparison/results/short-train-3256311
```

1 GPU・最大20分のジョブを投入し、終了を待って結果を表示します。
学習時のコード・設定・データの照合を維持し、checkpointと検証キャッシュは読み取り専用で使います。
学習データや予約した内部テストのファイルは開きません。
結果は`results/diagnose-<JOB_ID>/summary.json`、区間別の詳細は`segmentation.json`と`snn.json`です。

- 同じGT有効画素に対するモデルEPEと**ゼロフローEPE**、差、1/2/3-pixel error。
  全体のEPEは有効画素数で重み付けし、区間の単純平均とは区別します。
- 区間別の予測変位の平均・標準偏差、変位量の中央値・95%点・最大値、GTの変位量。
  全画素とGT有効画素を分けて記録します。
- モデルとゼロフローの共通損失、損失の内訳、ワープ後に画面内へ残るイベント率。
  画面内に残ることだけでevent collapseがないとは判断できません。
- モデル部分の推論時間：各区間5回warmup後に20回計測し、中央値・95%点を記録。
  batch=1、FP32、TF32無効、状態resetから全区間のforward、有限値検査、画素単位への変換までを含みます。
  SNNは5ステップ全体です。CUDAでは各計測の前後に`synchronize()`を実行します。
  データ読み込み、イベント表現の作成、CPU/GPU転送、損失・EPE計算、モデル読み込みは含めません。
  **前処理込みの速度やフル解像度の速度ではありません。**
- 区間別のGPUメモリ割当量、推論中の最大割当量と増分。CUDA allocatorの予約量やGPU全体の使用量とは異なります。

CPU CIでも両モデルの保存済み重みを別プロセスで読み込み、診断EPEが学習終了時と一致すること、
既知の合成変位に対するゼロフローEPEが正しいことを確認します。CPU CIの時間はGPU比較には使いません。

## 本比較に進む前に

- より十分な撮影グループ数と学習量を持つ分割を固定し、複数seedで検証。
- 左イベントカメラの座標補正、GTの時刻区間・方向・変位単位を統一。
- 同じ約100 ms区間のイベントを入力し、SNNの状態resetと観測履歴を明示。
- 両NNを共通条件で本学習し、独立した内部テストのGTで精度を評価。
- EPEと1/2/3-pixel errorを同じGT有効画素で計算。
- 速度計測には前処理、全SNNステップ、全最適化反復、後処理を含める。

## 出典

上流コードは同梱せず、`sources.lock.json`に固定コミットとハッシュを記録しています。

- [EV-LayerSegNet](https://github.com/Fulmen67/event_segmentation)
- [NeurIPS 2021 SNN](https://github.com/tudelft/event_flow)
- [Gallego et al. 2018](https://arxiv.org/abs/1804.01306)
- [DSEC-Flow](https://dsec.ifi.uzh.ch/uzh/dsec-flow-optical-flow-benchmark/)
- [SCITAS Python環境](https://scitas-doc.epfl.ch/user-guide/software/python/python-venv/)
- [PyTorchのバージョン指定インストール](https://pytorch.org/get-started/previous-versions/)
