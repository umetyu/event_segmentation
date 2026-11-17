# 光フロー比較の準備：EV-LayerSegNet / SNN / optimization

DSEC-Flowで3手法の精度・速度を比較するための、モデルサイズ調整と動作確認用コードです。
2つのニューラルネットワークのパラメータ計数・合成入力での前向き/逆向き計算に加え、
DSECの読み込み・GTマスク・時刻合わせ・実イベント入力の確認を実装しています。
**DSECでの学習、学習済みモデルの精度・速度比較、optimizationの実装はまだ含みません。**

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

`events_left/`またはシーケンス直下の`events.h5`と、
`optical_flow_forward/`または`optical_flow_forward_event/`に分けた配置も対応します。
時刻ファイルは`flow/forward_timestamps.txt`、フローPNGディレクトリ内の`timestamps.txt`、
またはシーケンス直下の`[<sequence>_]optical_flow_forward_timestamps.txt`を認識します。
PNG数と時刻行数の不一致、GTの欠損、未知の変換済み形式では評価せず停止します。

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

## 今後共通化する実験条件

- DSECの撮影シーケンス単位で学習・検証・内部テストを分割。
- 左イベントカメラの座標補正、GTの時刻区間・方向・変位単位を統一。
- 同じ約100 ms区間のイベントを入力し、SNNの状態resetと観測履歴を明示。
- 両NNを同じ自己教師あり損失・データ・学習量で再学習し、GTで精度を評価。
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
