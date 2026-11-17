# 光フロー比較の準備：EV-LayerSegNet / SNN / optimization

DSEC-Flowで3手法の精度・速度を比較するための、モデルサイズ調整と動作確認用コードです。
現在は2つのニューラルネットワークのパラメータ計数・合成入力での前向き/逆向き計算まで実装しています。
**DSECの学習、精度・速度評価、optimizationの実装はまだ含みません。**

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
