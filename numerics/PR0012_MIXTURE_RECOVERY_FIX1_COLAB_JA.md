# 可読ソース版の実行案内：pr0012_fix1

公開用の入口は[実験フォルダ](experiments/pr0012_fix1/README.md)、[検査付きPython](experiments/pr0012_fix1/run_colab.py)、[Notebook](experiments/pr0012_fix1/run_colab.ipynb)です。ソースを圧縮したOnecellは配布しません。

1. リポジトリ全体をColabのランタイムへ展開し、実験フォルダのREADMEと設定を読む。
2. Notebookの`EXPERIMENT_DIR`を指定して検査セルを実行する。この段階では実験もインストールも開始しない。
3. 元のCPU/GPU条件を確認し、実行するときだけ`START_FULL_RUN = True`へ変更する。Driveは接続しない。
4. 復元設定と自動DLは実験フォルダの案内に従う。過去の安全停止時間・容量上限・環境条件は実験ごとに異なる。ブラウザの転送要求と手元の保存完了は別に確認する。

科学コード・条件・元checkpoint識別子は保持しています。今回の変更では実際のColab本番・ブラウザDL・ランタイム消失を再試験していません。

当時の運転案内は[履歴テキスト](PR0012_MIXTURE_RECOVERY_FIX1_COLAB_JA.md.txt)として保持しています。履歴中のOnecell取得先・全文貼付指示は現在の公開入口ではありません。科学的な条件や失敗・修正経緯の参照用です。
