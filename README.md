# 日本語：
Windows OS，Linux OSどちらも可

## Poetry インストール
使用するときは python の poetry がインストールされているか確認してください：

```bash
$ poetry --version
```

インストールされていない場合，こちらからインストールしてください：

https://cocoatomo.github.io/poetry-ja/


## LUALCADの動作
最初に .../Larva ディレクトリに入る。次にjupyter notebook が開けるアプリで Larva/RUNTHIS.ipynb を開く（vscode等で）ファイル中の指示に従う。

Larva/sandbox/playground.py で実際に制作したPython関数を使うことができる。

Python の作動の仕方（必ず```$ poetry shell```後に行うこと）：
```bash
(larva-py3.10):$ python -m sandbox.playground
```