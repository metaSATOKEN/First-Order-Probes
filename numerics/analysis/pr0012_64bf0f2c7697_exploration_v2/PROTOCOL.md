# PR-0012 保存データの別角度点検

範囲は[計算前に保存した四面のプロトコル](../../../docs/OMITTED_MATERIALS.md#item-b9ebd6607096)のまま。主13・科学条件・ゲート・閾値・政策を変更しない、既知50反復の探索的点検である。

v1集計後のコード点検で、validationのcomponent_closureとbranch_closureが未更新の初期値0であることに気づいた。実際の枝計数のassertionは実行しているが、この二つは測定値ではない。旧v1とソースを保持し、未使用カウンタを削除したv2を正本とする。測定したmean_closure／variance_closure、実行した整数assertionは保持する。科学的条件・結果を変える訂正ではない。
