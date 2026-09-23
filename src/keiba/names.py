"""馬名の照合用正規化。

`data/repository.py`（マスタ側の索引を作る）と `input/match.py`（利用者の
入力を正規化する）の両方がこの関数を必要とする。`input/` を `data/` から
import すると層が逆転する（データ層が入力層に依存してしまう）ため、
どちらにも属さないこのモジュールに独立させてある。
"""

from __future__ import annotations

import re
import unicodedata

# 馬名の表記ゆれで意味を持たない記号。全角・半角の空白、中黒、ハイフン類。
_NOISE = re.compile(r"[\s　・･\-‐‑‒–—―ー]+")


def normalize_name(name: str) -> str:
    """馬名を照合用に正規化する。

    NFKC で全角・半角を揃えたあと、意味を持たない記号を落として大文字化する。
    カタカナはカタカナのまま残す（ひらがなに寄せると別の馬と衝突しうる）。

    注意: NFKC は長音符「ー」を変換しないので _NOISE で落としている。
    馬名の長音は表記ゆれが多く、照合のノイズになるため。
    """
    folded = unicodedata.normalize("NFKC", name)
    return _NOISE.sub("", folded).upper()
