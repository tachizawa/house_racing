"""版付き CSV を読み込むだけの層。

フィルタリングや型変換のルール（as_of・data_kubun の優先順位・欠損の扱い）は
一切ここに置かない。それらはすべて repository.py に集める。ここに混ぜると、
「どの版を読んでいるか」と「どの行を選ぶか」が同じ場所で絡み合い、
未来情報が紛れ込んでも気づきにくくなる。

読み込みは dtype=str で行う。ketto_num の "0000000001" や jyo_cd の "06" は
先頭ゼロを含む識別子であり、数値として読むと壊れる。数値化は使う側
（repository.py）が個々の列ごとに行う。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import cached_property
from pathlib import Path

import pandas as pd

from keiba.ingest.store import DataVersionError, current_version, read_manifest, version_dir
from keiba.names import normalize_name

__all__ = ["DataStore", "DataVersionError", "coverage_start"]

_JST = timezone(timedelta(hours=9))   # JRA-VAN の日時はすべて日本時間


def _read_csv(path: Path) -> pd.DataFrame:
    """CSV を文字列のまま読む。

    na_filter=False で pandas の欠損値推測を止める。既定のままだと dtype=str
    を指定していても空セルが NaN（float）になり、"0000000001" のような ID 列
    まで巻き添えで壊れることがある。空セルは空文字列のまま持ち、None への変換は
    repository.py 側で列ごとに行う。
    """
    return pd.read_csv(path, dtype=str, na_filter=False)


@dataclass(frozen=True)
class DataStore:
    """1つのデータ版（races / race_horses / horses）を保持するだけの入れ物。"""

    version_id: str
    fetched_at: datetime
    races: pd.DataFrame
    race_horses: pd.DataFrame
    horses: pd.DataFrame
    #: データ版の収録開始（coverage_start()）。「この日より前の出走は版に無い」という
    #: 境界で、過去走が見つからない理由を「取得期間外」と「本当に無い」に分けるのに使う。
    #: 古い manifest には無いので None になりうる（None は「境界が分からない」であって
    #: 「境界が無い」ではない）。
    coverage_from: datetime | None = None

    @classmethod
    def open_current(cls, data_dir: Path) -> "DataStore":
        """CURRENT が指す版を開く。

        CURRENT が無ければ None を返して呼び出し側に判断を委ねるのではなく、
        ここで例外にする。そうしないと「データ版が無い」というチェックが
        Repository を使うすべての場所に散らばってしまう。
        """
        version_id = current_version(data_dir)
        if version_id is None:
            raise DataVersionError(
                "データ版が見つかりません。scripts/update_data.py を実行してください。")
        return cls.open_version(data_dir, version_id)

    @classmethod
    def open_version(cls, data_dir: Path, version_id: str) -> "DataStore":
        """指定した版を開く。

        未完了・壊れた版は read_manifest がそのまま DataVersionError を送出する。
        ここで捕まえて握りつぶすと「読めたことになっている壊れた版」が生まれるので、
        そのまま伝播させる。
        """
        path = version_dir(data_dir, version_id)
        manifest = read_manifest(path)
        return cls(
            version_id=version_id,
            fetched_at=datetime.fromisoformat(manifest["fetched_at"]),
            races=_read_csv(path / "races.csv"),
            race_horses=_read_csv(path / "race_horses.csv"),
            horses=_read_csv(path / "horses.csv"),
            coverage_from=coverage_start(manifest),
        )

    @cached_property
    def name_index(self) -> dict[str, list[str]]:
        """正規化した馬名 → ketto_num のリスト。

        horses（UM）の bamei / bamei_kana と、race_horses（SE）の bamei の
        両方から作る。JVOpen("RACE") は UM を返さないことがあり、horses が
        空になりうる。UM を運ぶ JV-Data の差分配信（DIFF）も差分なので、
        取得期間中に更新の無かった馬は含まれず、たとえ取得できても master が
        完全とは限らない。一方 race_horses には、このデータ版の取得期間に
        走った全頭の bamei・ketto_num が入っている（取得期間の外で休養して
        いた馬は入らない）。「この版に履歴がある」ことと「race_horses に
        出てくる」ことは同値なので、名前の照合が master の完全性という
        運用者の操作次第の前提に依存しないよう、race_horses からも引けるようにする。
        race_horses には bamei_kana が無いので、そちら経由でしか見つからない
        馬は半角カナでの照合ができない（許容する劣化）。できあがった
        HorseMaster からどちらの情報源で見つけた馬かは区別できない。手がかりは
        bamei_kana が None かどうかだけで、models 側にそれを区別する欄は無い。

        馬単位の照合（Repository.find_horses_by_name）で毎回全件を正規化
        していると、18頭 × 六桁のマスタ・出走記録で予想1回あたり数十万回の
        正規化が走る。初回アクセス時に一度だけ作り、DataStore（データ版
        ごとに不変）に持たせる。frozen な dataclass でも cached_property は
        インスタンスの __dict__ を直接書き換えるので、既定値なしのまま
        遅延構築できる。

        DATA_KUBUN_DROP 系のフィルタや ketto_num 単位の重複排除はここでは
        しない。これは素の索引で、業務ルールは repository.py 側に置く
        （このファイル冒頭の docstring の原則のとおり。UM と SE ではデータ
        区分の意味が違うため、どちらの除外集合を使うかもここでは決められない）。
        同じ ketto_num が複数の列・複数の行から同じキーに正規化されることが
        あるので、リストには重複が入りうる。呼び出し側で ketto_num 単位に
        絞り込めば二重には見えない。
        """
        index: dict[str, list[str]] = defaultdict(list)
        for column in ("bamei", "bamei_kana"):
            for ketto_num, value in zip(self.horses["ketto_num"], self.horses[column]):
                if not value:
                    continue
                key = normalize_name(value)
                if key:
                    index[key].append(ketto_num)
        for ketto_num, value in zip(self.race_horses["ketto_num"], self.race_horses["bamei"]):
            if not value:
                continue
            key = normalize_name(value)
            if key:
                index[key].append(ketto_num)
        return dict(index)


def coverage_start(manifest: dict) -> datetime | None:
    """データ版の収録開始。manifest の fromdate と coverage.first_hasso_at の遅いほう。

    どちらか一方では足りない。option 4（セットアップ）では訂正再配信の古いレース
    （1989年など）が混じるので first_hasso_at が fromdate よりずっと前になる。
    option 1 では fromdate を古くしても約1年ぶんしか返らないので、fromdate が
    実際の収録開始より前になる。どちらの値も無ければ None。
    """
    candidates = []
    first = (manifest.get("coverage") or {}).get("first_hasso_at")
    if first:
        candidates.append(datetime.fromisoformat(first))
    fromdate = manifest.get("fromdate")
    if fromdate:
        candidates.append(datetime.strptime(fromdate[:8], "%Y%m%d").replace(tzinfo=_JST))
    return max(candidates) if candidates else None
