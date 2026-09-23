"""CSV の版付き保存。

更新途中の CSV を予想が読まないように、次の順で書く。
  1. <version_id>.partial/ に全ファイルを書く
  2. sha256 を計算して manifest を書き、status を complete にする
  3. ディレクトリを <version_id>/ にリネーム（同一 FS なので原子的）
  4. CURRENT を書き換える
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from keiba import config


class DataVersionError(RuntimeError):
    """データ版が読めない・壊れている。"""


RACE_COLUMNS: tuple[str, ...] = (
    "race_id", "year", "monthday", "jyo_cd", "kaiji", "nichiji", "race_num",
    "hasso_at", "kyori_m", "track_cd", "track_kind",
    "jyoken_cd_youngest", "jyoken_cd_all", "grade_cd", "class_level",
    "syubetu_cd", "kigo_cd", "jyuryo_cd", "course_kubun_cd",
    "tenko_cd", "siba_baba_cd", "dirt_baba_cd", "baba_cd",
    "toroku_tosu", "syusso_tosu", "nyusen_tosu",
    "lap_times_sec", "first_3f_sec", "last_3f_sec", "race_name",
    "data_kubun", "make_date", "fetched_at",
)

RACE_HORSE_COLUMNS: tuple[str, ...] = (
    "race_id", "umaban", "wakuban", "ketto_num", "bamei",
    "sex_cd", "barei", "futan_kg", "futan_before_kg",
    "kisyu_code", "kisyu_code_before", "tozai_cd", "chokyosi_code",
    "uma_kigo_cd", "blinker",
    "result_ijyo_cd", "result_kakutei_jyuni", "result_dochaku_kubun",
    "result_dochaku_tosu", "result_time_sec", "result_margin_sec",
    "result_jyuni_1c", "result_jyuni_2c", "result_jyuni_3c", "result_jyuni_4c",
    "result_last_3f_sec", "result_haron_l4_sec",
    "result_ba_taijyu_kg", "result_zogen_sa_kg", "result_odds", "result_ninki",
    "data_kubun", "make_date", "fetched_at",
)

# UM の累計成績列は入れない。過去時点の値を復元できないため。
HORSE_COLUMNS: tuple[str, ...] = (
    "ketto_num", "bamei", "bamei_kana", "birth_date", "sex_cd",
    "keiro_cd", "hinsyu_cd", "del_kubun", "reg_date", "del_date",
    "data_kubun", "make_date", "fetched_at",
)

_FILES = {
    "RA": ("races.csv", RACE_COLUMNS),
    "SE": ("race_horses.csv", RACE_HORSE_COLUMNS),
    "UM": ("horses.csv", HORSE_COLUMNS),
}


def new_version_id() -> str:
    """取得開始時刻の UTC タイムスタンプ。辞書順が時刻順になる。"""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def version_dir(data_dir: Path, version_id: str) -> Path:
    return Path(data_dir) / "versions" / version_id


def current_version(data_dir: Path) -> str | None:
    """CURRENT が指す版。未設定なら None。"""
    marker = Path(data_dir) / "CURRENT"
    if not marker.exists():
        return None
    value = marker.read_text(encoding="utf-8").strip()
    return value or None


def read_manifest(version_path: Path) -> dict:
    """manifest を読む。読めない・完了していないものはすべて拒否する。

    「読めない理由」はすべて DataVersionError に寄せる。呼び出し側は
    「この版は使えるか」だけを知りたいので、JSON の壊れ方まで区別させない。
    manifest.json の書き込み中に異常終了すると JSON が途中で切れるため、
    JSONDecodeError は現実に起こる。
    """
    path = Path(version_path) / "manifest.json"
    if not path.exists():
        raise DataVersionError(f"manifest が見つかりません: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DataVersionError(
            f"manifest が壊れています: {path} / {error}") from error
    if not isinstance(manifest, dict):
        raise DataVersionError(f"manifest の形式が想定と違います: {path}")
    if manifest.get("status") != "complete":
        raise DataVersionError(
            f"データ版 {manifest.get('version_id')} は書き込みが完了していません。"
            " 更新をやり直してください。")
    return manifest


def _cell(value: Any) -> str:
    """CSV の1セル。欠損は空文字列。0 や - に置き換えない。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VersionWriter:
    """1つのデータ版を書く。commit() まで CURRENT は切り替わらない。"""

    def __init__(self, data_dir: Path, version_id: str, *,
                 dataspecs: Sequence[str], fromdate: str, option: int) -> None:
        if isinstance(dataspecs, str):
            # "RACE" をそのまま渡すと 'R','A','C','E' の4個の dataspec として
            # 扱われてしまう。manifest のキーを dataspec から dataspecs に
            # 改名したのと同じ罠なので、ここでも明示的に弾く。
            raise TypeError(
                "dataspecs は文字列ではなく ['RACE', 'DIFF'] のようなシーケンスで"
                " 渡してください。")
        dataspecs = tuple(dataspecs)
        if not dataspecs:
            raise ValueError("dataspecs が空です。何も取得しない版は版として作れません。")
        if len(set(dataspecs)) != len(dataspecs):
            raise ValueError(
                f"dataspecs に重複があります: {dataspecs}"
                "（同じ dataspec を2回読むと、レコードが二重に取り込まれます）")
        self.data_dir = Path(data_dir)
        self.version_id = version_id
        self.dataspecs = dataspecs
        self.fromdate = fromdate
        self.option = option
        self.fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self._partial = version_dir(self.data_dir, version_id + ".partial")
        try:
            # exist_ok=False。new_version_id() は秒精度なので、同一秒に2つの
            # プロセスが走ると同じ .partial を共有し、互いの CSV を truncate
            # （"w" で開き直す）したまま両方が commit を目指す。既にあるなら
            # 進行中か異常終了の残骸なので、ここで止めて運用者に知らせる
            # （fix5 決定9-5）。
            self._partial.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise DataVersionError(
                f"書きかけのディレクトリが既にあります: {self._partial}。"
                " 別のプロセスが同じ版を書いているか、前回が異常終了した残骸です。"
                " 残骸なら削除し、実行中なら終わるまで待ってください。"
            ) from error
        self._counts: dict[str, int] = {"RA": 0, "SE": 0, "UM": 0}
        self._hasso_at: list[str] = []
        self._race_ids: set[str] = set()
        self._handles: dict[str, Any] = {}
        self._writers: dict[str, Any] = {}
        try:
            for spec, (name, columns) in _FILES.items():
                handle = (self._partial / name).open("w", encoding="utf-8", newline="")
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(columns)
                self._handles[spec] = handle
                self._writers[spec] = writer
        except BaseException:
            # 構築が完了していないので呼び出し側は abort() を呼べない（"writer =
            # VersionWriter(...)" の代入自体がまだ終わっていない）。自分が開いた
            # ハンドルと自分が作った .partial は自分で片付けてから投げ直す。
            # KeyboardInterrupt も含めて後始末したいので BaseException で受ける。
            self._close()
            shutil.rmtree(self._partial, ignore_errors=True)
            raise

    def add(self, spec: str, row: dict) -> None:
        if spec not in _FILES:
            return
        _, columns = _FILES[spec]
        enriched = dict(row)
        enriched["fetched_at"] = self.fetched_at
        self._writers[spec].writerow([_cell(enriched.get(c)) for c in columns])
        self._counts[spec] += 1
        if spec == "RA":
            if row.get("hasso_at"):
                self._hasso_at.append(row["hasso_at"])
            self._race_ids.add(row["race_id"])

    def abort(self) -> None:
        """書きかけを捨てる。CURRENT は触らない。"""
        self._close()
        shutil.rmtree(self._partial, ignore_errors=True)

    def commit(self) -> Path:
        """manifest を書き、本番ディレクトリにリネームし、CURRENT を更新する。"""
        self._close()
        files = []
        for spec, (name, _) in _FILES.items():
            path = self._partial / name
            files.append({
                "name": name,
                "rows": self._counts[spec],
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
        manifest = {
            "version_id": self.version_id,
            "schema_version": config.CSV_SCHEMA_VERSION,
            "fetched_at": self.fetched_at,
            "dataspecs": list(self.dataspecs),
            "fromdate": self.fromdate,
            "option": self.option,
            "coverage": {
                "first_hasso_at": min(self._hasso_at) if self._hasso_at else None,
                "last_hasso_at": max(self._hasso_at) if self._hasso_at else None,
                "race_count": len(self._race_ids),
            },
            "files": files,
            "record_counts": dict(self._counts),
            "status": "complete",
        }
        (self._partial / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        final = version_dir(self.data_dir, self.version_id)
        if final.exists():
            # 完成済みの版は不変。黙って上書きすると、予想ジョブが固定したはずの
            # 版の中身が後から変わる。new_version_id() は秒精度なので、短時間の
            # 再実行で version_id が衝突しうる。
            try:
                read_manifest(final)
            except DataVersionError:
                shutil.rmtree(final)   # manifest が無い/未完成なら異常終了の残骸
            else:
                raise DataVersionError(
                    f"データ版 {self.version_id} は既に完成しています。"
                    " 上書きしません。新しい版として取り直してください。")
        self._partial.rename(final)
        (self.data_dir / "CURRENT").write_text(self.version_id, encoding="utf-8")
        return final

    def _close(self) -> None:
        for handle in self._handles.values():
            if not handle.closed:
                handle.close()
