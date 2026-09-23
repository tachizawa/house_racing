"""JV-Link (JRA-VAN Data Lab の COM コンポーネント) への薄いアクセス層。

ここで引き受けるのは JVOpen/JVGets/JVClose の呼び出しと、走破タイムの整合性
チェックだけ。レコードのデコードと dict 化は records.py、CSV への保存は
store.py に任せる。COM 境界をここだけに薄く保つのがこのモジュールの役目。

pywin32 は Windows 専用の依存なので、モジュールの先頭では import しない。
``open_link`` の中でだけ import することで、このモジュール自体は非 Windows
環境でも import でき、テストも COM に触れずに済む。
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator, Sequence
from typing import Any

__all__ = [
    "JVLinkError",
    "describe_open_code",
    "open_dataspec",
    "open_link",
    "read_open_dataspec",
    "read_records",
    "verify_time_fields",
]


class JVLinkError(RuntimeError):
    """JV-Link の呼び出しに失敗した。"""


def describe_open_code(code: int, dataspec: str | None = None) -> str:
    """JVOpen の戻り値を日本語にする。0 正常 / -1 該当データなし / それ以外はエラー。

    dataspec を渡すと先頭に付ける。複数 dataspec を読むようになったので、
    「JVOpen がエラーを返しました (code=-111)」だけではどの dataspec で
    起きたか分からない。渡さなければ従来どおりの文言（既存の呼び出し側を
    壊さないため）。
    """
    if code == 0:
        message = "正常です。"
    elif code == -1:
        message = "該当データなしです。"
    else:
        message = f"JVOpen がエラーを返しました (code={code})。"
    return f"{dataspec}: {message}" if dataspec is not None else message


def open_link(sid: str = "UNKNOWN") -> Any:
    """JV-Link を初期化して返す。

    非 Windows 環境や pywin32 未導入の環境では ``JVLinkError`` にする。
    こうしておけば、呼び出し側は TypeSafe 側の例外と同じ調子で捕まえられる。
    """
    if sys.platform != "win32":
        raise JVLinkError("JV-Link は Windows 専用です。この環境では利用できません。")
    try:
        import win32com.client
    except ImportError as error:
        raise JVLinkError(
            "pywin32 がインストールされていません。"
            " .venv/Scripts/python.exe -m pip install -r requirements.txt"
            " を実行してください。"
        ) from error

    link = win32com.client.Dispatch("JVDTLab.JVLink")
    init_code = link.JVInit(sid)
    if init_code != 0:
        raise JVLinkError(f"JVInit に失敗しました (code={init_code})。")
    return link


def open_dataspec(link, dataspec, fromdate, option) -> tuple[int, int]:
    """JVOpen を呼び、(open_code, download_count) を返す。JVClose はしない。

    open_code が 0 以外のとき、呼び出し側はレコードを読まずに JVClose する。
    -1（該当データなし）はエラーではないので例外にしない——「今週は更新が
    無かった」と「そもそも取得できていない」を運用者が区別できるように、
    判断材料として返す。
    """
    open_code, _read_count, download_count, _lastfiletime = link.JVOpen(
        dataspec, fromdate, option, 0, 0, "")
    return open_code, download_count


def read_open_dataspec(link, download_count, *, buffer_size=110000) -> Iterator[bytes]:
    """open_dataspec() で既に開いた link から、ダウンロード待ち → JVGets の
    繰り返しでレコードを1件ずつ yield する。

    JVOpen も JVClose もしない。呼び出し側（update_data.py など）が
    open_dataspec() で開いたセッションをそのまま渡し、読み終えたら自分で
    JVClose() すること。JVOpen をもう一度呼び直すと、進行中のダウンロードを
    無駄に取り直すことになる。

    JVGets の戻り値:
       >0  : memview.tobytes() にレコード1件
       -1  : ファイル切替（fname が対象ファイル名）
        0  : EOF
       <-1 : エラー
    レコードは cp932。デコードは records.py 側で行う。
    """
    while link.JVStatus() < download_count:
        time.sleep(0.2)
    while True:
        code, memview, _fname = link.JVGets(bytearray(buffer_size), buffer_size,
                                              bytearray())
        if code == 0:
            return
        if code == -1:          # ファイル切替。レコードではない
            continue
        if code < -1:
            raise JVLinkError(f"JVGets がエラーを返しました (code={code})。")
        yield memview.tobytes()


def read_records(link, dataspec, fromdate, option, *, buffer_size=110000) -> Iterator[bytes]:
    """JVOpen → ダウンロード待ち → JVGets の繰り返し → JVClose。

    内部では open_dataspec() と read_open_dataspec() を組み合わせているだけで、
    シグネチャと振る舞い（-1 なら何も yield せず終わる、0 以外のエラーなら
    JVLinkError、finally で必ず JVClose）は変わらない。dataspec ごとの
    JVOpen 結果を見せずに丸ごと読みたいだけの呼び出し向けに残してある。

    注意: これはジェネレータなので、最後まで回しきるか close() するまで
    JVClose() は呼ばれない。途中で break して放置すると、COM の解放が
    ガベージコレクション任せになる。部分的にしか読まない場合は
    contextlib.closing で包むこと。
    """
    open_code, download_count = open_dataspec(link, dataspec, fromdate, option)
    if open_code == -1:
        link.JVClose()
        return
    if open_code != 0:
        link.JVClose()
        raise JVLinkError(describe_open_code(open_code, dataspec))
    try:
        yield from read_open_dataspec(link, download_count, buffer_size=buffer_size)
    finally:
        link.JVClose()


def verify_time_fields(rows: Sequence[dict]) -> list[dict]:
    """1着馬と2着馬の走破タイム差が、2着馬の margin_sec と一致するか確かめる。

    一致すれば records.margin_sec の解釈（符号1文字 + 0.1秒単位3桁）が正しい。
    手作業のメモにせず、update_data.py --verify-time-fields から再実行できる。
    """
    by_race: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("result_ijyo_cd") in ("1", "2", "3", "4", "5"):
            continue
        if row.get("result_time_sec") is None:
            continue
        by_race.setdefault(row["race_id"], []).append(row)

    report = []
    for race_id, entries in by_race.items():
        # result_kakutei_jyuni は着順がない馬で None になりうる（records.positive_int）。
        # sorted() の key に None が混じると比較できず落ちるので、先に取り除く。
        ranked = sorted(
            (e for e in entries if e.get("result_kakutei_jyuni") is not None),
            key=lambda e: e["result_kakutei_jyuni"],
        )
        first = next((e for e in ranked if e["result_kakutei_jyuni"] == 1), None)
        second = next((e for e in ranked if e["result_kakutei_jyuni"] == 2), None)
        if first is None or second is None:
            continue
        computed = round(second["result_time_sec"] - first["result_time_sec"], 1)
        recorded = second.get("result_margin_sec")
        report.append({
            "race_id": race_id,
            "computed": computed,
            "recorded": recorded,
            "matches": recorded is not None and abs(computed - recorded) < 0.05,
        })
    return report
