"""血統レコードの位置（オフセット）を SDK の JVData_Struct.py と照合する。

    .venv/Scripts/python.exe scripts/ml/check_offsets.py

keiba_ml/ingest/pedigree_records.py の位置は仕様書から写したもので、この PC 以外の
環境では SDK と突き合わせられていない。SDK の Python 版構造体定義をテキストとして
読み（import はしない。vendor/ は再配布不可なので依存させない）、各レコードの
SetDataB に現れる (位置, 長さ) の組に、こちらが使う組が含まれているかを調べる。

照合は名前ではなく (位置, 長さ) の組で行う。SDK 側の変数名の付け方に依存しないため。
配列（3代血統の14枠など）は先頭の位置だけを比べる（位置の式を添字 0 で計算する）。
"""

from __future__ import annotations

import argparse
import ast
import operator
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from keiba import console  # noqa: E402
from keiba_ml.ingest import pedigree_records as pr  # noqa: E402

DEFAULT_STRUCT = (ROOT / "vendor" / "jravan-sdk-500-64bit" / "JV-Data構造体" / "Python版"
                  / "JVData_Struct.py")

#: レコード種別 → (SDK のクラス名の候補, こちらが使う (名前, 位置, 長さ) の一覧)
EXPECTED = {
    "SK": (("JV_SK_SANKU",), [
        ("ketto_num", *pr.SK_KETTO_NUM), ("birth_date", *pr.SK_BIRTH_DATE),
        ("sex_cd", *pr.SK_SEX_CD), ("hinsyu_cd", *pr.SK_HINSYU_CD),
        ("keiro_cd", *pr.SK_KEIRO_CD), ("mochi_kubun", *pr.SK_MOCHI_KUBUN),
        ("import_year", *pr.SK_IMPORT_YEAR), ("breeder_code", *pr.SK_BREEDER_CODE),
        ("sanchi_name", *pr.SK_SANCHI_NAME),
        ("hansyoku_num[0]", pr.SK_HANSYOKU_START, pr.SK_HANSYOKU_LENGTH),
    ]),
    "HN": (("JV_HN_HANSYOKU",), [
        ("hansyoku_num", *pr.HN_HANSYOKU_NUM), ("ketto_num", *pr.HN_KETTO_NUM),
        ("bamei", *pr.HN_BAMEI), ("bamei_kana", *pr.HN_BAMEI_KANA),
        ("bamei_eng", *pr.HN_BAMEI_ENG), ("birth_year", *pr.HN_BIRTH_YEAR),
        ("sex_cd", *pr.HN_SEX_CD), ("hinsyu_cd", *pr.HN_HINSYU_CD),
        ("keiro_cd", *pr.HN_KEIRO_CD), ("mochi_kubun", *pr.HN_MOCHI_KUBUN),
        ("import_year", *pr.HN_IMPORT_YEAR), ("sanchi_name", *pr.HN_SANCHI_NAME),
        ("father_num", *pr.HN_FATHER_NUM), ("mother_num", *pr.HN_MOTHER_NUM),
    ]),
    "UM": (("JV_UM_UMA",), [
        ("ketto_num", *pr.UM_KETTO_NUM), ("birth_date", *pr.UM_BIRTH_DATE),
        ("bamei", *pr.UM_BAMEI), ("sex_cd", *pr.UM_SEX_CD),
        # SDK は3代血統を「番号10 + 馬名36 = 46バイト」の部分構造体で読むことがある。
        # 長さ 10 と 46 のどちらで書かれていても先頭位置が合えば一致とみなす。
        ("ketto3[0]", pr.UM_KETTO3_START,
         (pr.UM_KETTO3_NUM_LENGTH, pr.UM_KETTO3_NUM_LENGTH + pr.UM_KETTO3_NAME_LENGTH)),
    ]),
}

# self.KettoNum = MidB2S(bBuff, 12, 10)
# self.HansyokuNum[i] = MidB2S(bBuff, 67 + (10 * i), 10)
# self.Ketto3Info[i].SetDataB(MidB2B(bBuff, 205 + (46 * i), 46)) のような書き方を拾う。
# SDK 5.0.0 の Python 版は代入せずに並べる（MidB2S(b, 12, 10), や
# [MidB2S(b, 67 + 10 * i, 10) for i in range(14)]）ので、変数名の部分は省略可にする。
# 変数名が無くても拾うので、呼び出しは SDK の変換関数（MidB2S / MidB2B）だけに限る。
_CALL = re.compile(r"(?:([\w.\[\]]+)\s*(?:=\s*|\.SetDataB\(\s*))?"
                   r"\bMidB2[SB]\(\s*\w+\s*,\s*([^,()]+(?:\([^()]*\)[^,()]*)*?)\s*,\s*(\d+)\s*\)")


def class_block(text: str, names: tuple[str, ...]) -> str | None:
    for name in names:
        match = re.search(rf"^class\s+{name}\b.*?(?=^class\s|\Z)", text, re.S | re.M)
        if match:
            return match.group(0)
    return None


def extract(block: str) -> list[tuple[str, str, int]]:
    """(変数名, 位置の式, 長さ) を拾う。位置が式（配列の添字計算）でもそのまま残す。

    変数名の無い書き方では変数名を "" にする。
    """
    return [(m.group(1) or "", m.group(2), int(m.group(3))) for m in _CALL.finditer(block)]


_BINARY_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
               ast.FloorDiv: operator.floordiv}


def base_offset(expr: str) -> int | None:
    """位置の式を、添字などの名前をすべて 0 として計算する（= 配列の先頭の位置）。

    "205 + (46 * i)" も "(i * 46) + 205" も 205 になる。整数と + - * // 以外を含む式は
    計算せず None を返す（eval はしない）。
    """
    def walk(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            return 0
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            return _BINARY_OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        raise ValueError(expr)

    try:
        return walk(ast.parse(expr, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None


def main(argv: list[str] | None = None) -> int:
    console.enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--struct", type=Path, default=DEFAULT_STRUCT,
                        help="JVData_Struct.py の場所（既定 %(default)s）")
    parser.add_argument("--show", action="store_true", help="SDK 側で拾った全項目を表示する")
    args = parser.parse_args(argv)
    if not args.struct.exists():
        print(f"見つかりません: {args.struct}")
        return 2
    raw = args.struct.read_bytes()
    for encoding in ("utf-8", "cp932"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        print(f"UTF-8 でも CP932 でも読めません: {args.struct}")
        return 2
    exit_code = 0
    for spec, (class_names, expected) in EXPECTED.items():
        block = class_block(text, class_names)
        if block is None:
            print(f"[{spec}] クラス {class_names} が見つかりません。--show で SDK 側を確認してください。")
            exit_code = 1
            continue
        found = extract(block)
        pairs = {(base_offset(expr), length) for _, expr, length in found}
        print(f"[{spec}] {class_names[0]}: SDK 側で {len(found)} 項目を検出")
        if args.show:
            for name, expr, length in found:
                print(f"    {name or '-':<28} pos={expr:<16} len={length}")
        for name, pos, length in expected:
            lengths = length if isinstance(length, tuple) else (length,)
            ok = any((pos, n) in pairs for n in lengths)
            shown = "/".join(str(n) for n in lengths)
            print(f"    {'OK ' if ok else 'NG '} {name:<24} pos={pos:<5} len={shown}")
            if not ok:
                exit_code = 1
    if exit_code:
        print("\nNG があります。pedigree_records.py の位置を SDK に合わせて直してください。"
              "（SDK の書き方によっては照合自体が拾えていないこともあるので --show も見ること）")
    else:
        print("\nすべて一致しました。")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
