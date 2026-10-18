"""scripts/ml/check_offsets.py の照合ロジック（SDK の書き方の揺れを拾えるか）。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("check_offsets", ROOT / "scripts/ml/check_offsets.py")
check_offsets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_offsets)

STRUCT = '''
class JV_HN_HANSYOKU:
    def SetDataB(self, bBuff):
        self.HansyokuNum = MidB2S(bBuff, 12, 10)
        self.KettoNum = MidB2S(bBuff, 30, 10)
        self.Bamei = MidB2S(bBuff, 41, 36)
        self.HansyokuFNum = MidB2S(bBuff, 230, 10)

class JV_UM_UMA:
    def SetDataB(self, bBuff):
        for i in range(14):
            self.Ketto3Info[i].SetDataB(MidB2B(bBuff, 205 + (46 * i), 46))
'''


def test_extracts_plain_array_and_substruct_calls():
    block = check_offsets.class_block(STRUCT, ("JV_UM_UMA",))
    assert check_offsets.extract(block) == [("self.Ketto3Info[i]", "205 + (46 * i)", 46)]
    hn = check_offsets.extract(check_offsets.class_block(STRUCT, ("JV_HN_HANSYOKU",)))
    assert ("self.HansyokuFNum", "230", 10) in hn


def test_main_reports_mismatch(tmp_path, capsys):
    path = tmp_path / "JVData_Struct.py"
    path.write_text(STRUCT, encoding="utf-8")
    assert check_offsets.main(["--struct", str(path)]) == 1     # SK が無い・HN の項目不足
    out = capsys.readouterr().out
    assert "OK  father_num" in out
    assert "NG  mother_num" in out


def _sdk500_style_struct() -> str:
    """SDK 5.0.0 の Python 版の書き方（代入せずに並べる・3代血統は46バイトの部分構造体）で、
    こちらの期待する位置をすべて含む構造体定義を作る。"""
    lines = []
    for spec, (class_names, expected) in check_offsets.EXPECTED.items():
        lines += [f"class {class_names[0]}:", "    def SetDataB(self, b):", "        return ("]
        for name, pos, length in expected:
            if spec == "UM" and name.startswith("ketto3"):
                lines.append(f"            [KETTO3_INFO.SetDataB(MidB2B(b, {pos} + 46 * i, 46))"
                             " for i in range(14)],")
            elif name.endswith("[0]"):
                lines.append(f"            [MidB2S(b, {pos} + {length} * i, {length})"
                             " for i in range(14)],")
            else:
                lines.append(f"            MidB2S(b, {pos}, {length}),")
        lines += ["        )", ""]
    return "\n".join(lines)


def test_main_accepts_sdk500_style_when_everything_matches(tmp_path, capsys):
    path = tmp_path / "JVData_Struct.py"
    path.write_text(_sdk500_style_struct(), encoding="utf-8")
    assert check_offsets.main(["--struct", str(path)]) == 0
    out = capsys.readouterr().out
    assert "OK  ketto3[0]" in out          # 長さ 46 で書かれた3代血統も一致とみなす
    assert "NG " not in out


def test_base_offset_does_not_depend_on_how_the_expression_is_written():
    assert check_offsets.base_offset("205 + (46 * i)") == 205
    assert check_offsets.base_offset("(i * 46) + 205") == 205
    assert check_offsets.base_offset("12") == 12
    assert check_offsets.base_offset("pos()") is None


def test_undecodable_struct_file_returns_2(tmp_path, capsys):
    path = tmp_path / "JVData_Struct.py"
    path.write_bytes(b"\xff\xfe\x81")      # UTF-8 でも CP932 でも読めない
    assert check_offsets.main(["--struct", str(path)]) == 2
    assert "読めません" in capsys.readouterr().out


def test_only_sdk_conversion_calls_count_as_field_reads():
    """MidB2S / MidB2B 以外の3引数の呼び出しを、項目の読み出しとして拾わない。"""
    block = ("        helper(bBuff, 12, 10),\n"
             "        self.KettoNum = MidB2S(bBuff, 12, 10)\n"
             "        MidB2B(bBuff, 205 + (46 * i), 46),\n")
    assert check_offsets.extract(block) == [("self.KettoNum", "12", 10),
                                            ("", "205 + (46 * i)", 46)]
