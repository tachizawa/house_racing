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
