"""バックテストの usage が、答えが欠けた応答の利用量も数えることの確認。

応答が返った時点で課金は確定している。Top3IncompleteError で飛ばしたレースでも
その応答の requests / input_tokens を usage に残す（PR #2 レビュー 4297266007）。
"""

from __future__ import annotations

import types

from keiba.evaluate import top3_backtest as tbt
from tests.test_top3_backtest import PlanBClient, _run_b


class MissingPaceClient(PlanBClient):
    """pace の質問に、pace の答えが無い応答を返す。"""

    def system_one(self, state, questions, **kw):
        self.calls += 1
        return types.SimpleNamespace(nouls={}, choices={}, model="jev-1.13.0",
                                     usage=types.SimpleNamespace(input_tokens=123))


def test_missing_pace_answer_usage_is_counted(tmp_path):
    client = MissingPaceClient()
    rows = _run_b(tmp_path, client)
    assert client.calls == 1
    assert rows[0].skipped == "incomplete_answer"
    usage = rows[0].usage["plan_b"]
    assert usage["requests"] == 1
    assert usage["input_tokens"] == 123
    assert usage["failed_requests"] == 0
    rep = tbt.report(rows, n=3, run_info={"note": "test"})
    assert rep["usage"]["plan_b"]["input_tokens"] == 123
    assert rep["usage"]["plan_b"]["requests"] == 1
