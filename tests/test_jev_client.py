"""jev_client のうち、API に触れずに確かめられる部分のテスト。

実通信の確認は scripts/smoke_test.py が担う。ここで守りたいのは
「キーが無いときに何をすればよいか分かるか」と「失敗の理由が読めるか」の2点。
"""

import httpx2
import pytest
import typesafe_sdk as ts

from jev_client import (
    API_KEY_ENV,
    PLACEHOLDER_KEY,
    Job,
    JevConfigError,
    JevRequestError,
    ask,
    ask_parallel,
    describe_error,
    load_api_key,
)


# --- キーの解決 -------------------------------------------------------------

def test_load_api_key_returns_the_configured_value():
    assert load_api_key({API_KEY_ENV: "sk-live-1234"}) == "sk-live-1234"


def test_load_api_key_strips_whitespace_from_env_files():
    # .env に貼り付けると末尾に改行や空白が混ざりやすい
    assert load_api_key({API_KEY_ENV: "  sk-live-1234\n"}) == "sk-live-1234"


def test_load_api_key_missing_tells_the_user_where_to_put_it():
    with pytest.raises(JevConfigError) as excinfo:
        load_api_key({})
    message = str(excinfo.value)
    assert ".env" in message
    assert API_KEY_ENV in message


def test_load_api_key_rejects_the_untouched_placeholder():
    with pytest.raises(JevConfigError):
        load_api_key({API_KEY_ENV: PLACEHOLDER_KEY})


def test_load_api_key_rejects_blank_value():
    with pytest.raises(JevConfigError):
        load_api_key({API_KEY_ENV: "   "})


# --- 例外の日本語化 ---------------------------------------------------------

def test_describe_error_for_401_points_at_the_api_key():
    error = ts.TypeSafeAuthenticationError(401, {"error": "invalid"}, httpx2.Headers({}))
    message = describe_error(error)
    assert API_KEY_ENV in message


def test_describe_error_for_429_reports_how_long_to_wait():
    error = ts.TypeSafeRateLimitError(429, None, httpx2.Headers({"retry-after": "2"}))
    assert "2.0" in describe_error(error)


def test_describe_error_for_timeout_is_not_reported_as_a_connection_failure():
    # TypeSafeAPITimeoutError は TypeSafeAPIConnectionError の派生。
    # 判定順を間違えると「接続できない」と誤報する。
    error = ts.TypeSafeAPITimeoutError(10.0)
    message = describe_error(error)
    assert "10.0" in message
    assert "タイムアウト" in message


def test_describe_error_for_connection_failure_mentions_the_network():
    error = ts.TypeSafeAPIConnectionError("connection refused")
    assert "接続" in describe_error(error)


def test_describe_error_for_400_mentions_the_request_contents():
    error = ts.TypeSafeBadRequestError(400, {"error": "bad questions"}, httpx2.Headers({}))
    message = describe_error(error)
    assert "400" in message


def test_describe_error_falls_back_for_unexpected_exceptions():
    assert describe_error(ValueError("boom")) != ""


# --- 呼び出しのラップ -------------------------------------------------------

class _FakeClient:
    """system_one だけを持つ差し替え用クライアント。"""

    def __init__(self, handler):
        self._handler = handler
        self.calls = []

    def system_one(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        return self._handler(state, questions)


_QUESTIONS = {"q": ts.Noul(instructions="dummy")}


def test_ask_passes_state_and_questions_through():
    client = _FakeClient(lambda state, questions: "ok")
    assert ask(client, "state", _QUESTIONS) == "ok"
    assert client.calls[0][0] == "state"
    assert client.calls[0][1] == _QUESTIONS


def test_ask_converts_sdk_errors_into_jev_request_error():
    def fail(state, questions):
        raise ts.TypeSafeAuthenticationError(401, None, httpx2.Headers({}))

    client = _FakeClient(fail)
    with pytest.raises(JevRequestError) as excinfo:
        ask(client, "state", _QUESTIONS)
    assert API_KEY_ENV in str(excinfo.value)
    assert isinstance(excinfo.value.original, ts.TypeSafeAuthenticationError)


def test_ask_parallel_keys_every_result_by_its_job_name():
    client = _FakeClient(lambda state, questions: f"answer:{state}")
    jobs = {str(n): Job(state=f"horse{n}", questions=_QUESTIONS) for n in range(1, 6)}

    results = ask_parallel(client, jobs, max_workers=3)

    assert results == {str(n): f"answer:horse{n}" for n in range(1, 6)}


def test_ask_parallel_surfaces_the_failing_job_name():
    def fail_on_horse3(state, questions):
        if state == "horse3":
            raise ts.TypeSafeAPITimeoutError(10.0)
        return "ok"

    client = _FakeClient(fail_on_horse3)
    jobs = {str(n): Job(state=f"horse{n}", questions=_QUESTIONS) for n in range(1, 5)}

    with pytest.raises(JevRequestError) as excinfo:
        ask_parallel(client, jobs, max_workers=4)
    assert "3" in str(excinfo.value)


def test_ask_parallel_accepts_an_empty_job_set():
    client = _FakeClient(lambda state, questions: "ok")
    assert ask_parallel(client, {}) == {}


# --- return_exceptions -------------------------------------------------------

class _FailingClient:
    """state が ``fail_on`` に含まれるときだけ失敗する差し替え用クライアント。"""

    def __init__(self, fail_on):
        self._fail_on = fail_on

    def system_one(self, state, questions, **kwargs):
        if state in self._fail_on:
            raise ts.TypeSafeAPITimeoutError(10.0)
        return f"ok:{state}"


def _two_jobs():
    return {name: Job(state=name, questions=_QUESTIONS) for name in ("1", "2")}


class TestAskParallelReturnExceptions:
    def test_default_raises_on_first_failure(self):
        """既定の挙動を変えない。既存の呼び出し側が壊れないこと。"""
        client = _FailingClient(fail_on={"2"})
        with pytest.raises(JevRequestError):
            ask_parallel(client, _two_jobs())

    def test_return_exceptions_puts_error_in_result(self):
        client = _FailingClient(fail_on={"2"})
        results = ask_parallel(client, _two_jobs(), return_exceptions=True)
        assert set(results) == {"1", "2"}
        assert not isinstance(results["1"], JevRequestError)
        assert isinstance(results["2"], JevRequestError)

    def test_return_exceptions_message_names_the_job(self):
        client = _FailingClient(fail_on={"2"})
        results = ask_parallel(client, _two_jobs(), return_exceptions=True)
        assert "[2]" in str(results["2"])

    def test_return_exceptions_with_all_failing(self):
        client = _FailingClient(fail_on={"1", "2"})
        results = ask_parallel(client, _two_jobs(), return_exceptions=True)
        assert all(isinstance(v, JevRequestError) for v in results.values())
