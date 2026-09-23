"""Jev (TypeSafe AI の System One モデル) を叩くための薄いアクセス層。

このモジュールが引き受けるのは次の3つだけ:

1. API キーの解決 (.env → 環境変数) と、未設定時に何をすればよいかを示すこと
2. ``system_one`` 呼び出しのラップと、SDK 例外の日本語化
3. 馬単位の問い合わせ (第2層) のための並列実行

予想ロジックと重みはここには置かない。ここは通信の都合だけを閉じ込める場所。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

import typesafe_sdk as ts
from dotenv import load_dotenv

__all__ = [
    "API_KEY_ENV",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_TIMEOUT",
    "PLACEHOLDER_KEY",
    "Job",
    "JevConfigError",
    "JevError",
    "JevRequestError",
    "ask",
    "ask_parallel",
    "create_client",
    "describe_error",
    "load_api_key",
]

API_KEY_ENV = ts.constants.API_KEY_ENV  # "TYPESAFE_API_KEY"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"

#: .env.example に書いてある値。コピーしただけで書き換え忘れた場合を検出する。
PLACEHOLDER_KEY = "ここにコンソールで取得したキーを貼る"

#: SDK 既定は 10 秒。第2層で全頭を並列に投げると 10 秒では切れることがある。
DEFAULT_TIMEOUT = 30.0

#: レート上限は 1,200 req/min。フルゲート18頭を捌くには十分で、かつ余裕のある値。
DEFAULT_MAX_WORKERS = 8

DEFAULT_MAX_RETRIES = 2


class JevError(RuntimeError):
    """このモジュールが投げる例外の基底。"""


class JevConfigError(JevError):
    """通信する前に分かる設定の誤り (キー未設定など)。"""


class JevRequestError(JevError):
    """API 呼び出しの失敗。``original`` に SDK の元例外を保持する。"""

    def __init__(self, message: str, original: BaseException) -> None:
        super().__init__(message)
        self.original = original


class Job(NamedTuple):
    """1リクエスト分の入力。第2層では馬1頭がこの1件にあたる。"""

    state: Any
    questions: Mapping[str, Any]


# --- キーの解決 -------------------------------------------------------------

_MISSING_KEY_GUIDE = f"""API キー ({API_KEY_ENV}) が設定されていません。

次のどちらかで設定してください:
  1. プロジェクト直下の .env に1行書く   ← 推奨
       {API_KEY_ENV}=sk-...
     (.env.example をコピーして使う: cp .env.example .env)
  2. 環境変数として設定する
       PowerShell : $env:{API_KEY_ENV} = "sk-..."
       bash       : export {API_KEY_ENV}="sk-..."

キーは https://console.typesafe.ai/keys で取得できます。"""


def load_api_key(env: Mapping[str, str] | None = None) -> str:
    """API キーを取り出す。見つからなければ ``JevConfigError``。

    ``env`` を渡さない場合は .env を読み込んだうえで ``os.environ`` を見る。
    テストからは任意のマッピングを渡せる。
    """
    if env is None:
        load_dotenv(DOTENV_PATH)
        env = os.environ

    key = (env.get(API_KEY_ENV) or "").strip()
    if not key:
        raise JevConfigError(_MISSING_KEY_GUIDE)
    if key == PLACEHOLDER_KEY:
        raise JevConfigError(
            f".env の {API_KEY_ENV} が雛形のままです。"
            f"実際のキーに書き換えてください。\n\n{_MISSING_KEY_GUIDE}"
        )
    return key


def create_client(
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    model: str | None = None,
) -> ts.TypeSafeClient:
    """設定済みの ``TypeSafeClient`` を返す。``with`` 文で使える。

    ``RetryPolicy.timeout`` は「再試行を含めた総予算」なので、
    1回分の timeout より短いと最初の試行の途中で打ち切られる。
    試行回数分＋バックオフの余裕を足した値を渡す。
    """
    retry = ts.RetryPolicy(
        max_retries=max_retries,
        timeout=timeout * (max_retries + 1) + 10.0,
    )
    return ts.TypeSafeClient(
        api_key=load_api_key(),
        model=model,
        timeout=timeout,
        retry=retry,
    )


# --- 例外の日本語化 ---------------------------------------------------------

def _api_detail(error: ts.TypeSafeAPIError) -> str:
    request_id = getattr(error, "request_id", None)
    suffix = f" (request_id={request_id})" if request_id else ""
    return f"{error}{suffix}"


# 判定は上から順。派生クラスを先に置くこと。
# 例: TypeSafeAPITimeoutError は TypeSafeAPIConnectionError の派生なので先に来る。
_ERROR_RULES: tuple[tuple[type[BaseException], Callable[[Any], str]], ...] = (
    (
        ts.TypeSafeAuthenticationError,
        lambda e: f"認証に失敗しました (401)。{API_KEY_ENV} の値が誤っているか失効しています。"
        f" https://console.typesafe.ai/keys で確認してください。 / {_api_detail(e)}",
    ),
    (
        ts.TypeSafePermissionDeniedError,
        lambda e: f"このキーでは許可されていない操作です (403)。 / {_api_detail(e)}",
    ),
    (
        ts.TypeSafeRateLimitError,
        lambda e: "レート上限に達しました (429)。"
        + (
            f" {e.retry_after_ms / 1000:.1f} 秒待ってから再試行してください。"
            if e.retry_after_ms
            else " しばらく待ってから再試行してください。"
        )
        + f" 並列数 (max_workers) を下げることも検討してください。 / {_api_detail(e)}",
    ),
    (
        ts.TypeSafeNotFoundError,
        lambda e: f"エンドポイントまたはモデルが見つかりません (404)。 / {_api_detail(e)}",
    ),
    (
        ts.TypeSafeBadRequestError,
        lambda e: f"リクエストが不正です (400)。state か questions の作り方を見直してください。"
        f" / {_api_detail(e)}",
    ),
    (
        ts.TypeSafeUnprocessableEntityError,
        lambda e: f"リクエストの内容を処理できません (422)。質問の criteria や"
        f" state の型を確認してください。 / {_api_detail(e)}",
    ),
    (
        ts.TypeSafeInternalServerError,
        lambda e: f"TypeSafe 側のサーバーエラーです ({e.status})。時間をおいて再試行してください。"
        f" / {_api_detail(e)}",
    ),
    (
        ts.TypeSafeAPIResponseValidationError,
        lambda e: f"レスポンスが想定の形式と一致しません (項目: {e.field_path})。"
        f" SDK のバージョンを確認してください。 / {_api_detail(e)}",
    ),
    (
        ts.TypeSafeAPITimeoutError,
        lambda e: f"リクエストがタイムアウトしました (timeout={e.timeout})。"
        f" create_client(timeout=...) を伸ばすか、1リクエストあたりの state を小さくしてください。",
    ),
    (
        ts.TypeSafeAPIConnectionError,
        lambda e: f"API に接続できませんでした。ネットワークとプロキシ設定を確認してください。 / {e}",
    ),
    (
        ts.TypeSafeAPIError,
        lambda e: f"API がエラーを返しました ({e.status})。 / {_api_detail(e)}",
    ),
    (
        ts.TypeSafeError,
        lambda e: f"SDK がリクエストを拒否しました。questions が空か、Score の criteria が"
        f" 空リストになっていないか確認してください。 / {e}",
    ),
)


def describe_error(error: BaseException) -> str:
    """SDK 例外を、次に何をすればよいか分かる日本語メッセージに変換する。"""
    for error_type, render in _ERROR_RULES:
        if isinstance(error, error_type):
            return render(error)
    return f"想定外のエラーです ({type(error).__name__})。 / {error}"


# --- 呼び出し ---------------------------------------------------------------

def ask(
    client: Any,
    state: Any,
    questions: Mapping[str, Any],
    *,
    timeout: float | None = None,
) -> Any:
    """``system_one`` を1回呼ぶ。失敗は ``JevRequestError`` に変換する。"""
    kwargs: dict[str, Any] = {}
    if timeout is not None:
        kwargs["timeout"] = timeout
    try:
        return client.system_one(state, questions, **kwargs)
    except ts.TypeSafeError as error:
        raise JevRequestError(describe_error(error), error) from error


def ask_parallel(
    client: Any,
    jobs: Mapping[str, Job],
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    timeout: float | None = None,
    return_exceptions: bool = False,
) -> dict[str, Any]:
    """複数の state を並列に問い合わせ、ジョブ名をキーにした辞書で返す。

    第2層 (馬単位) 用。state が馬ごとに違うためリクエストを分けざるを得ないが、
    1頭分の中の質問はすべて1リクエストにまとめること。

    ``return_exceptions=True`` のとき、失敗したジョブの値は ``JevRequestError``
    のインスタンスそのものになる。1頭の失敗で全体を止めたくない用途
    (バックテスト) 向け。既定は False で、最初の失敗をそのまま投げる。
    """
    if not jobs:
        return {}

    def run(item: tuple[str, Job]) -> tuple[str, Any]:
        name, job = item
        try:
            return name, ask(client, job.state, job.questions, timeout=timeout)
        except JevRequestError as error:
            wrapped = JevRequestError(f"[{name}] {error}", error.original)
            if return_exceptions:
                return name, wrapped
            raise wrapped from error

    with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as pool:
        return dict(pool.map(run, jobs.items()))
