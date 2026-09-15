# -*- coding: utf-8 -*-
"""凭据与主机授权(从 UI.py 拆出)。

拆出来的这五个函数与 `UI.py` 的模块级状态无关(`load_api_config` 会读 `_HERE`/
`_HAS_OPENAI`;为了保留"可以按需替换这两个名字"的口子,它们**仍留在 UI.py**)。

`OFFICIAL_API_HOSTS` / `ALLOWED_HOST_ACCOUNT_PREFIX` 只是常量,从 `ui_constants`
导入即可;本模块**不得**导入 `UI.py`(会循环导入)。
"""
from ui_constants import ALLOWED_HOST_ACCOUNT_PREFIX, OFFICIAL_API_HOSTS


def _normalize_host(value):
    return str(value or "").strip().lower().rstrip("/")


def _confirmed_host_account(host):
    """用户已确认主机在 Windows 凭据管理器中的 account 名。"""
    return ALLOWED_HOST_ACCOUNT_PREFIX + _normalize_host(host)


def _new_credential_store():
    """构造凭据管理器适配器（keyring 缺失时 get 返回 None，set 明确报错）。

    在函数内导入，既避免 agent 依赖影响模块加载，也便于调用方替换
    agent.credentials.CredentialStore。
    """
    try:
        from agent.credentials import CredentialStore
        return CredentialStore()
    except Exception:
        return None


def _host_is_allowed(base_url, candidate_hosts=None):
    """凭据外发域名授权。信任根只有两个：

    1. 内置官方域名（OFFICIAL_API_HOSTS）；
    2. 用户曾在界面确认卡上点过「允许」、并写入 Windows 凭据管理器的主机。

    api_config.json 的 allowed_hosts 只是「待确认候选」提示，单独存在不再授权——
    否则能改写该文件的攻击者追加一个域名，就能让凭据静默外发（白名单自证）。
    candidate_hosts 仅用于生成更清楚的错误提示，不参与放行判定。
    """
    host = _normalize_host(base_url)
    if not host:
        return False
    if host in OFFICIAL_API_HOSTS:
        return True
    store = _new_credential_store()
    if store is None:
        return False
    try:
        confirmed = store.get(_confirmed_host_account(host))
    except Exception:
        return False
    return bool(confirmed)


def _is_candidate_host(host, candidate_hosts):
    """api_config.json 的 allowed_hosts 里是否列了该地址（仅提示用）。"""
    target = _normalize_host(host)
    return any(_normalize_host(h) == target for h in (candidate_hosts or []))
