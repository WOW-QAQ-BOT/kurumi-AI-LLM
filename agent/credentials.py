# -*- coding: utf-8 -*-
"""凭据管理:Windows 凭据管理器(keyring)与旧版明文 Key 的一次性迁移。"""
import json
import os
from pathlib import Path

try:
    import keyring as _keyring
except ImportError:
    _keyring = None

SERVICE_NAME = "kurumi-agent"


class CredentialStoreError(RuntimeError):
    pass


class CredentialStore:
    """keyring 适配;未安装时读取安全返回 None,写入明确报错。"""

    def __init__(self, backend=None):
        self._backend = backend if backend is not None else _keyring

    def get(self, id: str):
        if self._backend is None:
            return None
        try:
            return self._backend.get_password(SERVICE_NAME, id)
        except Exception:
            return None

    def set(self, id: str, secret: str) -> None:
        if self._backend is None:
            raise CredentialStoreError("未安装 keyring,无法使用 Windows 凭据管理器")
        self._backend.set_password(SERVICE_NAME, id, secret)

    def delete(self, id: str) -> None:
        if self._backend is None:
            return
        try:
            self._backend.delete_password(SERVICE_NAME, id)
        except Exception:
            pass


def migrate_legacy_key(path: Path, store, credential_id: str) -> None:
    """读取明文 api_key → 写入凭据管理器 → 原子移除明文。

    先写 keyring(失败则不动配置),再原子改写 JSON;改写失败则回滚 keyring。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return
    secret = str(data.get("api_key") or "").strip()
    if not secret:
        return
    store.set(credential_id, secret)
    new_data = {k: v for k, v in data.items() if k != "api_key"}
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(new_data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        # 临时文件写入或替换失败:回滚已写入凭据管理器的 Key
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        store.delete(credential_id)
        raise
