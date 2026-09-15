# -*- coding: utf-8 -*-
"""信任根写入与对话框文案。

拆出来的动机不只是"文件太长":`_persist_allowed_host` 若写死 `DEFAULT_CONFIG_PATH`,就
**直接改主人真实的 api_config.json**,也就无法被单独调用验证。搬到这里后
路径变成参数:调用方可以只作用于临时文件,从而完整检查"凭据管理器写失败必须抛出、
不写配置文件"这类安全语义。

安全约定(与 UI 行为一致,改动需谨慎):

1. **Windows 凭据管理器是唯一授权来源**;它写失败必须抛异常(调用方提示并允许重试),
   绝不允许"只写了 api_config.json 就当确认过";
2. api_config.json 的 `allowed_hosts` 只是**待确认候选**提示,单独填写不会放行。
"""
import json
import os

# 信任根常量住在 api 域(api/config.py):本模块与 ui/__init__.py 都从这里取,
# 依赖方向保持单向的 ui → api,不会绕回来形成循环导入。
from api.config import (
    ALLOWED_HOST_CONFIRMED_VALUE,
    _confirmed_host_account,
    _new_credential_store,
    _normalize_host,
)


def persist_allowed_host(host, cfg_path, store=None):
    """把主人确认的域名写入信任根,并留一条 api_config.json 候选记录。

    `cfg_path` 必须由调用方给出(生产传 DEFAULT_CONFIG_PATH);这样调用方可以用临时文件,
    不必碰真实配置。`store` 为 None 时构造真实凭据管理器。
    """
    if store is None:
        store = _new_credential_store()
    if store is None:
        raise RuntimeError("凭据模块不可用,无法保存域名确认")
    store.set(_confirmed_host_account(host), ALLOWED_HOST_CONFIRMED_VALUE)

    host = _normalize_host(host)
    try:
        with open(cfg_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    hosts = data.get("allowed_hosts")
    hosts = [_normalize_host(h) for h in hosts] if isinstance(hosts, list) else []
    if host not in hosts:
        hosts.append(host)
    data["allowed_hosts"] = hosts
    tmp = cfg_path.with_name(cfg_path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, cfg_path)


def host_confirmation_text(base, official="https://api.deepseek.com") -> str:
    """域名不在信任根内时的确认卡文案(纯函数,便于核对措辞与信息完整度)。"""
    return (
        f"⚠️ API 地址 {base} 不在信任根内(官方: {official}；"
        "或你此前在 Windows 凭据管理器中确认过的地址)。\n"
        f"继续操作会把你的 DeepSeek API Key 发送到 {base}。是否允许?\n"
        "(点「允许」后该确认写入 Windows 凭据管理器；"
        "api_config.json 的 allowed_hosts 只是待确认候选，单独填写不会放行。)"
    )


def migrate_key_text() -> str:
    """明文 Key 迁移确认卡文案。"""
    return "检测到 api_config.json 中保存了明文 API Key。是否将其迁移到 Windows 凭据管理器并删除明文?"
