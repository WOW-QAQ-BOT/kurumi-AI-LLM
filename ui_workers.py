# -*- coding: utf-8 -*-
"""后台线程(QThread):本地模型加载/生成、记忆整理、API 聊天与记忆(从 UI.py 拆出)。

这些类只依赖 Qt 与(本地路径才需要的)推理栈,不依赖 `UI.py` 里的任何状态。
拆分后的硬约束:**本模块不得导入 `UI.py`**(会循环导入)。需要的常量走 `ui_constants`。

**推理栈采用延迟导入**。纯 API 模式(不装本地推理栈,见 requirements.txt 里
「本地模型 + LoRA」那一段)不装
torch/transformers,因此本模块在导入期只能依赖 Qt/openai/标准库 —— `_TokenStop` 的基类
在运行时按需解析,本地类里用到的推理栈在方法内导入。这样"没有本地模型依赖"的环境
也能启动界面、跑完整离线流程。
"""

import queue
from threading import Thread
from typing import TYPE_CHECKING

from openai import OpenAI
from PySide6.QtCore import QThread, Signal

from chat_params import _chat_create, _take_thinking_degradation, _thinking_kwargs
from kurumi_memory import extract_prompt, history_to_text, parse_memories
from runtime_control import CancellationToken
from ui_constants import GEN_JOIN_TIMEOUT_S, MEMORY_MAX_NEW_TOKENS, MIN_NEW_TOKENS, REPETITION_PENALTY

if TYPE_CHECKING:      # 仅供类型检查/IDE,运行期不导入推理栈
    from transformers import StoppingCriteria as _StoppingCriteriaBase
else:
    # 基类占位:本地生成路径用到时再换成真正的 transformers.StoppingCriteria
    # (见 _stopping_base)。这样 `import ui_workers` 不再触发 4 GB 依赖。
    class _StoppingCriteriaBase:
        pass


def _stopping_base():
    """取出真正的 StoppingCriteria 基类(未安装 transformers 时沿用占位)。"""
    try:
        from transformers import StoppingCriteria
        return StoppingCriteria
    except ImportError:
        return _StoppingCriteriaBase


class _TokenStop(_StoppingCriteriaBase):
    """用取消令牌停止本地生成。

    `transformers.StoppingCriteria` 的 `__call__` 语义是"返回 True 即停止",
    本地生成前会通过 `_stopping_base()` 校验依赖是否可用(不可用则在更早处报错)。
    """

    def __init__(self, token):
        super().__init__()
        self._token = token

    def __call__(self, input_ids, scores, **kwargs):
        return self._token.cancelled



class ModelLoader(QThread):
    loaded = Signal(object, object)   # (model, tokenizer)
    failed = Signal(str)

    def __init__(self, base, adapter, use_quantize, cancel_token=None):
        super().__init__()
        self.base, self.adapter, self.use_quantize = base, adapter, use_quantize
        self.cancel_token = cancel_token or CancellationToken()

    def cancel(self):
        """协作式取消：设置令牌后，加载线程在下一个检查点自行退出。

        from_pretrained 本身不可中断，只能在步骤之间检查，因此这里只能缩短而非瞬间终止。
        """
        self.cancel_token.cancel()

    def run(self):
        try:
            # 推理栈在这里才导入(没有它时给出可读的原因,而不是 ImportError 栈)
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            kwargs = dict(device_map="auto", dtype=torch.bfloat16)
            if self.use_quantize:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                )
            if self.cancel_token.cancelled:
                return
            model = AutoModelForCausalLM.from_pretrained(self.base, **kwargs)
            if self.cancel_token.cancelled:
                return
            if self.adapter:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, self.adapter)
                model = model.merge_and_unload()
            if hasattr(model, "generation_config"):
                model.generation_config.enable_thinking = False
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(self.base)
            if self.cancel_token.cancelled:
                return   # 已取消：不再把模型交给 UI（窗口可能已关闭）
            self.loaded.emit(model, tokenizer)
        except Exception as e:
            self.failed.emit(str(e))


class GenerationWorker(QThread):
    chunk = Signal(str)          # 增量文本(delta)
    done = Signal(str)           # 成功（完整回复）
    failed = Signal(str)         # 生成异常（不当作回复保存）
    stale = Signal(str)          # 超时/取消后底层 generate 线程仍未退出：期间禁止新生成
    stale_cleared = Signal()     # 底层 generate 线程已真正退出，可以再次生成

    def __init__(self, model, tokenizer, messages, args, cancel_token=None):
        super().__init__()
        self.model, self.tokenizer, self.messages, self.args = model, tokenizer, messages, args
        self.cancel_token = cancel_token or CancellationToken()
        self._gen_error = None

    def cancel(self):
        self.cancel_token.cancel()

    def _generate_safe(self, gen_kwargs):
        """在子线程里跑 model.generate；异常时记下并确保流式循环能结束。"""
        try:
            self.model.generate(**gen_kwargs)
        except Exception as e:
            self._gen_error = str(e)
            try:
                gen_kwargs["streamer"].end()
            except Exception:
                pass

    def _watch_stale(self, gen_thread):
        """等遗留的 generate 线程真正结束，再通知 UI 解除「禁止新生成」封锁。"""
        try:
            gen_thread.join()
        except Exception:
            pass
        try:
            self.stale_cleared.emit()
        except Exception:
            pass

    def _report_stale(self, gen_thread, message):
        """把仍在运行的 generate 线程交给 UI 关注：取消令牌 + 封锁新生成 + 结束通知。"""
        self.cancel_token.cancel()
        self.stale.emit(message)
        Thread(target=self._watch_stale, args=(gen_thread,), daemon=True).start()

    def run(self):
        try:
            from transformers import StoppingCriteriaList, TextIteratorStreamer

            text = self.tokenizer.apply_chat_template(
                self.messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True,
                timeout=self.args.stream_stall_timeout,
            )
            gen_kwargs = dict(
                **inputs, max_new_tokens=self.args.max_new_tokens, min_new_tokens=MIN_NEW_TOKENS,
                do_sample=True, temperature=self.args.temperature, top_p=self.args.top_p,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
            )
            if self.cancel_token is not None:
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_TokenStop(self.cancel_token)])
            gen_thread = Thread(target=self._generate_safe, args=(gen_kwargs,), daemon=True)
            gen_thread.start()
            partial = ""
            try:
                for token in streamer:
                    if self.cancel_token.cancelled:
                        break
                    # 只发增量：UI 侧自行累加，避免每 token 重发全文(O(n²) 拷贝)
                    partial += token
                    self.chunk.emit(token)
            except queue.Empty:
                # 流停滞：取消令牌并确认底层 generate 是否已真正退出
                message = f"生成停滞超过 {self.args.stream_stall_timeout} 秒，已停止"
                self.cancel_token.cancel()
                gen_thread.join(timeout=GEN_JOIN_TIMEOUT_S)
                if gen_thread.is_alive():
                    self._report_stale(gen_thread, message)
                else:
                    self.failed.emit(message)
                return
            was_cancelled = self.cancel_token.cancelled
            gen_thread.join(timeout=GEN_JOIN_TIMEOUT_S)
            if gen_thread.is_alive():
                # 超时/取消后底层 generate 仍在跑：必须先 cancel，再封锁新生成，
                # 否则 UI 已回 IDLE 允许再次发送 → 两次 generate 抢同一份 4-bit 权重。
                if self._gen_error:
                    message = self._gen_error
                elif was_cancelled:
                    message = "已取消（本地生成尚未完全停止）"
                else:
                    message = "生成超时（本地模型无响应），请重试"
                self._report_stale(gen_thread, message)
            elif self._gen_error:
                self.failed.emit(self._gen_error)
            elif was_cancelled:
                self.failed.emit("已取消")
            else:
                self.done.emit(partial)
        except Exception as e:
            self.failed.emit(str(e))


class MemoryWorker(QThread):
    done = Signal(object, str)   # (解析出的记忆条目列表, 失败原因)；entries 为 None 表示失败

    def __init__(self, model, tokenizer, history, args, cancel_token=None):
        super().__init__()
        self.model, self.tokenizer, self.history, self.args = model, tokenizer, history, args
        self.cancel_token = cancel_token or CancellationToken()

    def cancel(self):
        """协作式取消：generate 通过 StoppingCriteria 在每个解码步检查该令牌。"""
        self.cancel_token.cancel()

    def run(self):
        try:
            # 先查取消:已取消的 worker 不该因为「本地依赖缺失」而报出另一种原因
            if self.cancel_token.cancelled:
                self.done.emit(None, "已取消")
                return
            from transformers import StoppingCriteriaList  # 本地路径才需要

            prompt = extract_prompt(history_to_text(self.history))
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            out = self.model.generate(
                **inputs, max_new_tokens=MEMORY_MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([_TokenStop(self.cancel_token)]),
            )
            if self.cancel_token.cancelled:
                self.done.emit(None, "已取消")
                return
            reply = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            self.done.emit(parse_memories(reply), "")
        except Exception as e:
            self.done.emit(None, str(e))


class ApiChatWorker(QThread):
    chunk = Signal(str)
    done = Signal(str)    # 成功
    failed = Signal(str)  # 失败（不当作回复保存）
    degraded = Signal(str)  # 思考参数不被端点支持而降级（只需提示一次）

    def __init__(self, cfg, messages, args, cancel_token=None):
        super().__init__()
        self.cfg, self.messages, self.args = cfg, messages, args
        self.cancel_token = cancel_token or CancellationToken()
        self._stream = None

    def cancel(self):
        self.cancel_token.cancel()
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass

    def run(self):
        try:
            client = OpenAI(
                api_key=self.cfg["api_key"], base_url=self.cfg["base_url"],
                timeout=self.args.api_timeout_seconds,
            )
            stream = _chat_create(
                client,
                model=self.cfg["model"],
                messages=self.messages,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                max_tokens=self.args.max_new_tokens,
                stream=True,
                **_thinking_kwargs(getattr(self.args, "api_thinking", "disabled")),
            )
            warned = _take_thinking_degradation()
            if warned:
                # 降级不能让用户以为 --api_thinking 生效了:上屏说明(不影响本次回复)
                self.degraded.emit(warned)
            self._stream = stream
            partial = ""
            reasoning_chars = 0
            for chunk in stream:
                if self.cancel_token.cancelled:
                    self.failed.emit("已取消")
                    return
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue
                # 思考内容不展示,但要计数:它与正文共享 max_tokens,是"空回复"的根因
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    reasoning_chars += len(reasoning)
                if delta.content:
                    partial += delta.content
                    # 只发增量：与 GenerationWorker 一致，UI 侧自行累加
                    self.chunk.emit(delta.content)
            if not partial.strip():
                # 正文为空绝不能当正常回复:否则界面出现空气泡、history 里塞进空 assistant
                mode = getattr(self.args, "api_thinking", "disabled")
                if reasoning_chars:
                    self.failed.emit(
                        f"本轮未产出正文：模型输出了 {reasoning_chars} 字思考内容并占满 "
                        f"max_tokens={self.args.max_new_tokens}（当前 --api_thinking={mode}）；"
                        "用 --api_thinking disabled 可省下这部分开销，必要时同时提高 --max_new_tokens")
                else:
                    self.failed.emit("本轮未产出正文：模型返回空内容，请重试")
                return
            self.done.emit(partial)
        except Exception as e:
            self.failed.emit(str(e))


class ApiMemoryWorker(QThread):
    done = Signal(object, str)   # (记忆条目列表, 失败原因)；entries 为 None 表示失败
    degraded = Signal(str)       # 思考参数不被端点支持而降级（只需提示一次）

    def __init__(self, cfg, history, args, cancel_token=None):
        super().__init__()
        self.cfg, self.history, self.args = cfg, history, args
        self.cancel_token = cancel_token or CancellationToken()

    def cancel(self):
        self.cancel_token.cancel()

    def run(self):
        try:
            if self.cancel_token.cancelled:
                self.done.emit(None, "已取消")
                return
            prompt = extract_prompt(history_to_text(self.history))
            client = OpenAI(api_key=self.cfg["api_key"], base_url=self.cfg["base_url"])
            resp = _chat_create(
                client,
                model=self.cfg["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=MEMORY_MAX_NEW_TOKENS,
                stream=False,
                **_thinking_kwargs(getattr(self.args, "api_thinking", "disabled")),
            )
            warned = _take_thinking_degradation()
            if warned:
                self.degraded.emit(warned)
            if self.cancel_token.cancelled:
                self.done.emit(None, "已取消")
                return
            message = resp.choices[0].message
            reply = (message.content or "").strip()
            if not reply:
                # 思考占满预算时正文为空:明确说明原因,而不是"静默什么都没学到"
                used = len(getattr(message, "reasoning_content", None) or "")
                self.done.emit(None, f"记忆提炼未产出内容(思考内容 {used} 字,"
                                     f"预算 {MEMORY_MAX_NEW_TOKENS} tokens)")
                return
            self.done.emit(parse_memories(reply), "")
        except Exception as e:
            self.done.emit(None, str(e))
