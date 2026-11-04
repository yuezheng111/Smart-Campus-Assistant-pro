"""校验与终止判定：判断「这次回答能不能交付」。

为什么用代码而不是让 LLM 当裁判：
    字符串/规则判定是可复现的 —— 同样的输入永远同样的结论，回归测试才有意义。
    LLM 裁判每次可能不同，出了问题也没法解释「它凭什么说不行」。
    这和 evals/README.md 里「事实匹配不用 LLM 判分」是同一个理由。

它不生成任何内容，只回答三个问题：
    这条回答有依据吗？（证据为空却在陈述事实，就是幻觉）
    引用编号指得到东西吗？（[资料7] 但只有 3 段资料）
    数字能在资料里找到吗？（金额、名额、日期错一位就是事故）

只做「标注」不做「重写」：
    发现问题时给出 caveat，由输出层和人决定怎么处理。
    自动重生成会引入预算控制和死循环风险，那是下一步的事 —— 先把「能发现」做出来。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.agent.evidence import evidence_text, source_count

# 引用编号：[资料1] / [资料 1]
_CITATION = re.compile(r"\[资料\s*(\d+)\]")

# 只查「带量词的数量」。为什么不查所有数字：
#   日期会被拆成 2025 / 9 / 12 三个碎片，逐个比对必然误报（实测里
#   「2026 年 9 月 23 日」就报了 2026 和 23 两个假警）。
#   而 verifier 真正要抓的是金额、名额、比例这类「错一位就是事故」的事实。
_QUANTIFIED = re.compile(
    r"(\d{2,}(?:\.\d+)?)\s*"
    r"(元|万元|名|人|个|天|日|分|学分|%|％|次|项|门|周|课时|小时)"
)
# 完整日期整体剔除，避免「9 月 12 日」里的 12 被当成数量
_DATE = re.compile(
    r"\d{4}\s*[-/年.]\s*\d{1,2}\s*[-/月.]\s*\d{1,2}\s*日?"
    r"|\d{4}\s*年\s*\d{1,2}\s*月"
)

# 能证明事实的证据类型。`tool_trace`（工具调用日志）刻意不在其中 ——
# 它只能证明「调用发生过」，不能证明「有这件事」。旧版把日志当正文，
# 于是「查了 0 条」和「查到 3 条」在校验层看起来没有区别。
_FACT_KINDS = ("document", "structured_record")

# 需要真的去查证据的路由。general 是闲聊，本来就没有证据。
_NEEDS_EVIDENCE = ("document", "structured", "hybrid", "planning")

MAX_CAVEATS = 3


@dataclass
class Verdict:
    status: str = "pass"          # pass | warn | degrade
    caveats: list[str] = field(default_factory=list)   # 给用户看的短句
    detail: list[str] = field(default_factory=list)    # 给 trace / 日志看的细节

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "caveats": self.caveats, "detail": self.detail}


def _normalize(text: str) -> str:
    """压掉空白和千分位逗号，让「8,000 元」和「8000元」能对上。"""
    return re.sub(r"[,\s，]", "", str(text or ""))


def _has_number(haystack: str, needle: str) -> bool:
    """数字命中必须带边界，否则 45 会被 4500 骗过去。"""
    return re.search(rf"(?<![\d.]){re.escape(needle)}(?![\d.])", haystack) is not None


def check_citations(answer: str, total_docs: int) -> list[str]:
    """引用编号是否指得到东西。返回问题描述。"""
    cited = {int(n) for n in _CITATION.findall(answer or "")}
    if not cited:
        return []
    bad = sorted(n for n in cited if n < 1 or n > total_docs)
    if not bad:
        return []
    if total_docs == 0:
        return [f"回答引用了资料 {bad[0]}，但本轮没有检索到任何文件"]
    return [f"回答引用了不存在的编号 {bad}（本轮只有 {total_docs} 份资料）"]


def check_numbers(answer: str, evidence: str, limit: int = MAX_CAVEATS) -> list[str]:
    """回答里的数量能不能在证据里找到出处。

    这是幻觉最廉价、也最有效的探针：金额、名额、截止日期错一位，
    关键词匹配完全看不出来，但数字对不上会立刻暴露。

    两个刻意的收敛，都是被真实误报逼出来的：
      1. 先剔除完整日期，再抽数量 —— 否则「2026 年 9 月 23 日」会报 2026 和 23；
      2. 只认带量词的数字 —— 「前 23 名」要查，「第 3 条」不用查（那是序号）。
    宁可漏报也不要误报：一条假警会让用户从此不再看这些提示。
    """
    if not evidence.strip() or not answer.strip():
        return []

    haystack = _normalize(evidence)
    # 回答先归一化（千分位逗号会让 8,000 被切成 000），再剔除日期，最后抽数量
    cleaned = _DATE.sub(" ", _normalize(answer))

    missing: list[str] = []
    for raw in dict.fromkeys(m.group(1) for m in _QUANTIFIED.finditer(cleaned)):
        if not _has_number(haystack, raw):
            missing.append(raw)
    if not missing:
        return []
    shown = missing[:limit]
    more = "等" if len(missing) > limit else ""
    return [f"以下数字未在资料中找到出处：{'、'.join(shown)}{more}"]


def _has_fact_evidence(evidence: list[dict[str, Any]] | None) -> bool:
    """证据里有没有「真的取到了东西」的那一条。

    只要**任意一条**成立就算有依据 —— 不能因为某条线查空，
    就把整体判成「知识库里没有这件事」（局部失败 ≠ 全局不存在）。
    """
    for item in evidence or []:
        kind = item.get("kind")
        if kind not in _FACT_KINDS:
            continue
        if kind == "document":
            if (item.get("text") or "").strip():
                return True
        elif item.get("has_results"):
            return True
    return False


def check_grounding(answer: str, evidence: list[dict[str, Any]], route: str) -> list[str]:
    """有没有「无证据却在下结论」。

    三个刻意的收紧，都是被体测那次静默失败逼出来的：

      1. 不再看 text 是否为空，而是看证据的**类型和实际内容** ——
         工具调用日志也占着 text，旧写法会把它当成事实依据；
      2. 不再因为答案里有「未检索到」就放行。拒答是诚实，但拒答的原因
         恰恰是「没查到」，这本身就是缺依据，应该让用户看得见；
      3. 只要任意一条证据真的取到内容就算有依据，不把局部失败
         扩大成「整个知识库不存在相关信息」。
    """
    if route not in _NEEDS_EVIDENCE:
        return []
    if _has_fact_evidence(evidence):
        return []
    return ["本轮没有任何检索或查询结果，回答缺少资料依据"]


def verify_answer(
    answer: str,
    evidence: list[dict[str, Any]] | None,
    route: str = "",
) -> Verdict:
    """跑完全部检查，汇总成一个结论。"""
    evidence = evidence or []
    verdict = Verdict()

    grounding = check_grounding(answer, evidence, route)
    citation = check_citations(answer, source_count(evidence))
    numbers = check_numbers(answer, evidence_text(evidence))

    # 无依据是硬问题，直接降级；引用和数字是软问题，只提醒。
    if grounding:
        verdict.status = "degrade"
        verdict.caveats.append("这条回答没有检索到学校资料支撑，请以官方通知为准")
        verdict.detail.extend(grounding)
    elif citation or numbers:
        verdict.status = "warn"

    if citation:
        verdict.caveats.append("部分引用编号对应不到文件，建议核对原文")
        verdict.detail.extend(citation)
    if numbers:
        verdict.caveats.append("部分数据未在资料中确认，请以原文为准")
        verdict.detail.extend(numbers)

    return verdict
