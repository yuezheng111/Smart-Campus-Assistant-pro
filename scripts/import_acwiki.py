"""把开源仓库 Ac-Wiki 的公开文档清洗成本项目能入库的 Markdown。

为什么要引外部语料
------------------
`data/raw` 原本只有 22 份自建样例文档（约 40 个向量片段）。这个体量下
retrieval 评测的 hit@5 几乎没有区分度 —— 从 22 份里挑 5 份，瞎猜都能中。
要真正测出「检索器能不能在多份同类文件里排对」，语料必须先变多、变像。

来源与许可
----------
https://github.com/HUNSage/Ac-Wiki  (CC-BY-4.0)
选它是因为它的 `campus-life/` 目录讲的就是奖助学金、转专业、医保、图书馆、
学生邮箱这些校园事务，和本项目的知识域对得上，而且内容是真实中文文本。

刻意排除的
----------
`scholarship.md` 和 `faq-national-student-loan.md` 没有收录。它们写的是
全国通用政策（如「2024 年起国家奖学金由 8000 元提高到 10000 元」），
而 `data/raw/01`、`02` 是本项目的校级通知。两套金额并存会让 Agent 答出
自相矛盾的结论 —— 那是数据本身有冲突，不是检索的锅，混进来只会让指标
没法归因。要测「多来源冲突」，应该单独设计，不该顺手带进语料。

用法
----
    python scripts/import_acwiki.py --src var/_acwiki/Ac-Wiki-main --dry-run
    python scripts/import_acwiki.py --src var/_acwiki/Ac-Wiki-main
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/HUNSage/Ac-Wiki/blob/main"

# (仓库内相对路径, doc_type, 落盘标题)
# doc_type 必须落在 app/tools/document.py 声明的可选值里：
# 奖学金、助学金、考试、竞赛、学籍、教学、规章制度、活动、后勤
#
# 选篇标准是「能不能被当成一条校园信息被检索到」，不是「写得好不好」。
# 因此剔掉了三类：`social-connections`（人际关系）与 `resolving-conflicts`
# （宿舍矛盾）是价值观随笔，行文口语化，一旦被召回会把回答口吻带偏；
# `choose-major`（选专业）讲的是高考填志愿，和在校生无关，正文还只剩几个
# 链接标题，切出来的片段没有检索价值。
SOURCES: list[tuple[str, str, str]] = [
    ("docs/campus-life/medical-insurance.md", "后勤", "大学生医保参保与报销指南"),
    ("docs/campus-life/different-courses.md", "教学", "大学课程类型与学分绩点计算说明"),
    ("docs/campus-life/first-lesson-of-school.md", "教学", "新生开学第0课：入学要办的事"),
    ("docs/campus-life/student-email.md", "后勤", "学生邮箱申请与使用说明"),
    ("docs/campus-life/international-exchange.md", "教学", "国际交流项目与海外学习申请指南"),
    ("docs/campus-life/mental-health.md", "后勤", "大学生心理健康与心理咨询途径"),
    ("docs/campus-life/work-study-program.md", "助学金", "勤工俭学岗位申请与注意事项"),
    ("docs/campus-life/major-transfer-guide.md", "学籍", "转专业的基本流程与准备"),
    ("docs/academic-resources/thesis-bachelor-science.md", "教学", "本科毕业论文写作流程与规范要点"),
    ("docs/academic-resources/how-to-get-resources.md", "其他", "文献检索与学术资源获取方法"),
    ("docs/academic-resources/some-key-concepts.md", "其他", "学术研究中的关键概念解释"),
    ("docs/academic-resources/student-resources.md", "其他", "大学生常用学习资源汇总"),
    ("docs/academic-resources/math/learning-resources.md", "教学", "大学数学学习方法与资源"),
]

# 统一发布日期。Ac-Wiki 是持续维护的 wiki，单篇没有「发布日期」这个语义，
# 硬凑一个精确日期等于编数据。这里给一个区间起点，并在正文注明来源，
# 让读的人一眼看出这是参考资料而不是某天的校发通知。
PUBLISH_DATE = "2025-09-01"
DEPARTMENT = "参考资料"  # 不占任何真实部门的过滤位，避免被「学生工作处」类查询误召回
COLLEGE = "全校"

_CN_NUM = "一二三四五六七八九十"


def strip_admonitions(text: str) -> str:
    """去掉 MkDocs 的 `!!! abstract` / `??? note` 块及其缩进正文。

    这类块在本仓库里几乎全是「相关推荐」「返回顶部」之类的导航，留着会让
    每个 chunk 都带着同一串链接，检索时变成跨文档的高频噪声。
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if re.match(r"^\s*(?:!!!|\?\?\?\+?)\s*\S", lines[i]):
            i += 1
            # admonition 的正文靠缩进归属，空行不打断块
            while i < len(lines) and (
                not lines[i].strip()
                or lines[i].startswith("    ")
                or lines[i].startswith("\t")
            ):
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def strip_markup(text: str) -> str:
    """把所有纯排版的 Markdown 记号剥掉，只留能读的中文。"""
    t = text
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)              # 图片
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)          # 链接 → 纯文字
    t = re.sub(r"<[^>]{0,80}>", "", t)                       # <br> / <BR> 之类
    t = t.replace("**", "").replace("~~", "").replace("`", "")
    t = re.sub(r"(?<!\w)_([^_\n]{1,80})_(?!\w)", r"\1", t)  # 斜体下划线
    t = re.sub(r"^\s*([-*_])(?:\s*\1){2,}\s*$", "", t, flags=re.M)  # 分隔线
    t = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", t)       # emoji
    return t


def normalize(text: str) -> str:
    """压掉多空格、多空行，并修正标点后的松散空格。"""
    t = text.replace("\u00a0", " ").replace("\ufeff", "")
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"(?m)^\s+$", "", t)
    return t.strip()


def drop_leading_h1(text: str) -> str:
    """正文第一行的 H1 和 front-matter 的 title 重复，去掉。

    只去第一个 —— 后面若还有 H1（比如勤工俭学那篇分了「官方/非官方」两节），
    那是真的分节标题，要留。
    """
    lines = text.lstrip().split("\n")
    if lines and lines[0].startswith("# "):
        return "\n".join(lines[1:]).lstrip()
    return text


def clean(raw: str) -> str:
    return normalize(drop_leading_h1(normalize(strip_markup(strip_admonitions(raw)))))


def build(title: str, doc_type: str, body: str, src_path: str) -> str:
    """拼成和现有 data/raw 一致的 front-matter + 正文。"""
    url = f"{REPO_URL}/{src_path}"
    head = [
        "---",
        f"title: {title}",
        "doc_no: ACWIKI-2025",
        f"department: {DEPARTMENT}",
        f"doc_type: {doc_type}",
        f"college: {COLLEGE}",
        f"publish_date: {PUBLISH_DATE}",
        f"url: {url}",
        "---",
        "",
    ]
    tail = [
        "",
        "",
        f"资料整理自开源项目 Ac-Wiki（{url}），"
        "内容为全国通用的经验与政策说明，具体执行以本校现行文件为准。",
        "",
    ]
    return "\n".join(head) + body + "\n".join(tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="var/_acwiki/Ac-Wiki-main", help="解压后的仓库根目录")
    ap.add_argument("--out", default="data/raw", help="输出目录")
    ap.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    args = ap.parse_args()

    src_root = ROOT / args.src
    out_dir = ROOT / args.out
    if not src_root.is_dir():
        print(f"找不到源目录：{src_root}", file=sys.stderr)
        return 1

    # 编号分配：先认领库里已经存在同名文件，再给新文件往后排。
    # 不能简单地「取已有最大编号 +1」—— 那样重跑一次编号就整体漂移一轮，
    # 同一篇文档会以两个 file_path 入库，向量库里出现重复片段。
    existing = sorted(out_dir.glob("*.md"))
    taken: dict[str, int] = {}          # 标题 → 已占用编号
    max_no = 0
    for p in existing:
        m = re.match(r"^(\d+)_", p.name)
        if not m:
            continue
        max_no = max(max_no, int(m.group(1)))
        # 文件名形如 NN_类型_日期_标题.md，标题可能自己带下划线，只切前三段
        parts = p.stem.split("_", 3)
        if len(parts) == 4:
            taken[parts[3]] = int(parts[0])

    rows: list[tuple[str, int, int]] = []
    for rel, doc_type, title in SOURCES:
        path = src_root / rel
        if not path.is_file():
            print(f"跳过（源文件缺失）：{rel}", file=sys.stderr)
            continue
        raw = path.read_text(encoding="utf-8")
        body = clean(raw)
        if len(body) < 200:
            print(f"跳过（清洗后仅 {len(body)} 字，信息量不足）：{rel}", file=sys.stderr)
            continue

        if title in taken:
            no = taken[title]
        else:
            max_no += 1
            no = max_no
            taken[title] = no
        fname = f"{no:02d}_{doc_type}_{PUBLISH_DATE}_{title}.md"
        text = build(title, doc_type, body, rel)
        rows.append((fname, len(body), len(text)))

        if not args.dry_run:
            (out_dir / fname).write_text(text, encoding="utf-8")

    print(f"{'[dry-run] ' if args.dry_run else ''}处理 {len(rows)} 篇，输出到 {out_dir}")
    print(f"{'序号':<6}{'正文字数':>8}{'落盘字节':>10}  文件名")
    for fname, body_len, total in rows:
        print(f"{fname[:2]:<6}{body_len:>8}{total:>10}  {fname}")
    total_body = sum(r[1] for r in rows)
    print(f"\n合计正文字数 {total_body}，平均 {total_body // max(len(rows), 1)} 字/篇")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
