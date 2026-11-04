"""跨格式共用的小启发式。

放在这里而不是各解析器各写一份：同一条规则在 docx 里判对了、在 xlsx 里
判错了，排查成本极高，而且没人会想到去比对两个文件。
"""

from __future__ import annotations

import re

from app.parsing.models import BlockType, DocumentBlock, TableData

_NUMERIC = re.compile(r"^-?\d+(\.\d+)?%?$")

#: 公文的章级标题：「第一章　总　则」「第二节 评审程序」
CN_CHAPTER = re.compile(r"^第[一二三四五六七八九十百零]+[章节]")
#: 公文的小节标题：「一、资助标准与名额」。要求「、」后面还有内容，
#: 否则「一、」单独一行多半是列表残片。
CN_SECTION = re.compile(r"^[一二三四五六七八九十]{1,3}、\s*\S")


def cn_heading_level(text: str, max_chars: int = 40) -> int | None:
    """中文编号标题的层级；不像标题就返回 None。

    为什么把这条规则提到跨格式共用：docx 的公文和 pdf 的公文是同一批文件，
    只是导出格式不同。如果只有 pdf 认「一、」，同一份通知换个格式存进来，
    切分粒度就从「一节一块」退化成「整篇一块」—— 学生搜「资助标准」拿到的是
    一整篇 700 字通知，而搜同一份 pdf 版本时只拿到那一节。

    三条排除条件都是为了不把正文当标题：太长的不算、以句读结尾的不算
    （「一、二年级学生可在……」这种是真句子）。
    """
    cleaned = str(text or "").strip()
    if not cleaned or len(cleaned) > max_chars:
        return None
    if cleaned.endswith(("。", "；", "，", "：", ":", ";")):
        return None
    if CN_CHAPTER.match(cleaned):
        return 1
    if CN_SECTION.match(cleaned):
        return 2
    return None


def looks_numeric(value: str) -> bool:
    """是不是「纯数字/百分比」——用来判断某一行像数据还是像表头。"""
    cleaned = str(value or "").strip().replace(",", "").replace("，", "")
    if not cleaned:
        return False
    return bool(_NUMERIC.match(cleaned))


def guess_header_row_count(table: TableData, max_probe: int = 3) -> int:
    """猜表头行数。猜错不影响内容，只影响 Markdown 渲染时哪一行当表头。

    判据按可靠性排序：
        1. 首行有横向合并 → 几乎一定是分组表头（「国家奖学金」横跨名额/金额），
           而且第二行若全是一格一列，表头就是 2 行；
        2. 首行每格都非空、且都不像数字 → 当作 1 行表头；
        3. 其它情况返回 0（没有表头），宁可少标也不要错标。
    """
    if table.n_rows == 0 or table.n_cols == 0:
        return 0

    probe = min(max_probe, table.n_rows)
    first_row = [cell for cell in table.cells if cell.row == 0]
    if any(cell.colspan > 1 for cell in first_row) and probe > 1:
        second_row = [cell for cell in table.cells if cell.row == 1]
        if second_row and all(cell.colspan == 1 for cell in second_row):
            return 2
        return 1

    grid = table.grid()
    if grid and all(value.strip() for value in grid[0]) and not any(
        looks_numeric(value) for value in grid[0]
    ):
        return 1
    return 0


def apply_header_flags(table: TableData) -> int:
    """猜表头行数，并把落在表头区的单元格标上 is_header。

    统一走这个函数而不是各处自己设：不标 is_header 的话，下游要判断
    「这格是不是表头」就得再算一遍 `row < header_row_count`，
    两处规则迟早不一致。
    """
    table.header_row_count = guess_header_row_count(table)
    for cell in table.cells:
        cell.is_header = cell.row < table.header_row_count
    return table.header_row_count


_FILENAME_PREFIX = re.compile(r"^\d{1,3}_[^_]{1,10}_\d{4}-\d{2}-\d{2}_")


def title_from_filename(stem: str) -> str:
    """从文件名里剥出人看得懂的标题。

    素材库里的文件普遍叫 `38_助学金_2026-09-22_关于做好2026年国家助学金评审
    工作的通知.docx`，前面三段是采集时加的归档前缀。直接拿 stem 当标题，切出来
    每个 chunk 的前缀都会顶着「38_助学金_2026-09-22_」，既占字数又干扰语义。

    剥不掉就原样返回 —— 宁可留着前缀，也不能把标题清空。
    """
    cleaned = _FILENAME_PREFIX.sub("", str(stem or "")).strip()
    return cleaned or str(stem or "")


def normalize_heading_paths(blocks: list[DocumentBlock]) -> None:
    """按标题序列重建 heading_path，顺手把层级压到从 1 开始。

    为什么必须重建
    --------------
    各解析器原来是就地 `heading_path[: level - 1] + [text]` 维护路径，这个写法
    隐含了两个假设：层级连续从 1 开始、且路径长度等于层级深度。只要有一个不成立，
    路径就会错。

    实测的反例：`37_奖学金_...国家励志奖学金评审工作的通知.pdf` 全文没有第 1 级
    标题，「一、奖励标准与名额」被中文编号启发式判成 2 级 —— 于是
    `["一、"][:1] + ["二、"]` 得到 `["一、", "二、"]`，两个**同级**标题被拼成了
    父子关系。切分时前缀就变成「一、奖励标准与名额 > 二、申请条件」，
    模型据此认为「申请条件」是「奖励标准」的下级内容。

    改成显式的层级栈：来了一个 L 级标题，先把栈里所有 >= L 的弹出，再压入自己。
    这样「同级替换」是天然结果，不依赖路径长度和层级深度一致。

    层级平移：文档里最浅的标题就是第 1 级。一份没有章标题的通知，它的「一、二、三、」
    就是顶层，不该永远挂在虚构的第 0 级下面。
    """
    heading_levels = [
        block.heading_level
        for block in blocks
        if block.block_type is BlockType.HEADING and block.heading_level
    ]
    if not heading_levels:
        # 没有标题块（xlsx 的 sheet 名、pptx 的散装文本）时不动，
        # 它们自己填的路径是对的，重建反而会清空
        return

    # 层级可能跳级（1 → 3），也可能整份从 2 起。base 只用来把最浅的抬到 1，
    # 不改变相对关系。
    base = min(heading_levels)
    stack: list[tuple[int, str]] = []

    for block in blocks:
        if block.block_type is BlockType.HEADING and block.heading_level:
            level = max(1, block.heading_level - base + 1)
            while stack and stack[-1][0] >= level:
                stack.pop()
            text = block.text.strip()
            if text:
                stack.append((level, text))
            block.heading_level = level
        block.heading_path = [text for _, text in stack]
