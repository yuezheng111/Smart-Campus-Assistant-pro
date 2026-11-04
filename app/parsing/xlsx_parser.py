"""XLSX 解析器。

Excel 不能变成「没有上下文的字符串」
------------------------------------
一张工作表里往往并排放着好几张结构不同的表，中间用空行隔开。所以这里的
单位是**数据区**（连续非空行的一段），不是整张 sheet：

    行 1-5   2026 年国家奖学金名额分配表（表头 2 行）
    行 6     空行
    行 7-12  各学院咨询电话（另一套表头）

整张 sheet 拍成一个字符串，这两张表就会混成一个"表格"，表头对不上数据。

保留的东西
----------
- merged_cells → rowspan/colspan（不拍平）
- 公式：text 用缓存值（RAG 要读的是结果），extra["formulas"] 里留公式原文
- 隐藏行列：记在 extra 里，**不删数据** —— 隐藏不等于无效，可能是分组折叠
- cell_range：能定位回"A1:D9"
- 日期：转 ISO 字符串，避免 "2026-09-18 00:00:00" 这种带着零点的怪格式

两次加载
--------
openpyxl 的 data_only 是加载期开关，一次只能拿一种：拿值就拿不到公式。
所以加载两遍。大文件会多花内存，但比"公式和值只能二选一"划算。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.utils import column_index_from_string, get_column_letter

from app.logging_config import get_logger
from app.parsing._heuristics import apply_header_flags
from app.parsing.base import BaseParser, ParseOptions
from app.parsing.models import (
    BlockType,
    DocumentBlock,
    DocumentMetadata,
    FileType,
    ParsedDocument,
    ParseError,
    ParseStatus,
    SourceLocation,
    TableCell,
    TableData,
    new_block_id,
)

log = get_logger(__name__)

MAX_ROWS = 20_000
MAX_COLS = 200


def _cell_to_text(value: Any) -> str:
    """单元格值 → 文本。日期转 ISO，浮点去掉无意义的 .0。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, dt.datetime):
        return value.date().isoformat() if value.time() == dt.time(0, 0) else value.isoformat(sep=" ")
    if isinstance(value, (dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _row_is_empty(values: list[str]) -> bool:
    return all(not value for value in values)


class XLSXParser(BaseParser):
    name = "xlsx"
    version = "0.1.0"
    file_types = (FileType.XLSX,)

    def _parse(
        self, path: Path, options: ParseOptions, document_id: str
    ) -> ParsedDocument:
        workbook = load_workbook(str(path), data_only=True)
        formulas = load_workbook(str(path), data_only=False)

        blocks: list[DocumentBlock] = []
        errors: list[ParseError] = []

        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            sheet_formula = formulas[sheet_name]

            max_row = min(int(sheet.max_row or 0), MAX_ROWS)
            max_col = min(int(sheet.max_column or 0), MAX_COLS)
            if max_row == 0 or max_col == 0:
                continue
            if int(sheet.max_row or 0) > MAX_ROWS or int(sheet.max_column or 0) > MAX_COLS:
                errors.append(
                    ParseError(
                        stage="sheet",
                        message=f"工作表 {sheet_name} 超出处理上限"
                                f"（{sheet.max_row} 行 × {sheet.max_column} 列），已截断",
                    )
                )

            hidden_rows = self._hidden_rows(sheet, max_row)
            hidden_cols = self._hidden_cols(sheet, max_col)

            # 逐行读取一次，避免后面反复 sheet.cell() 触发重复解析
            grid: dict[int, list[str]] = {}
            for r in range(1, max_row + 1):
                grid[r] = [
                    _cell_to_text(sheet.cell(row=r, column=c).value)
                    for c in range(1, max_col + 1)
                ]

            regions = self._split_regions(grid, max_row)
            if not regions:
                continue

            # 工作表名本身是一层上下文：脱离 sheet 名，"名额 10" 没有意义
            blocks.append(
                DocumentBlock(
                    block_id=new_block_id(document_id, len(blocks)),
                    block_type=BlockType.HEADING,
                    text=sheet_name,
                    heading_level=2,
                    heading_path=[sheet_name],
                    source_location=SourceLocation(sheet_name=sheet_name),
                    extra={"sheet_index": workbook.sheetnames.index(sheet_name)},
                )
            )

            for start_row, end_row in regions:
                try:
                    table, cell_range, formulas_in_region = self._build_table(
                        sheet, sheet_formula, grid,
                        start_row, end_row, max_col,
                    )
                except Exception as exc:
                    log.warning("工作表数据区解析失败 | %s | %d-%d | %s",
                                sheet_name, start_row, end_row, exc)
                    errors.append(
                        ParseError(
                            stage="table",
                            message=f"工作表 {sheet_name} 第 {start_row}-{end_row} 行解析失败："
                                    f"{type(exc).__name__}: {exc}",
                            location=SourceLocation(
                                sheet_name=sheet_name,
                                cell_range=f"A{start_row}:{get_column_letter(max_col)}{end_row}",
                            ),
                        )
                    )
                    continue
                if table.n_rows == 0 or table.n_cols == 0:
                    continue
                extra: dict[str, Any] = {}
                if formulas_in_region:
                    extra["formulas"] = formulas_in_region
                if hidden_rows or hidden_cols:
                    extra["hidden_rows"] = hidden_rows
                    extra["hidden_cols"] = hidden_cols
                blocks.append(
                    DocumentBlock(
                        block_id=new_block_id(document_id, len(blocks)),
                        block_type=BlockType.TABLE,
                        table_data=table,
                        heading_path=[sheet_name],
                        source_location=SourceLocation(
                            sheet_name=sheet_name,
                            cell_range=cell_range,
                        ),
                        extra=extra,
                    )
                )

        metadata = DocumentMetadata(sheet_count=len(workbook.sheetnames))

        if not blocks:
            return ParsedDocument(
                document_id=document_id,
                file_name=path.name,
                file_type=FileType.XLSX,
                title=path.stem,
                metadata=metadata,
                blocks=[],
                parser_name=self.name,
                parser_version=self.version,
                parse_status=ParseStatus.EMPTY,
                errors=errors
                + [ParseError(stage="extract", message="工作簿里没有提取到任何数据")],
            )

        return ParsedDocument(
            document_id=document_id,
            file_name=path.name,
            file_type=FileType.XLSX,
            title=path.stem,
            metadata=metadata,
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            parse_status=ParseStatus.PARTIAL if errors else ParseStatus.COMPLETED,
            errors=errors,
        )

    # ---- 行/列 ----

    @staticmethod
    def _split_regions(grid: dict[int, list[str]], max_row: int) -> list[tuple[int, int]]:
        """按连续非空行切出数据区。

        空行是 Excel 里最自然的分区信号：同一张 sheet 里并排放两套表时，
        中间几乎总会空一行。把整张 sheet 当成一张表，表头就会张冠李戴。
        """
        regions: list[tuple[int, int]] = []
        start: int | None = None
        for r in range(1, max_row + 1):
            empty = _row_is_empty(grid.get(r, []))
            if not empty and start is None:
                start = r
            elif empty and start is not None:
                regions.append((start, r - 1))
                start = None
        if start is not None:
            regions.append((start, max_row))
        return regions

    @staticmethod
    def _hidden_rows(sheet: Any, max_row: int) -> list[int]:
        """列出隐藏行。

        遍历 row_dimensions 而不是 1..max_row：max_row 只算**有值的**最后一行，
        末尾被隐藏的空行（或者被折叠的分组）不在这里面，按 max_row 扫会漏掉。
        """
        hidden = [
            int(index)
            for index, dimension in sheet.row_dimensions.items()
            if dimension is not None and dimension.hidden
        ]
        # 不按 max_row 截断：被折叠的分组可能整段都在数据区之外，
        # 但"这一页有隐藏行"本身是文件事实，应该如实记下来。
        return sorted(row for row in hidden if row >= 1)

    @staticmethod
    def _hidden_cols(sheet: Any, max_col: int) -> list[str]:
        hidden: list[str] = []
        for letter, dimension in sheet.column_dimensions.items():
            if dimension is not None and dimension.hidden:
                try:
                    index = column_index_from_string(letter)
                except Exception:
                    continue
                if 1 <= index <= max_col:
                    hidden.append(letter)
        return sorted(hidden, key=column_index_from_string)

    # ---- 单张表 ----

    def _build_table(
        self,
        sheet: Any,
        sheet_formula: Any,
        grid: dict[int, list[str]],
        start_row: int,
        end_row: int,
        max_col: int,
    ) -> tuple[TableData, str, dict[str, str]]:
        """把一个数据区构造成 (TableData, cell_range, 公式表)。

        region 的列范围按**区内实际有内容的列**收紧，而不是取整张 sheet 的
        max_column —— 否则右边几百个空列会把表拉成几百列。

        返回值里带上 cell_range 和公式，而不是塞进 TableData：
        TableData 是内容模型，位置和公式属于"这一块在文件哪里"，
        混在一起会让阶段五不知道该把哪些字段写到 Chroma。
        """
        rows = list(range(start_row, end_row + 1))

        used_cols: set[int] = set()
        for r in rows:
            for idx, value in enumerate(grid.get(r, []), start=1):
                if value:
                    used_cols.add(idx)
                    continue
                # 空格子也要看一眼公式：只在公式里出现过的列（典型是最后一列
                # 「合计」）如果被当成空列裁掉，那一列的计算逻辑就整列消失了。
                # openpyxl 保存的文件没有缓存值，data_only=True 读出来就是 None。
                raw = sheet_formula.cell(row=r, column=idx).value
                if isinstance(raw, str) and raw.startswith("="):
                    used_cols.add(idx)
        if not used_cols:
            return TableData(n_rows=0, n_cols=0), "", {}

        min_col = min(used_cols)
        max_used_col = max(used_cols)
        n_rows = len(rows)
        n_cols = max_used_col - min_col + 1

        # 合并单元格：只处理「起点落在本区内」的合并，区外的起点直接忽略
        origin_span: dict[tuple[int, int], tuple[int, int]] = {}
        covered: set[tuple[int, int]] = set()
        for merged in list(sheet.merged_cells.ranges):
            r1, r2 = merged.min_row, merged.max_row
            c1, c2 = merged.min_col, merged.max_col
            if r1 < start_row or r1 > end_row:
                continue
            if c1 < min_col or c1 > max_used_col:
                continue
            r2 = min(r2, end_row)
            c2 = min(c2, max_used_col)
            origin_span[(r1, c1)] = (r2 - r1 + 1, c2 - c1 + 1)
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    if (r, c) != (r1, c1):
                        covered.add((r, c))

        cells: list[TableCell] = []
        formula_map: dict[str, str] = {}

        for r in rows:
            row_values = grid.get(r, [])
            for c in range(min_col, max_used_col + 1):
                if (r, c) in covered:
                    continue
                text = row_values[c - 1] if c - 1 < len(row_values) else ""
                rowspan, colspan = origin_span.get((r, c), (1, 1))

                raw_formula = sheet_formula.cell(row=r, column=c).value
                if isinstance(raw_formula, str) and raw_formula.startswith("="):
                    address = f"{get_column_letter(c)}{r}"
                    formula_map[address] = raw_formula
                    if not text:
                        # 缓存值缺失（openpyxl 生成的文件通常没有）时，至少别丢公式
                        text = raw_formula

                cells.append(
                    TableCell(
                        text=text,
                        row=r - start_row,
                        col=c - min_col,
                        rowspan=max(1, rowspan),
                        colspan=max(1, colspan),
                    )
                )

        table = TableData(
            n_rows=n_rows,
            n_cols=n_cols,
            cells=cells,
            header_row_count=0,
        )
        table.header_row_count = apply_header_flags(table)

        cell_range = (
            f"{get_column_letter(min_col)}{start_row}:"
            f"{get_column_letter(max_used_col)}{end_row}"
        )
        return table, cell_range, formula_map
