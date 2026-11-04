# 校园智助 · Campus Agent

> 一个辅助学生自助查询学校相关通知及重要考试比赛的 Agent

用自然语言检索学校文件、查询竞赛与考试信息、生成学业规划建议的校园 Agent 系统。

**RAG（非结构化文件）+ MySQL（结构化数据）+ Tools + LangGraph Agent**。

---

## 一、整体架构

```
                          学生提问
                             │
                        ┌────▼─────┐
                        │ classify │  意图识别（LLM 结构化输出）
                        └────┬─────┘
          ┌──────────┬───────┴────────┬──────────────┐
          ▼          ▼                ▼              ▼
      document   structured        hybrid       planning      general
          │          │                │              │            │
          ▼          │                ▼              │            │
   retrieve_docs     │         retrieve_docs         │            │
          │          │                │              │            │
          │          ▼                ▼              ▼            │
          │     query_data ──────────┘         query_data        │
          │                                         │             │
          │                                         ▼             │
          │                                  retrieve_docs        │
          ▼                                         │             ▼
          └──────────────► compose ◄────────────────┘─────────────┘
                            │
                         最终回答（带 [资料N] 引用）
```

两条知识线在物理上完全分离：

| | 第一条线：学校文件 | 第二条线：竞赛 / 考试 |
|---|---|---|
| 数据形态 | 非结构化（通知、规章制度） | 结构化（时间、类别、年级） |
| 存储 | Chroma 向量库 + MySQL 元数据表 | MySQL 关系表 |
| 检索 | Embedding 语义检索 + 元数据硬过滤 | 参数化 SQL |
| 解决的问题 | 「某份文件写了什么」 | 「什么时候、有哪些、多少」 |

---

## 二、目录结构

```
Agent学习6/
├── .env                      # 本地配置（已 gitignore）
├── .env.example              # 配置模板
├── requirements.txt
├── run_ingest.py             # 学校文件灌库入口（第一条线）
│
├── app/
│   ├── config.py             # 全局配置 + Windows 注册表兜底读 API Key
│   ├── db.py                 # MySQL 访问层（参数化查询构造器）
│   ├── ingest.py             # 入库总入口：按 PARSING_PIPELINE 分派 v2 / legacy
│   ├── logging_config.py     # 文件日志（按天轮转）
│   ├── admin_change.py       # 管理端元数据变更（改档案不改正文）
│   ├── chat_attachment.py    # 对话中的附件临时解析
│   │
│   ├── parsing/              # 多格式解析链路（v2，默认路径）
│   │   ├── models.py         #   Canonical Document：表格保留合并单元格
│   │   ├── base.py           #   BaseParser 基类 + 侧车元数据合并
│   │   ├── factory.py        #   按 magic bytes 分发，不信后缀
│   │   ├── pdf_parser.py     #   pdfplumber 取结构 + pypdf 取图片字节
│   │   ├── docx_parser.py    #   标题层级 / 表格 / 图片
│   │   ├── pptx_parser.py    #   按形状几何推断标题，备注单独成块
│   │   ├── xlsx_parser.py    #   多 sheet，表头 + 合并区域
│   │   ├── text_parser.py    #   md / txt，认中文编号标题
│   │   ├── normalize.py      #   清洗 + 质量校验（四态状态判定）
│   │   ├── chunker.py        #   结构感知切分
│   │   ├── index.py          #   Chroma 写入（整批先删后写）
│   │   └── pipeline.py       #   入库状态机 + 重试 + 并发边界
│   │
│   ├── rag/                  # 检索层（两条链路共用）+ 老链路（legacy 退路）
│   │   ├── loader.py         #   老链路：文档加载 + front-matter 抽取
│   │   ├── splitter.py       #   老链路：按字符切分
│   │   ├── vectorstore.py    #   Chroma + DashScope Embedding
│   │   └── retriever.py      #   语义检索 + 元数据过滤
│   │
│   ├── tools/                # Tools 层（Agent 的手）
│   │   ├── common.py         #   年级翻译、报名窗口判断、同义词映射
│   │   ├── document.py       #   search_school_document / list_school_documents
│   │   │                     #   get_document_detail
│   │   ├── admin.py          #   管理端：列文件 / 改元数据 / 对账 / 废止
│   │   ├── competition.py    #   query_competition / recommend_competition
│   │   ├── exam.py           #   query_exam
│   │   └── student.py        #   get_student_profile
│   │
│   ├── agent/                # LangGraph Agent
│   │   ├── llm.py            #   LLM 客户端工厂
│   │   ├── router.py         #   意图识别 + 参数抽取（结构化输出）
│   │   ├── state.py          #   状态定义（reducer vs 覆盖）
│   │   ├── graph.py          #   学生端状态机组装
│   │   ├── admin_graph.py    #   管理端状态机（与学生端共用 RAG、逻辑隔离）
│   │   ├── evidence.py       #   检索证据链，供引用编号校验
│   │   └── verifier.py       #   回答校验（引用是否越界、数字是否有据）
│   │
│   ├── api/
│   │   ├── main.py           # FastAPI 路由
│   │   └── auth.py           # 管理端 token 校验
│   └── web/                  # 前端（无框架无构建）
│       ├── index.html        #   学生端
│       ├── admin.html        #   管理端（文件管理 + 原件对账）
│       ├── admin_chat.html   #   管理端对话
│       ├── styles/{tokens,base,components}.css
│       └── scripts/main.js
│
├── db/schema.sql             # 4 张表：document / competition / exam / student
├── data/raw/                 # 学校文件（115 份：pdf / docx / md / txt）
├── docs/                     # 系统设计、Agent 重构记录
├── evals/                    # 问题集 + 评测脚本 + 历史评测结果
├── tests/                    # 156 个单元测试
├── scripts/
│   ├── init_db.py            # 建库建表 + 结构化种子数据
│   ├── gen_sample_docs.py    # 生成仿真学校文件
│   ├── seed_data.py          # 竞赛/考试/学生种子数据
│   ├── test_tools.py         # 工具层冒烟测试（不经过 LLM）
│   ├── bench_retrieval.py    # 检索延迟测量（warmup + p50/p95）
│   └── bench_ingest_v2.py    # 入库耗时与幂等验收
└── var/chroma/               # 向量库持久化目录
```

---

## 三、快速开始

### 0. 前置条件

- Python 3.12+（本项目在 3.12.6 上验证过），建议放进虚拟环境
- MySQL 5.7+（库名 `school_agent`）：连接信息写进 `.env`，不要提交
- API Key（同样写进 `.env`，字段名见 `.env.example`）：
  - `DEEPSEEK_API_KEY` — 对话模型
  - `DASHSCOPE_API_KEY` — 向量模型（DeepSeek 没有 embedding 接口）

> 首次使用请先补依赖：`pip install -r requirements.txt`

### 1. 初始化数据库（结构化数据）

```bash
python scripts/gen_sample_docs.py        # 生成 21 份基础 md 文件
python scripts/gen_corpus_multiformat.py # 生成 94 份多格式语料（md/txt/pdf/docx + 侧车 yaml）
python scripts/init_db.py                # 建表 + 灌竞赛/考试
```

> 语料是可选的：`data/raw/` 里已经带了一份 115 份的仿真语料，直接用就行。
> 想换成自己的文件，把原件丢进 `data/raw/` 再跑灌库即可。

> ⚠️ **顺序很重要**：`init_db.py` 里的 `schema.sql` 会 `DROP TABLE document`，
> 所以它必须在 `run_ingest.py` **之前**跑。反过来跑会把文件元数据冲掉。

### 2. 学校文件灌库（第一条线）

```bash
python run_ingest.py          # 默认走 v2 多格式解析链路
python run_ingest.py --purge  # 连留档一起清空后重建（换向量库时用）
```

这一步会：解析文档 → 清洗校验 → 结构感知切分（115 份 → 828 个 chunk）
→ 写 MySQL 元数据表 → 写 Chroma 向量库。需要调用 DashScope Embedding 接口。

### 3. 自检（可选但推荐）

```bash
# 不消耗 LLM，直接验证 7 个工具
python scripts/test_tools.py

# 只验证检索质量
python -m app.rag.retriever

# 只验证路由分类
python -m app.agent.router

# 端到端跑 5 条不同类型的提问
python -m app.agent.graph
```

### 4. 启动服务

```bash
python -m app.api.main
```

打开 <http://127.0.0.1:8000>。管理端在 `/admin`（默认只允许本机回环访问；
需要远程访问就在 `.env` 里配 `ADMIN_TOKEN`，然后用 `/admin?token=...` 打开）。

### 5. 测试与验收

```bash
python -m pytest tests -q                          # 156 例单元测试，不调外部接口
python scripts/bench_ingest_v2.py --step rebuild   # 全量重建 + 计时
python scripts/bench_ingest_v2.py --step verify    # 换进程校验 828 条片段可读
python scripts/bench_ingest_v2.py --step incremental  # 幂等复跑，应 0 次 embedding
python scripts/bench_retrieval.py                  # 检索延迟 p50 / p95
python evals/run_eval.py --layer all               # 50 条问题的三层评测
```

> 跑完 `run_ingest.py` 或 `bench_ingest_v2.py` 之后，**web 服务必须重启**才能搜到新文件
> —— 原因见「已知限制」和「设计决策 10」。

---

## 四、关键设计决策

### 1. 为什么 Router 用 `with_structured_output` 而不是让它输出 JSON 文本

自由文本输出要自己 `json.loads`，模型偶尔会加解释、加 ```json 代码块，
线上就变成「偶发解析失败」。结构化输出把约束下沉到 API 层（本质是 function calling），
稳定性完全不同。看 `app/agent/router.py`。

### 2. 为什么检索要「元数据过滤 + 向量语义排序」两段式

学生的问题自带约束：「2024 年 **计算机学院** 的 **奖学金** 通知」。
纯向量检索会被「2025 年」「其他学院」的相似文档污染。所以先用元数据硬过滤缩小范围，
再在候选里做语义排序。看 `app/rag/retriever.py::build_filter`。

`college` 字段有个细节：写「全校」的文件对所有学院都适用，所以用
`{"college": {"$in": [目标学院, "全校"]}}` 而不是精确等于。

### 3. 为什么 chunk 要补上标题

切片后拿到的是一段光秃秃的正文，向量模型不知道它属于哪份文件。
切分时统一加上 `【文件标题】` 前缀（见 `app/rag/splitter.py`），召回准确率明显提升。

### 4. 为什么「阶段」要在 Tool 里算，而不是让模型推

「这个比赛现在还能不能报名」需要拿今天的日期去比对报名窗口。
让模型自己推，跨年的场次（11 月报名、次年 4 月比赛）十次里有三次会推错。
在 `app/tools/common.py::window_status` 里算准，直接返回
`正在报名 / 未开始报名 / 报名已截止，待举行 / 已结束`，模型只负责转述。

### 5. 为什么工具要分两组下发

`DOC_TOOLS` 和 `DATA_TOOLS` 分开（`app/tools/__init__.py`）。
「只查文件」的分支里只挂 `DOC_TOOLS`，模型**根本没有机会**误调竞赛库。
用工具可见性约束行为，比在 prompt 里写「请你不要调用 xxx」可靠得多。

### 6. 为什么要显式做同义词映射

学生会说「四六级」，数据库里存的是「全国大学英语四级考试（CET-4）」，
`LIKE '%四六级%'` 一条都查不到。这是真实项目里最高频的坑之一，
所以在 `app/tools/exam.py::KEYWORD_ALIASES` 里做了显式映射，
并在 `db.build_where` 里加了 `or_like` 操作符来支持 OR 展开。

### 7. 为什么种子数据的年份要跟着「今天」走

写死 `[2021..2025]` 的后果是：过两年跑起来全是「已结束」，演示直接垮掉。
所以 `seed_data.py` 用 `range(今天年份-5, 今天年份+1)` 生成，永远有「正在报名」的数据。

### 8. 为什么解析结果要先变成结构化文档，而不是直接切字符串

老链路（`app/rag/loader.py`）把文件读成一个 `str`，这一步是不可逆的降维：
读成字符串的那一刻，表格的行列、标题层级、图片位置全丢了，后面想「按章节切」
「表格整体成块」根本没有数据可用 —— 只能在字符串上猜。

所以 v2 链路的解析器统一吐 `ParsedDocument`（`app/parsing/models.py`）：
一个 `blocks` 列表，每个 block 带 `type`（heading / paragraph / table / image）、
`text`、`heading_level`。切分器（`chunker.py`）消费的是结构，不是字符串。
**模型必须先于解析器存在，不能等解析完再补。**

### 9. 为什么不把合并单元格拍平

「拍平」指把合并区域里每个单元格都填上值。后果是一张跨 5 行的「学院」列
会重复 5 次同样的文字，切块后模型看到 5 行重复，而且**再也分不清哪一格本来是合并的**。
学校的名额分配表几乎全是这种结构。

所以单元格存 `(row, col, rowspan, colspan)` 四元组，渲染时才用 `grid()` 展开成矩阵。
代价是每个读表格的地方都得先调 `grid()`，收益是原始版式一点没丢。

### 10. 为什么批量写 Chroma 要「先整批删，再逐份写」

这是本项目踩得最深的一个坑。原来的写法是逐份文件「删掉旧片段 → 写入新片段」，
115 份跑完，**HNSW 索引直接坏掉**：新进程打开集合，连 `count()` / `get()` 都报
`Error loading hnsw index`，整个知识库对学生完全不可检索 —— 而 SQLite 里的数据一条没丢，
所以从数据库看一切正常，极具迷惑性。

写了个最小复现（`var/_exp_chroma_churn2.py`，805 条 / 1024 维）做了四组对照：

| 写入模式 | 结果 |
|---|---|
| 逐份「删一次 → 写一批」交替 | ✗ 坏 |
| 先逐份删 115 次，再统一写（分相） | ✗ 坏 |
| 一次删空，再统一写 | ✓ 正常 |
| 删集合重写 | ✓ 正常 |

结论：变量是**删除调用的次数**，跟是否交替、跟有没有 `close()` 都无关
（关之前 `sleep(10)` 照样坏）。20 条的小集合复现不出来，量级到数百条才触发。

所以现在 `IngestPipeline.run()` 是三段式：全部文件解析+向量化完 → **一次** `$in` 删完整批
→ 逐份 `upsert`（不是 `add`）。单份上传只有一份文件，删两次无所谓，不受这条约束。

---

## 五、API 一览

| 方法 | 路径 | 说明 | 消耗 LLM |
|---|---|---|---|
| POST | `/api/chat` | **Agent 主入口**，走完整 LangGraph | ✅ |
| GET | `/api/search` | 纯 RAG 语义检索 | ❌ |
| GET | `/api/documents` | 按条件列文件元数据 | ❌ |
| GET | `/api/documents/{id}` | 文件详情（元数据） | ❌ |
| GET | `/api/documents/{id}/raw` | **文件原文正文**（剥掉 YAML 头，供前端阅读器渲染） | ❌ |
| GET | `/api/documents/{id}/download` | **下载原件**（保留原始扩展名） | ❌ |
| GET | `/api/competitions` | 竞赛查询 | ❌ |
| GET | `/api/exams` | 考试查询 | ❌ |
| POST | `/api/recommend` | 竞赛推荐（规则打分） | ❌ |
| GET | `/api/students/{no}` | 学生画像 | ❌ |
| GET | `/api/stats` | 统计 | ❌ |
| GET | `/health` | 健康检查 | ❌ |

管理端（`/api/admin/*`，**默认只允许本机回环访问**；配了 `ADMIN_TOKEN` 后需带
`X-Admin-Token` 请求头，也支持 `?token=`）：

| 方法 | 路径 | 说明 | 消耗 LLM |
|---|---|---|---|
| POST | `/api/admin/upload` | 上传文件入库（支持 pdf/docx/pptx/xlsx/md/txt） | ❌ |
| GET | `/api/admin/documents` | 文件管理列表（可按状态 / 原件缺失筛选） | ❌ |
| POST | `/api/admin/documents/{id}/supersede` | 废止（只改状态，原文与片段保留） | ❌ |
| POST | `/api/admin/documents/{id}/restore` | 撤销废止 | ❌ |
| GET | `/api/admin/orphans` | 原件对账：扫出「库里有、磁盘上没有」的留档 | ❌ |
| POST | `/api/admin/orphans/mark` | 标记缺失（需人工确认） | ❌ |
| POST | `/api/admin/orphans/restore` | 恢复（原件确认回来后） | ❌ |
| POST | `/api/admin/chat` | 管理端对话（与学生端共用同一套 RAG，逻辑隔离） | ✅ |
| GET/POST | `/api/admin/changes` | 元数据变更提案 / 确认应用 | ✅ / ❌ |

> 设计原则：**只有对话接口会烧钱**。用户在前端点筛选器、翻列表都不消耗 LLM。
> 交互式文档：<http://127.0.0.1:8000/docs>

### `/api/chat` 返回结构

```json
{
  "answer": "带 [资料1] 引用的 Markdown 回答",
  "route": "document | structured | hybrid | planning | general",
  "plan": { "reason": "...", "rewritten_query": "...", "doc_type": "...", "...": "..." },
  "sources": [
    {
      "ref": 1,
      "id": 2,
      "title": "关于做好2025年国家奖学金评选工作的通知",
      "publish_date": "2025-09-12",
      "department": "学生工作处",
      "doc_type": "奖学金",
      "snippet": "各学院、各位同学：根据《本专科生国家奖学金评审办法》……"
    }
  ],
  "documents": [ "去重后的文件列表" ],
  "trace": [ { "节点": "classify", "步骤": "意图识别", "结果": "...", "说明": "..." } ],
  "tool_calls": [ { "工具": "query_competition", "参数": {}, "结果": {} } ]
}
```

`sources` 里每个来源都带 `id` 和 `snippet`，这是前端「点开读原文」和「高亮被引用段落」的依据：

- `id` —— 向量库里只有 `file_path`，`id` 是 `_attach_source_ids()` 从 MySQL 回填的，
  没有它前端取不到原文；
- `snippet` —— 该文件被命中的原文片段，阅读器用它定位并高亮回答实际参考的那一段；
- `ref` —— 与回答里 `[资料N]` 的角标一一对应。

`trace` 与 `tool_calls` 仍然返回，但**前端不再展示**（用户不需要看决策链）。
留着是为了 `/docs` 里调试，以及以后做可观测性面板。

---

## 六、和你 10 阶段路线图的对应关系

| 阶段 | 状态 | 落在哪里 |
|---|---|---|
| 1. 学校文件解析 | ✅ | `app/parsing/`（默认链路，支持 pdf/docx/pptx/xlsx/md/txt）+ `app/rag/loader.py`（老链路） |
| 2. RAG 知识库 | ✅ | `app/parsing/chunker.py` + `app/rag/vectorstore.py` |
| 3. 自然语言文件搜索 | ✅ | `app/rag/retriever.py` + `search_school_document` |
| 4. MySQL 竞赛/考试库 | ✅ | `db/schema.sql` + `scripts/init_db.py` |
| 5. Python Tools | ✅ | `app/tools/*`（7 个工具） |
| 6. Agent 自动选择 RAG / Tool | ✅ | `app/agent/router.py` + `graph.py` |
| 7. LangGraph 工作流 | ✅ | `app/agent/graph.py`（4 节点 + 3 组条件边） |
| 8. 学习/竞赛规划 | ✅ | `recommend_competition` + planning 路由 |
| 9. Memory（多轮记忆） | ⚠️ 部分 | 目前由前端持有 `history` 并回传（无状态服务端）。真正的持久化记忆需要给 `StateGraph` 加 checkpointer |
| 10. FastAPI + 前端 | ✅ | `app/api/main.py` + `app/web/` |

### 阶段 9 还没做的部分

现在是「无状态服务端 + 客户端持有历史」，好处是简单、可水平扩展。
如果要升级成服务端记忆，思路是：

```python
from langgraph.checkpoint.sqlite import SqliteSaver

graph = build_graph().compile(checkpointer=SqliteSaver.from_conn_string("var/memory.sqlite"))
graph.invoke(state, config={"configurable": {"thread_id": session_id}})
```

但要注意：`trace` 和 `data_records` 用的是 `operator.add` reducer，
跨轮会不断累加，到时候需要在每轮开始时把它们重置掉。
（`sources` 已经在改版时改成覆盖语义了，不用再处理。）

---

## 七、前端设计说明

**方向**：Industrial / 工业实用 · 亮色基底。

选这个方向是因为校园信息检索就是「信息密集 + 大量时间/状态字段」的场景，而不是营销页。

### 界面只有两栏：左边功能，右边内容

试用第一版之后砍掉的东西，砍的理由都写在这里：

| 砍掉 | 为什么 |
|---|---|
| 首页那段技术说明（「非结构化走向量检索，结构化走 MySQL」） | 这是**实现原理**，不是用户要办的事。用户只关心「我的问题有没有答案、答案是从哪份文件来的」 |
| 右侧常驻的「Agent 决策过程」面板 | 同上。决策链是开发者的调试信息，放在页面上只会挤占阅读空间 |
| 回答末尾模型自己抄一遍的「📎 信息来源」清单 | 前端已经有一条可点击的参考资料列表，再抄一遍就是重复 |

留下并加强的：**每条回答正下方的「参考资料」**。

### 参考资料挂在回答下面，而不是放侧栏

这是这次改动里最核心的一个判断。参考资料属于**这一条回答**，不属于整个页面。
放侧栏 → 用户滚动时对不上是哪条回答引用的；挂在回答下方 → 一眼就是因果关系。

每张来源卡片包含四层信息，密度高但不吵：

```
[1]  关于做好2025年国家奖学金评选工作的通知          ← 文件标题
     2025-09-12 · 学生工作处 · 奖学金                ← 元数据（等宽数字）
     ▏各学院、各位同学：根据《本专科生国家奖学金…    ← 命中的原文片段（回答的依据）
                                              读原文  ← 整张卡片可点
```

- 回答正文里的 `[1]` 角标可点，点击后滚动到对应卡片并闪烁一下 → 满足「这句话是哪来的」。
- 点卡片 → 弹出**阅读器**，渲染该文件的完整原文。

### 阅读器：让「打开文件」真的有用

只给一个「下载」按钮等于没解决问题——用户要看的是内容。所以：

1. **完整原文内联渲染**：`GET /api/documents/{id}/raw` 读本地文件，剥掉 YAML 头（元数据
   顶部已经单独显示了，正文再来一遍是噪音），按「读一份文件」的排版渲染（15.5px / 行高 1.9）。
2. **自动高亮被引用的那一段**：拿 `sources[].snippet` 在原文里定位。片段入库时空白被压成了
   单空格，原文却保留着换行与缩进，所以匹配时把空白当等价处理（`[\s\u3000]*`）。
   实现上没有用「先转义再匹配」这种容易出错的路子，而是**先在原文上打哨兵字符**（`\x01`/`\x02`，
   不会被 HTML 转义、也不会出现在正文里），**再走 Markdown 渲染，最后一步才换成 `<mark>`**。
3. **复制全文 / 下载原件**：下载走 `Content-Disposition: attachment`，文件名带原始扩展名。
4. **路径安全**：`file_path` 虽然来自自己的数据库，但「把数据库字段当路径用」本身就是危险动作。
   `_resolve_document_file()` 会把解析后的绝对路径和项目根做一次 `is_relative_to` 校验，
   万一哪天库里有 `../../etc/passwd` 也读不出去。

### 视觉与响应式

- **字体**：JetBrains Mono（展示字 + 所有数字/时间字段，`tabular-nums`）+
  IBM Plex Sans（正文）。中文回退到 PingFang SC / 微软雅黑 / Noto Sans SC。
  没有用 Inter/Roboto/system-ui 这类默认字体做展示字。
- **配色**：暖纸白基底（`#f1eee6`，避开纯白）+ 暖调近黑（`#1a1815`，避开纯黑）
  + 单一强调色赭红 `#b8410f`。状态色（正在报名/未开始/已截止/已结束）是**语义色**，不算第二个强调色。
  卡片 hover 用左侧强调条擦出 + 右移 3px，而不是单纯变亮。
- **氛围层**：径向渐变 + 网格线（带径向遮罩淡出）+ 噪点叠加层。
- **响应式**：≥861px 两栏；≤860px 左栏变横向标签条、阅读器铺满全屏；
  ≤520px 输入区纵向堆叠、参考资料卡片收成两列（隐藏「读原文」提示，整张卡依然可点）。
- 全部动效包在 `prefers-reduced-motion: reduce` 降级内。

令牌定义在 `app/web/styles/tokens.css`，组件在 `components.css`。
第一版设计（三栏 + 轨迹面板）的截图留在 `.recon/ui_0*.png`，改版后的在 `.recon/ui4/`。

### 管理端：独立页面，刻意不进学生导航

管理端是 `admin.html` + `admin_chat.html`，和学生端**共用同一套 RAG 和同一个 collection**，
只是换了一套 `where` 条件和返回字段（`retriever.search_documents_admin` 传
`include_superseded=True, include_missing=True`，并回填 `document.id`）。
不另起一套向量库——那样两份库迟早不同步，而且重建一次要花双倍 embedding。

三条刻意的边界：

- **只改档案，不改正文**。管理端的「改元数据」改的是标题、发布部门、适用学院这类定位信息，
  正文永远以 `data/raw/` 里的原件为准。要改内容就改原件再灌库。
- **废止 ≠ 删除**。废止只改 `status`，原文和向量片段全部保留，`/raw`、`/download` 仍然可访问，
  撤销废止不用重算 embedding。「下架一份文件」的正确操作是改状态。
- **原件缺失只标记不删除**。文件从磁盘消失后，MySQL 留档、Chroma 片段保留
  （片段里有正文，本身就是备份），只把两处标记改成缺失、退出默认检索。
  扫描 `find_orphans()` 永远只读，真正的处置（标记/恢复）必须人工确认——
  路径配错一次就能把整个知识库标成缺失。

---

## 八、已知限制

1. **学生端没有鉴权**。学生画像是演示表，真实场景要接学校统一身份认证。
   管理端有门禁（`app/api/auth.py`：默认只允许本机回环，配了 `ADMIN_TOKEN` 后走请求头校验，
   并且是整个 `/api/admin` 前缀的中间件而非逐个路由加依赖），
   但本质是「一个共享令牌」，不是真正的账号体系，也没有操作审计。
2. **数据是仿真的**。`data/raw/` 下 115 份文档是我按真实公文格式造的（pdf / docx / md / txt 混合），
   把真实文件放进去重新跑 `run_ingest.py` 即可替换。元数据来源优先级：
   文件内 front-matter > 同目录的 `xxx.pdf.meta.yaml` 侧车 > 文件名日期 > 默认值。
3. **按日期判断报名状态依赖数据新鲜度**。竞赛库是种子数据，真实部署要接学校教务数据源。
4. **单轮检索**。复杂问题没有做 query 改写迭代（multi-hop），
   目前只做了一次 LLM 查询改写（`plan.rewritten_query`）。
5. **前端 Markdown 渲染器是自制的**，只支持回答里会出现的语法
   （标题/列表/表格/加粗/链接/引用块），不支持嵌套列表和 HTML 混排。
6. **列表页没有分页**。学科竞赛 / 考试信息默认一次渲染 50 行，
   页面高度到 3400~4000px。数据量再大需要接分页或虚拟滚动。
7. **字体走 CDN 非阻塞加载**。拉丁字体（IBM Plex Sans / JetBrains Mono）从
   Google Fonts 取，中文走系统字体栈（PingFang SC / Microsoft YaHei / Noto Sans SC）。
   国内加载不到拉丁字体时会静默回退，视觉不塌——这是刻意设计，
   但代价是首屏字体可能有一次替换闪烁（FOUT）。
8. **阅读器对部分格式还是空白**。`/raw` 里 `.md` / `.txt` 直接按 UTF-8 读，
   `.pdf` / `.docx` 走 `loader.load_one()` 解析出正文；但老链路 loader 不认
   `.pptx` / `.xlsx`，这两种文件能入库、能检索，点「查看原文」却是空白。
   修法是把 `/raw` 也接到 `app/parsing/` 上（同一套解析器，输出 Markdown 即可）。
   `/download` 不受影响，一直返回原件。
9. **高亮是「尽力而为」**。片段匹配用的是空白容忍的正则，
   如果模型引用的段落跨了 chunk 边界，或者片段被切分截断，就可能定位不到——
   这时不报错，只是不高亮。
10. **扫描件没有 OCR**。纯图片 PDF（一个字都没抽出来）会被标成 `empty` 状态并跳过，
    不会入库。当前 40 份测试 PDF 里有 4 份属于这种，是真实能力边界而非误判。
11. **复杂版式仍会漏**。双栏 PDF 的左右栏文字若在同一水平线上，会被拼进同一段
    （比"顺序不对"更糟）；没画线的表格（靠空格对齐）检测不到，会被当普通正文。
    有框线的表格渲染正常（实测 29 列 × 50 行的生源计划表，表头与合并单元格都对）。
12. **相关度闸门当前不生效**。`retriever` 里 `low_confidence` 判据依赖
    relevance 分数（`1-距离`，越大越相关），但 embedding 函数没注册到 Chroma 全局表，
    LangChain 的 relevance 接口会抛错，实际一律退回距离分数（越小越相关）。
    两者方向相反，所以闸门宁可不用也不误判 —— 代价是「相关度低」的提示永远不会触发。
13. **向量库跨进程不刷新**。Chroma 的 HNSW 索引是进程启动时的内存快照：
    另一个进程（`run_ingest.py`）写进去的片段，本进程 `count()` / `get()` 看得到，
    `query()` 却查不到。所以**灌库之后必须重启 web 服务**，否则「库里明明有、
    线上就是搜不到」，很容易误判成入库失败。同进程内上传则立即可查。

---

## 九、验证记录

| 检查项 | 结果 |
|---|---|
| 文档解析入库 | 115 份（pdf 20 / docx 20 / md 55 / txt 20）→ **828 个 chunk**，0 失败 0 跳过 |
| 向量库 | 828 条片段；换进程后 `count` / `get` / `query` 全通，chunk_id 828/828 合规 |
| 元数据 ↔ 向量库对账 | 115 ↔ 115，两向差集为 0 |
| 入库耗时 | 全量重建 45s（空库）/ 50s（重写已有）；增量复跑 2.3s、**0 次 embedding**；单份上传 1.4s |
| 单元测试 | `pytest` 156 passed |
| 5 条真实提问检索命中 | 5/5 命中，片段精确到「四、材料要求」这一级章节 |
| 结构化数据 | 竞赛 72 条 / 考试 72 条 / 学生 3 条 |
| 7 个 Tool 冒烟测试 | 全部通过 |
| 路由分类（7 条样例） | 全部正确 |
| 端到端（5 条不同类型提问） | 全部正确路由并生成带引用回答 |
| HTTP 接口回归（含 raw / download） | 全部 200；不存在的 id 正确返回 404 |
| JS 语法检查 | 通过（曾修复一处：中文对象键含全角逗号需加引号） |
| DOM id 一致性 | 49 个 JS 引用的 id 全部能在 HTML 中找到 |
| Python 未使用 import | 无（唯一报告项为 `from __future__ import annotations` 的检测误报） |
| 浏览器真实渲染 | Chrome 152 headless + CDP 实测：5 个标签页全部渲染，**无 JS 运行时错误** |
| 问答交互实测 | 点发送 → 真实返回，参考资料卡片 1 张并显示命中片段 |
| 点开原文实测 | 点卡片 → 阅读器渲染 10 个段落，**被引用段落高亮 1 处**，下载按钮可用 |
| 引用-来源对齐实测 | 回答中的 `[1]` 角标与参考资料卡片编号一致（`ref=1` 已回填） |
| 结构化问题不串味 | 「蓝桥杯历年报名时间」返回 0 张参考文件卡 + 表格 1 个（没瞎挂文件） |
| 换页状态保持 | 智能问答 ↔ 其余标签页来回切换，对话记录与筛选条件不丢失 |

浏览器验证的截图在 `.recon/ui4/`（截图脚本走 CDP 而非 selenium，
动作清单在 `.recon/steps_ui.txt`）。

实测两条链路的真实输出：

```
Q: 帮我找一下2025年国家奖学金评选的通知
→ route = document  (17.0s)
→ 轨迹：意图识别 → 学校文件检索(RAG) 命中 1 份 → 整合生成回答
→ 检索过滤条件：{'$and': [{'doc_type': {'$eq': '奖学金'}}, {'year': {'$eq': 2025}}]}
→ 回答主动加上：「该时间安排属于 2025 年评选周期，今天是 2026-09-15，窗口已过去，
                 不能据此认为现在正在报名」   ← 幻觉修复彻底生效

Q: 我是大二软件工程专业的学生，想参加明年的计算机类竞赛，现在应该准备什么？
→ route = planning  (16.2s)
→ 轨迹：意图识别 → 结构化数据查询(调用 2 个工具) → 学校文件检索(RAG) 命中 3 份 → 整合生成回答
→ 工具入参：recommend_competition(grade=大二, major=软件工程, interests=算法,计算机竞赛)
           query_competition(category=计算机, grade=大二, limit=20)
→ 回答了「2027 届尚无数据」而不是编造明年日期，并给出按往年规律推算的时间节点
```

### 修复过的真实缺陷

| # | 问题 | 原因 | 修复 |
|---|---|---|---|
| 1 | 2025 年的奖学金通知被判为「正在报名」 | 模型把文件里的「9月15日—9月21日」当成今年的日期 | 在 compose 提示词里明确「文件内日期属于该文件发布年份」，并要求只有结构化数据给了「阶段」字段才能断言报名状态 |
| 2 | 给 Web 开发兴趣的学生推荐英语演讲赛 | 加分项（国家级 +2、专业不限 +1）被当成了**放行**条件 | 改为方向不匹配直接淘汰，加分项只用于排序 |
| 3 | 「四六级」查不到任何记录 | 库里存的是「四级」「六级」，字面 LIKE 匹配不上 | 加 `KEYWORD_ALIASES` 同义词映射 + `or_like` OR 展开 |
| 4 | 同一赛事在推荐列表里出现 5 次 | 未按赛事名去重 | 排序后按名称去重，保留最相关的一届 |
| 5 | `蓝桥杯 2025` 的报名时间算成了 2025-11 | 报名月份晚于比赛月份时，年份没往前推 | 抽 `_align()`：year 表示**比赛年份**，报名跨年时落在前一年 |
| 6 | 引用编号 `[资料2]` 和来源卡片对不上 | 上下文按 chunk 编号，卡片按文件去重 | `format_context` 改为按文件分组编号，与来源卡片 1:1 |
| 7 | 前端 JS 直接语法报错 | 对象键 `报名已截止，待举行` 含全角逗号，不是合法标识符字符 | 键加引号 |
| 8 | `document` 表被清空 | 重跑 `init_db.py` 时 `schema.sql` 会 `DROP TABLE document` | 在 `init_db.py` 结尾打印顺序提醒 |
| 9 | 点「读原文」一律 404 | 灌库时 `file_path` 只存了 `path.name`（文件名），没人知道它在哪个目录 | 改存相对项目根的 POSIX 路径（`data/raw/xxx.md`），修完重跑 `run_ingest.py` |
| 10 | 回答里的 `[1]` 角标点不动来源卡片 | `doc_context_count` 没写进 `AgentState`，**LangGraph 的状态 schema 会直接丢弃未声明的键**，引用编号回填逻辑永远拿到 0 | 在 `AgentState` 里声明该字段 |
| 11 | `sources` 用 `operator.add` 累加 | 来源是「本轮检索结果」，累加语义下同一份文件会逐轮追加、编号错位（接 checkpointer 后会立刻爆） | 改成覆盖语义 |
| 12 | 阅读器里文件自己的「1. 2. 3.」全没了 | `base.css` 全局清掉了 `list-style`（页面用自定义标记），`.reader` 没有重新声明 | `.reader` 单独恢复真实序号 |

---

## 十、下一步可以做什么

1. **加 checkpointer 做真正的多轮记忆**（阶段 9）
2. **加 rerank**：检索后用 bge-reranker 之类的小模型重排，召回精度会再上一档
3. **Query 改写多跳**：把「我是大二计算机专业适合参加什么」拆成多个子查询并行检索
4. **接入真实数据**：把 `data/raw/` 换成学校真实文件，竞赛库接教务系统
5. **给解析链路补评测**。`evals/` 已经覆盖路由、检索、答案忠实度三层，
   但没有「切分切得对不对」的量化指标 —— 表格有没有被切成独立块、章节标题有没有漏，
   目前只能靠人工抽查逐份看。
6. **把线上换成 v2 链路**。服务器上跑的还是旧代码 + 旧向量库。切换时要按顺序：
   上传代码 → `run_ingest.py` 重建 → **重启 systemd**。
   最后一步不能省，Chroma 的 HNSW 索引是进程启动时的内存快照，
   不重启的话库里明明有新数据，线上却搜不到。
