# 校园智助 · Campus Agent

用自然语言检索学校文件、查询竞赛与考试信息、生成学业规划建议的校园 Agent 系统。

严格按你给的架构实现：**RAG（非结构化文件）+ MySQL（结构化数据）+ Tools + LangGraph Agent**。

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
│   │
│   ├── rag/                  # 第一条线：RAG
│   │   ├── loader.py         #   文档加载 + YAML front-matter 元数据抽取
│   │   ├── splitter.py       #   中文友好切分（chunk 补标题）
│   │   ├── vectorstore.py    #   Chroma + DashScope Embedding
│   │   └── retriever.py      #   语义检索 + 元数据过滤
│   │
│   ├── tools/                # Tools 层（Agent 的手）
│   │   ├── common.py         #   年级翻译、报名窗口判断
│   │   ├── document.py       #   search_school_document / list_school_documents
│   │   │                     #   get_document_detail
│   │   ├── competition.py    #   query_competition / recommend_competition
│   │   ├── exam.py           #   query_exam
│   │   └── student.py        #   get_student_profile
│   │
│   ├── agent/                # LangGraph Agent
│   │   ├── llm.py            #   LLM 客户端工厂
│   │   ├── router.py         #   意图识别 + 参数抽取（结构化输出）
│   │   ├── state.py          #   状态定义（reducer vs 覆盖）
│   │   └── graph.py          #   状态机组装
│   │
│   ├── api/main.py           # FastAPI（14 个路由）
│   └── web/                  # 前端（无框架无构建）
│       ├── index.html
│       ├── styles/{tokens,base,components}.css
│       └── scripts/main.js
│
├── db/schema.sql             # 4 张表：document / competition / exam / student
├── data/raw/                 # 学校文件（21 份仿真文档）
├── scripts/
│   ├── init_db.py            # 建库建表 + 结构化种子数据
│   ├── gen_sample_docs.py    # 生成仿真学校文件
│   ├── seed_data.py          # 竞赛/考试/学生种子数据
│   └── test_tools.py         # 工具层冒烟测试（不经过 LLM）
└── var/chroma/               # 向量库持久化目录
```

---

## 三、快速开始

### 0. 前置条件

- Python：`D:\AgentLearnVenv`（Python 3.12.6）
- MySQL 5.7+（本地 3306 端口，库名 `school_agent`）：连接信息请写进 `.env`，不要提交
- API Key（机器级环境变量，已存在）：
  - `DEEPSEEK_API_KEY` — 对话模型
  - `DASHSCOPE_API_KEY` — 向量模型（DeepSeek 没有 embedding 接口）

> 首次使用请先补依赖：`pip install -r requirements.txt`

### 1. 初始化数据库（结构化数据）

```powershell
cd "D:\111网安学习\Agent学习6"
D:\AgentLearnVenv\Scripts\python.exe scripts\gen_sample_docs.py   # 生成 21 份学校文件
D:\AgentLearnVenv\Scripts\python.exe scripts\init_db.py           # 建表 + 灌竞赛/考试
```

> ⚠️ **顺序很重要**：`init_db.py` 里的 `schema.sql` 会 `DROP TABLE document`，
> 所以它必须在 `run_ingest.py` **之前**跑。反过来跑会把文件元数据冲掉。

### 2. 学校文件灌库（第一条线）

```powershell
D:\AgentLearnVenv\Scripts\python.exe run_ingest.py
```

这一步会：解析文档 → 切分（21 份 → 37 个 chunk）→ 写 MySQL 元数据表 → 写 Chroma 向量库。
需要调用 DashScope Embedding 接口，约 4 次请求。

### 3. 自检（可选但推荐）

```powershell
# 不消耗 LLM，直接验证 7 个工具
D:\AgentLearnVenv\Scripts\python.exe scripts\test_tools.py

# 只验证检索质量
D:\AgentLearnVenv\Scripts\python.exe -m app.rag.retriever

# 只验证路由分类
D:\AgentLearnVenv\Scripts\python.exe -m app.agent.router

# 端到端跑 5 条不同类型的提问
D:\AgentLearnVenv\Scripts\python.exe -m app.agent.graph
```

### 4. 启动服务

```powershell
D:\AgentLearnVenv\Scripts\python.exe -m app.api.main
```

打开 <http://127.0.0.1:8000>。

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

> 设计原则：**只有 `/api/chat` 会烧钱**。用户在前端点筛选器、翻列表都不消耗 LLM。
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
| 1. 学校文件解析 | ✅ | `app/rag/loader.py` |
| 2. RAG 知识库 | ✅ | `app/rag/splitter.py` + `vectorstore.py` |
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

---

## 八、已知限制

1. **没有鉴权**。学生画像是演示表，真实场景要接学校统一身份认证。
2. **数据是仿真的**。`data/raw/` 下 21 份文档是我按真实公文格式造的，
   把真实 PDF/Word 放进去重新跑 `run_ingest.py` 即可替换（保留 YAML 头部格式，
   或删掉头部让程序从文件名猜元数据）。
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
8. **阅读器只认文本类文件**。`/raw` 直接把文件当 UTF-8 文本读，
   所以 `.md` / `.txt` 能读；换成真实 `.pdf` / `.docx` 时这里会读成乱码，
   需要接 loader 里已有的 `_read_pdf` / `_read_docx` 来做解析。
9. **高亮是「尽力而为」**。片段匹配用的是空白容忍的正则，
   如果模型引用的段落跨了 chunk 边界，或者片段被切分截断，就可能定位不到——
   这时不报错，只是不高亮。

---

## 九、验证记录

| 检查项 | 结果 |
|---|---|
| 文档加载与切分 | 21 份 → 37 个 chunk |
| 向量库 | 37 条记录 |
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
5. **加评测**：建一个 30~50 条的问题-标准答案集，量化路由准确率和答案忠实度
   （这一步对面试最有说服力）
