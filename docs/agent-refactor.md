# 校园智助 · Agent 化重构方案

> 目标读者：项目作者本人。用途有两个 —— ① 分阶段实施时的对照依据；② 面试时讲"架构演进"的底稿。
> 结论先行：**不推翻现有代码，把它从「带意图路由的 RAG 流水线」升级为「分层 Agent」。**

---

## 〇、一句话诊断

现在的链路是：

```
classify → 5 条写死的分支 → compose → END
```

它有三个特征说明自己不是 Agent：

1. **终止由边决定**，不由目标判断 —— `compose → END` 是无条件的，检索 0 命中也会生成一段"未检索到"然后返回 200。
2. **决策权在编译期** —— 路径由 `_after_classify` 返回的边名决定，模型做的是"分类"，不是"决定下一步"。把 LLM 换成一个查表函数，图照样能跑完。
3. **记忆在浏览器里** —— `session_id` 传进了 config，但编译图时没挂 checkpointer，它只被工具用来定位附件目录。

真正的 Agent 骨架只有一个半成品：管理端的 `agent ⇄ tools` ReAct 循环。

所以重构的核心不是"改成 ReAct"，而是补三个东西：**持久状态、目标判定、可恢复的执行层**。

---

## 一、角色定位与业务目标

### 1.1 角色

现状的隐含定位是「校园信息检索器」——只回答"文件里写了什么"。

目标定位：**学业事务助理（Academic Task Assistant）**。

差别在于学生真正的问题往往是一串动作，不是一个查询：

> 「我是大二软件工程的，明年想参加蓝桥杯，现在该准备什么？」

这句话里藏着：查学生画像 → 查竞赛窗口 → 查章程原文 → 比对年级资格 → 排时间线。现在系统会把它判成 `planning`，然后固定走 `query_data → retrieve_docs`，两边各查一次就收尾，中间没有任何"我还缺什么"的判断。

### 1.2 能力边界（必须写死，否则一定做成空话）

**做：**
- 检索与解释学校文件、政策、通知
- 查询竞赛 / 考试等结构化信息，判断报名窗口
- 结合学生画像给建议、排时间线
- 明确指出"库里没有"，并给出可执行的替代建议（找谁、查哪里）

**不做：**
- 不代办任何事务（报名、提交材料、缴费）—— 需要对接教务/一卡通，越权且不可回滚
- 不回答知识库外的事实性问题（校长是谁、食堂几点开）
- 不自主修改知识库（写操作一律人工确认，沿用现有变更单机制）

这条边界同时是**产品边界**和**安全边界**。面试被问"你的 Agent 能干什么"时，能说清"不做什么"比罗列功能可信得多。

### 1.3 业务目标的度量方式

不设拍脑袋的目标值，先立指标，跑完基线再定线：

| 指标 | 含义 | 怎么测 |
|---|---|---|
| 事实通过率 | `must_contain` 全中的比例 | `evals/run_eval.py --layer e2e` |
| 多轮一次解决率 | 3 轮内不需要用户重述前提的比例 | 新增多轮评测集（见 §7.4） |
| 误答率 | 该拒答却给了答案的比例 | 负样本集（知识库外 + 参数错） |
| 降级成功率 | 依赖故障时仍给出可用结果的比例 | 故障注入（见 §7.4） |
| P95 延迟 / 单次成本 | 体验与钱 | 现有 bench 脚本 + token 统计 |

---

## 二、能力模块拆解

六个模块。**每个模块都必须能回答"我不做什么"** —— 边界不清是 Agent 重构最常见的死法。

### M1 会话与状态（`app/agent/memory.py`，新增）

| | |
|---|---|
| **职责** | 持久化会话；维护两层记忆；提供上下文预算 |
| **不做什么** | 不做业务判断、不调 LLM、不碰检索 |
| **输入** | `session_id`、本轮 question |
| **输出** | `MemoryView{turns, entities, facts, summary}` + `Budget` |

两层记忆要分开，这是设计要点：

- **会话记忆（跨轮、持久）**：完整 turns、**已确认实体**（`doc_ids`、`competition_names`、`filters`、`student_no`）、已确认事实（"国奖名额 45"这类已经查证过的结论）。
- **工作记忆（单轮、易失）**：本轮 `tasks`、`evidence`、`tool_records`、`trace`。

为什么实体要单独存：现在"12 号那份"这个指代，每一轮都得靠把 6 轮原文重塞给模型让它重读。存成 `entities.current_doc_id = 12` 之后，下一轮直接查表，不靠猜。

**现状差异**：`AgentState` 是单次 invoke 内的临时结构，`history` 由前端传入（`main.js` 只留 3 轮，刷新即失）。改造后服务端成为真相源，前端传入降级为兼容路径。

### M2 理解与分解（`app/agent/understanding.py`，由 `router.py` 演进）

| | |
|---|---|
| **职责** | 指代消解 → 独立 query；意图判定；**多意图拆成子任务**；参数抽取（带枚举约束） |
| **不做什么** | 不决定执行顺序（M3 的事）、不选工具（M4 的事） |
| **输入** | `question` + `MemoryView` |
| **输出** | `TaskSpec{tasks: list[SubTask], constraints, confidence, need_clarify}` |

`SubTask` 的定义是整个重构的地基：

```python
SubTask = {
    "id": "t1",
    "route": "document | structured | hybrid | planning | general",
    "query": "改写后的独立检索语句",
    "filters": {"doc_type": "奖学金", "year": 2025, ...},
    "target": "这份子任务要回答的那句话",
}
```

**现状差异**：
- 现在 `classify()` 的入参只有 `question`，而 `compose` 有 history —— 回答层知道上下文、检索层不知道。`RoutePlan` 里其实已经有 `rewritten_query` 字段，**缺的只是把 history 喂进去**。这是全项目性价比最高的一处修复。
- 现在 `route` 是单值，装不下"找报名时间 + 对比两个竞赛"这种复合问题。

**关键取舍**：指代消解和意图分类可以合成一次 LLM 调用（把 history 塞进同一个 prompt），省一次往返；也可以拆成两个节点，各自 prompt 更短、更可控。建议**先合并**，因为延迟敏感；等评测证明分类准确率掉了再拆。

### M3 规划与编排（`app/agent/planner.py`，新增）

| | |
|---|---|
| **职责** | 决定**用哪档算力**，而不是决定第几步做什么 |
| **不做什么** | 不执行、不生成答案、不选具体工具 |
| **输入** | `TaskSpec` + `Budget` |
| **输出** | `mode ∈ {fast, deep, clarify}` |

三档的含义：

| 档位 | 触发条件 | 执行方式 | LLM 次数 |
|---|---|---|---|
| `clarify` | 关键参数缺失且无法推断（问"我适合什么竞赛"但不知道年级、也没学号） | 生成一个反问，不检索 | 1 |
| `fast` | 单子任务 + 高置信 | 固定链（现 retrieve/query + compose） | 2–3 |
| `deep` | 多子任务 / 低置信 / planning 类 / 需要多步探询 | ReAct 循环，工具全集，带预算 | 3–N |

**这是保留 Workflow 优点的关键设计。** 90% 的校园问题（"国奖多少名额"）不值得走 ReAct：多几次 LLM 往返、多几秒延迟、还更容易跑偏。把 router 的职责从「选分支」升级为「选算力档位」，既拿到了 Agent 的灵活性，又不牺牲简单场景的体验。

**现状差异**：`_after_classify` 返回的是边名，路径在 `build_graph()` 里就固定了。改造后 `mode` 只决定进入哪个子图，子图内部仍可自由循环。

### M4 工具执行（`app/agent/executor.py` + `app/tools/base.py`，新增）

| | |
|---|---|
| **职责** | 执行工具；去重；错误分类；结果转成**结构化证据**；预算扣减 |
| **不做什么** | 不承载业务规则（业务规则留在各 tool 内，这一点现状是对的） |
| **输入** | 模型给出的 tool_calls + `Budget` |
| **输出** | `list[Evidence]` + `list[ToolRecord]` |

**核心变化：把"上下文字符串"升级成"结构化证据"。**

现在 `format_context()` 把 hits 拼成一大段带 `[资料N]` 的文本塞给模型，信息在这里就退化成不可计算的东西了 —— 没人知道这段文字对应哪个子任务、可靠度多少、是不是放宽条件后得到的。

```python
Evidence = {
    "task_id": "t1",            # 它回答了哪个子任务
    "kind": "doc | table",      # 文件片段 / 结构化数据
    "source_ids": [12, 15],     # 对应 MySQL document.id
    "text": "...",              # 进上下文的那段
    "score": 0.83,              # 相关度（表类为 None）
    "relaxed": False,           # 是否放宽过过滤条件
    "filter_rejected": [],      # 被工具忽略的参数（见下）
}
```

这个结构一出来，M5 才有东西可判、M6 才有依据分配引用编号。**其余所有模块的产出都汇到这里，这是整个重构的枢纽。**

工具返回值也要统一信封，但**必须平铺兼容**：

```python
{
    # ...原有字段原样保留（"文件列表"、"原文片段"、"数量"）—— prompt 依赖它们
    "_meta": {
        "ok": True,
        "found": 3,
        "relaxed": False,             # 是否放宽过滤条件
        "filter_rejected": ["category=计算机类"],  # 被忽略的非法枚举
        "retryable": True,
    }
}
```

为什么要平铺而不是包一层：现有 prompt 和评测都依赖工具返回的字段名，包成 `{data: {...}}` 会让模型看不见原字段，属于静默回归。

**去重与无进展检测**：

```python
fingerprint = (name, json.dumps(args, sort_keys=True))
```

重复调用直接返回"这一步已查过，结果同上"，并计数；连续两次重复 → 判定绕圈，强制进入收尾。

### M5 校验与终止（`app/agent/verifier.py`，新增）

| | |
|---|---|
| **职责** | 判断"证据够不够"、"能不能收尾"、"输出是否被证据支撑" |
| **不做什么** | 不生成内容（用代码，不用 LLM 裁判） |
| **输入** | `tasks` + `evidence` + `draft_answer` |
| **输出** | `verdict ∈ {pass, retry, degrade}` + `caveats` |

四条判定，全部可代码实现：

1. **覆盖度**：每个 `SubTask` 至少有一条 `evidence`。缺的标出来 → `retry`（针对缺的那条重查一次）。
2. **可靠度**：`max(score) >= SCORE_FLOOR` 且 `not relaxed`。阈值按 embedding 模型实测标定，不拍脑袋。
3. **引用一致性**：正则抽 `[资料N]`，检查 `N ∈ [1, len(sources)]`。
4. **数字可溯**：回答里的数字（含中文数字，归一化后）能在 evidence 文本里找到。这条能挡住最常见的幻觉：金额、名额、日期写错一位。

第 4 条可能误伤（模型做了合理的单位换算或四舍五入），所以**只记录不拦截**，作为质量信号进 trace 和评测。

**为什么不用 LLM 当裁判**：和 `evals/README.md` 里同样的理由 —— 可复现、可解释、可回归。

**现状差异**：现在没有任何校验，`compose` 之后无条件 `END`。"我还缺信息"和"我已经答完了"在图里是同一条边。

### M6 输出与呈现（`app/agent/compose.py`，迁出现有 `compose_node`）

| | |
|---|---|
| **职责** | 组装答案、分配引用编号、生成保留意见、降级文案 |
| **不做什么** | 不重新检索、不做判断（判断在 M5） |
| **输出契约** | `{answer, sources[], confidence, caveats[], trace[]}` |

两处新增：

- **按证据而不是按检索顺序编号**：编号跟着 `Evidence.source_ids` 走，只给真正进了上下文的那几条。现在 `doc_context_count` 用 `min(len(documents), 5)` 推上限，一旦 `format_context` 的去重逻辑和 `documents` 的排序不一致就会错位。
- **`confidence` 与 `caveats`**：前端要能渲染"这条答案是基于放宽条件后的结果，可能不完全匹配你的限定"。没有这个字段，模型的不确定性只能挤在正文里，用户读不出来。

---

## 三、数据流

```
 请求 (question, session_id)
   │
   ▼
┌──────────────────────────────────────────┐
│ M1 会话与状态  载入 turns / entities / budget │
└───────────────┬──────────────────────────┘
                ▼
┌──────────────────────────────────────────┐
│ M2 理解与分解  (question + memory) → TaskSpec │
└───────────────┬──────────────────────────┘
                ▼
┌──────────────────────────────────────────┐
│ M3 规划        TaskSpec → mode               │
└───┬───────────┬───────────────────┬──────┘
    │clarify    │fast               │deep
    ▼           ▼                   ▼
 ┌──────┐  ┌────────────┐   ┌──────────────────┐
 │ 反问  │  │ 固定两查    │   │ ReAct: 调工具 → 判  │
 └──┬───┘  └─────┬──────┘   │ 够不够 → 再调 ...   │
    │            │          └────────┬─────────┘
    │            └───────┬───────────┘
    │                    ▼
    │        ┌──────────────────────────┐
    │        │ M5 校验: 覆盖 / 可靠 / 引用   │
    │        │   pass │ retry(定向补查) │ degrade
    │        └───────────┬──────────────┘
    ▼                    ▼
┌──────────────────────────────────────────┐
│ M6 输出  answer + sources + confidence     │
└───────────────┬──────────────────────────┘
                ▼
       M1 回写: turns、entities、本轮来源 id
```

**注意两处与现状的本质差别**：

1. `M5` 在 `deep` 模式里**每步都参与**，不是最后一道闸门。每一步执行完就问一次"够了吗"，够了就提前退出，不必跑满预算。这是"终止判断"从拓扑驱动变成目标驱动的落点。
2. `M1 → M6 → M1` 是一个环。现在这条环是断的（前端自己 push 历史），所以系统没有"这一轮学到了什么"的概念。

### 3.1 State 契约

模块之间只通过 state 通信，不互相 import 业务函数。这条要在重构时立住，否则半年后又变回一坨。

```python
class AgentState(TypedDict, total=False):
    # 输入
    question: str
    session_id: str
    history: list[dict]                 # 兼容：无 session 时的降级路径

    # M1
    memory: dict                        # turns_summary, entities, facts
    budget: dict                        # max_steps, deadline_ms, max_chars

    # M2
    tasks: list[dict]
    confidence: float
    need_clarify: bool

    # M3
    mode: str                           # fast | deep | clarify

    # M4
    evidence: list[dict]                # 覆盖语义，每轮重算
    tool_records: Annotated[list, operator.add]

    # M5
    verdict: str                        # pass | retry | degrade
    caveats: list[str]

    # M6
    answer: str
    sources: list[dict]                 # 覆盖语义（沿用现有注释里的理由）
    trace: Annotated[list, operator.add]
```

| 字段 | 谁写 | 谁读 |
|---|---|---|
| `memory` | M1 | M2 / M3 / M6 |
| `tasks` | M2 | M3 / M4 / M5 |
| `mode` | M3 | 图边 |
| `evidence` | M4 | M5 / M6 |
| `verdict` | M5 | 图边 / M6 |
| `sources` | M6 | 前端 |
| `trace` | 所有 | 前端 / 评测 |

**一条硬规则：`evidence` 和 `sources` 用覆盖，`tool_records` 和 `trace` 用累加。** 理由和现有 `state.py` 里写的一样 —— 证据是"本轮结论"，来源是"这一条回答的引用"，累加会让同一份文件逐轮追加、编号错位。

---

## 四、模块协作方式

### 4.1 结构化协作（图边）

只有两个地方需要"图"来表达分支，其余全是线性：

- `M3 → {clarify, fast, deep}`：三档
- `M5 → {retry, compose, degrade}`：三态

`retry` 是唯一的新边，必须带**定向性**：不是"整轮重跑"，而是"只针对没覆盖的 task_id 重查"。否则一次失败就会把已成功的子任务也重跑一遍，延迟翻倍。

### 4.2 隐式协作（state 契约）

其余协作靠 state 字段，不靠直接调用。举一个现在就该改的例子：

```python
# 现在：节点里直接调业务函数，检索逻辑焊死在节点上
result = search_documents(query, doc_type=..., ...)

# 重构后：节点只做 取 state → 调服务 → 写 state
ev = tool_layer.search_documents(query, filters, task_id="t1")
return {"evidence": [ev]}
```

好处不是"代码好看"，而是**换实现不用改图**：以后要加 BM25 混合检索、要加 rerank，只改 tool 层，图和 prompt 一行不动。

### 4.3 三条关键流程

**流程 A · 单轮事实型（fast，最短路径）**

```
"2025 年国奖我们学校有几个名额"
→ M2: tasks=[{t1, document, "2025年国家奖学金名额", {doc_type:奖学金, year:2025}}]
→ M3: fast
→ M4: search_school_document → Evidence(t1, doc#45, score 0.83)
→ M5: 覆盖✓ 可靠✓ → pass
→ M6: 答案 + [资料1] + sources=[{id:45, ref:1}]
→ M1: entities.doc_ids=[45]
```

**流程 B · 多轮指代（deep，重构价值最集中的一条）**

```
轮1 "帮我找计算机学院的竞赛通知"
  → M2 写入 entities={college:计算机学院, doc_ids:[8,9]}
  → sources 落库

轮2 "那报名截止呢"
  现在：classify 只看到这六个字 → 判不出指代 → 检索跑偏 → 答"未检索到"
  重构后：
    M1 载入 entities{college:计算机学院, doc_ids:[8,9]}
    M2 消解 → query="计算机学院竞赛通知的报名截止时间"，继承 filters
    M3 → hybrid（既要知道哪份通知，也要确切日期）
    M4 先查 doc_ids 命中的那两份的元数据，再查 competition 表
    M5 覆盖：两个子问题都有证据
    M6 回答 + 标注"依据第 8、9 号文件"

轮3 "我大二能参加吗"
  → M1 的 entities 已含 doc_ids + 用户年级（轮1/2 若提过）
  → M3 deep：要查 target_grade 再比对
  → 不需要用户重述"我问的是哪个竞赛"
```

这条流程里，**每一轮的输出都在给下一轮铺路**，这是现在完全缺失的能力。

**流程 C · 失败降级（异常路径）**

```
检索 0 命中 + max_score 0.21
→ M5: 覆盖✗（t1 无证据）→ retry(定向：放宽过滤 + k×2)
→ 仍 0 命中
→ M5: degrade
→ M6: "没找到 2026 年的评审通知。库里最近的是 2025 年那份（附卡片），
       2026 年的通常 9 月发布，可以关注教务处。"
      confidence=low, caveats=["未命中精确年份"]
```

关键差别：现在的系统在这种情况下会输出"未检索到相关内容"然后完事 —— 用户拿走的是零信息。降级路径要保证**即使查不到，也给下一步**。

---

## 五、兼容性要点

这四条不做，重构会变成一个永远上不了线的分支。

### 5.1 接口只加不删

`POST /api/chat` 的响应字段 `answer / route / plan / sources / documents / trace / tool_calls` 前端都在用。新增 `confidence / caveats / tasks / mode` 可以，**删改任何一个 = 前端白屏**。

`route` 字段在 `deep` 模式下语义会变（不再单值），处理方式：保留 `route` 为"主路由"，另加 `tasks[]`。前端不改也能跑。

### 5.2 无 session 时降级为无状态

`ChatRequest.history` 必须继续能用。判据：

```python
if req.session_id and memory_available():
    memory = load_session(req.session_id)
else:
    memory = MemoryView.from_history(req.history)   # 老路径
```

这样灰度期间新旧两条路都能跑，出问题可以按 session 回滚。

### 5.3 数据层不动

- **Chroma collection 名称、metadata 结构不改**，否则要全量重灌、重花 embedding 的钱。要加 `content_type`（区分表格片段）这类字段，走"新文件才带、老片段缺字段时按文本处理"的兼容分支 —— 现有 `_library_lacks_state_fields()` 已经是这个套路，照抄。
- **MySQL 只加表不加列**：会话表 `agent_session` 新开一张，不动 `document` / `competition`。理由：`init_db.py` 会 `DROP TABLE document`，往老表加列意味着每次初始化都要重建。
- 写操作链路（变更单 → 人工确认 → apply）**完全不动**，它是现在唯一做对"人机权限边界"的地方。

### 5.4 意图枚举要外置

现在 `DocType` / `Category` / `ExamCategory` 是硬编码在 `router.py` 的 `Literal` 里的。加一个新领域（比如"宿舍报修"）要改代码、重启、还得同步改 prompt 里的枚举说明。

改造时把它们抽到配置或从库里 `SELECT DISTINCT` 读，**但 `Literal` 不能直接换成 `str`** —— 那会把现在的"模型传错值"的防护也一起丢掉。正确做法：保留 Pydantic 校验，枚举值从外部注入（`Enum` 动态构造或 `validator` 里查白名单）。

---

## 六、可扩展性要点

| 扩展场景 | 现在要改什么 | 重构后 |
|---|---|---|
| 加一个工具 | 改 `tools/__init__` 分组 + 检查各分支是否需要它 | 注册即用（M4 统一工具集） |
| 加一种意图 | 改 `Literal` + 改 router prompt + 改 `build_graph` 的边映射 | 加枚举值 + 加一个子链实现 |
| 加一个领域（报修/图书馆） | 上面三项全改 | 加工具 + 加枚举值 |
| 换检索实现（BM25/rerank） | 改 `retriever.py`，可能牵动 `format_context` 的返回结构 | 只改 tool 层，`Evidence` 接口不变 |
| 换向量库 | 改 `vectorstore.py` + 重灌 | 同左（这层本来就隔离得不错） |
| 加评测维度 | 改 `run_eval.py` 的判分 | 同左，但 `evidence` 让"哪一步导致的错误"可自动定位 |

**一个必须提前想的扩展点**：`evidence` 带 `task_id` 之后，评测就能做"归因" —— 最终答案错了，是 task 拆错了、检索没命中、还是 compose 读错了？现在只能人工翻 trace。这个能力一旦有，迭代速度会上一个台阶。

---

## 七、异常处理要点

### 7.1 分层降级矩阵

原则：**任何一层失败都不应该让整轮 500。** 校园场景里，给出"文件列表 + 一句说明"比给出错误页有价值得多。

| 失败点 | 现象 | 降级动作 | 用户看到 |
|---|---|---|---|
| M2 结构化输出解析失败 | Pydantic validation error | 退化为 `document` + 原问题当 query | 正常回答（可能略偏） |
| M2 置信度低 | confidence < floor | 转 `clarify` 或 `deep` | 一个反问 / 更多检索 |
| M4 embedding 超时 | DashScope 5xx | 关键词检索（`list_school_documents` 的 LIKE 路径） | 结果标注"关键词匹配" |
| M4 DB 断连 | 连接异常 | 只走 RAG | 标注"结构化数据暂不可用" |
| M4 工具参数非法 | `filter_rejected` 非空 | 让模型改一次参数 | 正常（若仍失败则如实说"没有该类竞赛"） |
| M5 retry 后仍无证据 | 空 evidence | `degrade` | 近似结果 + 下一步建议 |
| M6 LLM 失败 | 超时/5xx | 模板文案 + sources 卡片 | "已找到这些文件，但生成回答失败" |
| 预算耗尽 | steps / 时间到 | `summarize` 节点 | 部分答案 + 还缺什么 |

**最后一行是重点**：现在步数用尽是硬截断，然后模板拼一句"我查了 N 次还没得到确定结论"（`admin_graph.py:292-298`）。改成"到顶前插一个总结节点"，回答从废话变成"已确认 A、B，C 需要你补充条件"。

### 7.2 错误要分类，不能一律 `except Exception`

现在两类错误被混在一起：**参数错（可以重试）** 和 **环境坏（重试白搭）**。模型看到"错误"会盲目重试同一个 dead tool，把预算烧光。

```
retryable=True   : 参数类型错、枚举非法、找不到 id（可以改一次再试）
retryable=False  : DB 断连、文件系统权限、向量库缺字段（立刻终止循环）
```

### 7.3 幂等与安全

- `evidence` / `sources` 用覆盖不用累加（否则重试一次就多一套引用编号）。
- 写操作（`admin_propose_*`）永远只生成变更单，`apply` 只走人工接口 —— 这条现状是对的，**重构时不许为了"Agent 自主性"把它放开**。这是全项目最重要的一条设计约束。
- session 表要按 `session_id` 隔离，读历史前校验归属，否则任何客户端都能读别人的会话。

### 7.4 评测必须同步扩

现有 50 条用例**全是单轮**，重构后有一半能力（指代消解、多轮实体、定向重试）根本没有样本能验证。不加评测就重构，等于闭着眼改。

需要补三组（细节见 `evals/README.md` §7 已列出的局限，这里落成具体项）：

| 组 | 条数（建议） | 考什么 |
|---|---|---|
| 多轮指代 | 15 条（每条 2–3 轮） | entities 是否被正确继承；第 3 轮是否还需要用户重述 |
| 参数鲁棒 | 6 条 | 故意让模型传非法枚举，看是否被规范化而非静默返回 0 |
| 故障注入 | 8 条 | monkeypatch 掉 DB / embedding，看降级路径是否给出可用结果 |

写进 `cases.jsonl` 时用 `turns: [...]` 替代 `question`，`run_eval.py` 加一个多轮分支；老用例不动，保证 `--layer all` 仍能跑。

---

## 八、实施顺序

分五段，每段独立可上线、可回滚、可用评测对比。顺序按「依赖关系 × 收益」排，不按模块编号。

| 阶段 | 内容 | 改动量 | 风险 | 验收 |
|---|---|---|---|---|
| **P0 契约层** | `AgentState` 加字段；工具返回加 `_meta`（平铺兼容）；trace 加 `step_id`；`Budget` | 小 | 极低（行为不变） | 全部旧用例结果不变 |
| **P1 理解层** | history 进 classify；枚举参数规范化；检索分数闸门 + `relaxed` 标注 | 小 | 低 | `--layer router` 不退；负样本误答率下降 |
| **P2 记忆层** | session 表 + checkpointer；entities 回写 | 中 | 中（涉及持久化） | 新增多轮集通过 |
| **P3 编排层** | planner 三档；deep 模式统一工具集 ReAct | 大 | 中高（改主链路） | e2e 不退；P95 不恶化超过既定阈值 |
| **P4 校验层** | verifier + 定向重试 + 降级矩阵 | 中 | 低（可开关） | 故障注入集通过；误答率下降 |

**P1 单独拿出来说**：它改动最小、收益最大，而且不依赖任何其他阶段。

- history 进 classify → 多轮从"废"到"能用"，改的是一个入参和一段 prompt。
- 检索分数闸门 → 直接砍掉"自信地胡说"。
- `relaxed` 标注 → 消灭"放宽了条件但模型不知道"这种静默错误。

**P3 之前必须先有 P0**：没有 `Evidence` 结构，deep 模式的每步"够不够"没法判断，只能靠 LLM 自觉 —— 那就退化成现在管理端那样"模型说够了就够了"。

---

## 九、明确不做的事

写下来是为了防止重构失控。

1. **不把学生端改成全 ReAct。** 延迟和成本都会翻倍，而 90% 的问题不需要。分层才是答案。
2. **不引入规划用的第二个大模型 / 多智能体。** 现在的瓶颈不是"规划不够聪明"，是"没有状态、没有校验"。加角色只会加延迟。
3. **不让 Agent 自主写库。** 变更单 → 人工确认这条链路是产品的安全底线。
4. **不为"支持更多格式"而支持更多格式。** Excel/CSV 值得加（表格事实多），PPT/图片 OCR 优先级低 —— 除非评测显示真实缺口在那里。
5. **不追求"什么都能答"。** 边界外的问题明确说不知道，比答得像更像专业系统。

---

## 附录：与现有代码的对应关系

| 目标模块 | 现有落点 | 处理方式 |
|---|---|---|
| M1 会话与状态 | `app/agent/state.py` | 扩字段；新增 `memory.py` |
| M2 理解与分解 | `app/agent/router.py` | 演进为 `understanding.py`，入参加 history |
| M3 规划 | `app/agent/graph.py` 的 `_after_*` | 抽成 `planner.py`，输出 `mode` 而非边名 |
| M4 工具执行 | `app/agent/graph.py:query_data_node`、`admin_graph.py:_tools_node` | 抽成 `executor.py`，两处合并 |
| M5 校验 | 无 | 新增 `verifier.py` |
| M6 输出 | `app/agent/graph.py:compose_node` | 迁出为 `compose.py`，加 confidence/caveats |
| 检索 | `app/rag/retriever.py` | 加分数闸门、`chunk_index` 排序、relaxed 标注 |
| 工具 | `app/tools/*.py` | 加 `_meta` 信封、枚举规范化 |

---

## 附录二：实施进度

### 已完成（2026-10-23）

| 阶段 | 内容 | 落点 |
|---|---|---|
| **P1 理解层** | classify 接收 history；枚举参数规范化；检索分数闸门 + relaxed 标注 | `router.py`、`tools/common.py`、`retriever.py` |
| **P0 契约层** | `Evidence` 结构；`merge_evidence` 按 task_id 合并；Budget；state 新字段 | `agent/evidence.py`（新）、`state.py`、`graph.py` |
| **P4 最小版** | 引用编号 / 数字溯源 / 依据校验；结果转成 caveats 给前端 | `agent/verifier.py`（新）、`api/main.py`、`main.js` |

三处额外修掉的真问题：

1. `format_context` 同文件多 chunk 按相似度序拼接 → 改按 `chunk_index`，并标注断裂与截断。
2. fallback 放宽条件只写 trace、模型不知情 → 改写成 `notice` 拼进上下文。
3. Chroma 的相关度范围是 `[-1, 1]`（`1 - cosine距离`），越界会刷 `UserWarning` → 压掉，并在 `config.py` 记下实测分布。

### 评测对比（同一套 50 题）

| 指标 | 基线 10-19 | 现在 10-23 | 说明 |
|---|---|---|---|
| 路由准确率（router 层） | 94.0% | **98.0%** | 同版本两次运行结果完全一致，非抖动 |
| 召回 hit@5 | 100% | 100% | 改动未触及召回逻辑 |
| MRR | 0.9615 | 0.9615 | 同上 |
| e2e 路由准确率 | 96.0% | 98.0% | |
| e2e 事实通过率 | 93.2% | **95.5%** | +1 题（D10） |
| e2e 引用率 | 88.5% | 88.5% | |
| e2e p50 | 5008ms | 5180ms | 多了一次 verify 节点，本身不调 LLM |
| e2e p95 | 12644ms | 15418ms | n=50 时 p95 只由最后 2–3 个样本决定，不稳定 |

**必须说明的局限**：50 题、单次运行。98% 的 95% 置信区间大约 ±5.6 个百分点，所以"94% → 98%"不能当成普适结论，只能说「在本评测集上没有退化，且修掉了两个边界错误」。

### 一条被验证过的教训

尝试把「多轮要还原指代」写进 `ROUTER_SYSTEM`，单轮评测立刻翻动 3 题（2 条 hybrid 被压成 document，准确率掉到 92%）。最终改成**指令跟着数据走**：history 为空时 prompt 一个字都不加，多轮提示只出现在有历史的那次请求里。

效果实测（同一个问题「那报名截止呢」）：

```
无历史：route=structured  query="报名截止时间"                          college= 空
有历史：route=hybrid      query="2025年计算机学院全国大学生信息安全竞赛…报名截止时间"
                          college=计算机学院  keyword=信息安全竞赛
```

**prompt 里每加一个字都会改变输出分布**，能靠输入结构解决的，不要靠指令解决。

### 还没做

| 项 | 为什么先不做 |
|---|---|
| P2 记忆落库（checkpointer / session 表） | 涉及持久化，要单独验证；多轮目前靠前端传 history 已能跑通 |
| P3 planner 三档 + 统一工具集 ReAct | 改主链路，风险最高，必须先有多轮评测集兜底 |
| verifier 的自动重生成 | 需要预算控制和死循环防护，先让系统「能发现」比「能自愈」更划算 |
| 多轮评测集 / 故障注入集 | 现有 50 题全单轮，扩集是 P2/P3 的前置条件 |

### 会话持久化：做到哪一层（2026-10-23）

「能多轮」和「能持久」是两件事，这里把边界划清楚：

| 需求 | 现在 | 实现方式 |
|---|---|---|
| 同一页面内追问 | ✅ | 前端 `historyPayload()` 把最近 3 轮带进 `/api/chat` |
| **刷新页面后接着聊** | ✅ | `localStorage` 存 3 轮（问+答+引用），加载时重放 |
| 关掉浏览器再打开 | ✅ | 同上，只要没清缓存 |
| 换设备 / 换浏览器 | ❌ | 历史在**本机浏览器**，服务端无 session 记录 |
| Agent 跨会话记住你（偏好、身份） | ❌ | 没有长期记忆，每次会话都是陌生人 |

实现要点：

- 存的是 `{q, a, sources, caveats, elapsed}`，**不存 trace** —— 前端不该知道决策过程，存下来也没人看。
- `chatHistory` 这个可变数组被删掉了，历史由 `sessionTurns` **派生**（`historyPayload()`）。
  两份数据各自更新迟早不同步。
- 实时回答与「重放历史」共用 `renderAssistantBody()` 同一条渲染路径，
  否则两处各写一份 HTML，早晚出现「刷新后样式不一样」。
- 恢复的消息 foot 会标 `· 历史记录`，不让人以为那是刚刚发生的。

**顺手修掉一个真 bug**：点「清空对话」时如果还有请求在途，回答回来后会把新一轮
写进已经清空的会话里 —— 用户看到的是「清了又自己冒出来」。现在用 `AbortController`
掐掉在途请求，`AbortError` 单独处理，不画「回答失败」。

隐私边界要跟用户说清楚：问答记录留在这台机器的浏览器里，服务端不留存。


---

## 附录三：多进程部署（2026-10-23）

### 问题

`uvicorn --workers 4` 直接开会得到一个很难查的 bug：管理员上传成功 → 学生搜不到 →
重启服务又好了。

根因不是"性能不够"，是 Chroma 嵌入式模式的索引在**进程内存里**，是启动时的快照。
四个 worker 各持一份，彼此看不见对方的写入。所以问题不是"要不要多进程"，
而是"向量库必须先搬出进程"。

### 方案

不换向量库，改**连接方式**：Chroma 自己就有 client/server 模式。

```bash
# 1. 向量库服务（常驻，独占 var/chroma 目录）
chroma run --path var/chroma --host 127.0.0.1 --port 8001 --allow-reset

# 2. 多 worker（每条命令一个终端）
CHROMA_MODE=server python scripts/serve.py --workers 4

# 3. 单进程（默认，行为和改之前完全一样）
python scripts/serve.py
```

`app/rag/vectorstore.py` 的 `get_vectorstore()` 按 `CHROMA_MODE` 二选一：
`server` 走 `chromadb.HttpClient`，`embedded` 走原来的 `persist_directory`。
**业务代码零改动** —— 这就是当初把向量操作全收在 175 行里的回报。

### 四处配套改动

| 改动 | 落点 | 不改会怎样 |
|---|---|---|
| 模式开关 | `config.py` `CHROMA_MODE` / `CHROMA_HOST` / `CHROMA_PORT` | — |
| 向量库连接 | `vectorstore.py:get_vectorstore()` | 多 worker 各持一份索引，写入互不可见 |
| 日志按 pid 分文件 | `logging_config.py` | Windows 上午夜轮转时多进程 rename 同一文件 → `PermissionError`，日志中断 |
| 连接池按 worker 算额度 | `db.py:_pool_config()` | 8 worker 时 `10+5` × 8 = 120，逼近 `max_connections=151` |
| 启动防呆 | `scripts/serve.py` | embedded 模式下开 4 worker **必须在启动时就被拦住**，不能等线上出 bug |

启动脚本还会在 server 模式下先探一次 `heartbeat`。连不上就拒绝启动 —— 否则 worker
起来后第一次检索才失败，错误被包在工具返回里，看起来像「知识库空了」。

### 实测证据

| 验证项 | 结果 |
|---|---|
| 跨进程可见性 | 进程 41328 写入 38→39；进程 16468（不同 PID）立刻 count=39，query 命中新数据，距离 **0.0** |
| 4 worker 分发 | 4 个日志文件（p13736 / p31364 / p33920 / p34196），12 次请求分别落到 4 个进程 |
| worker 实际后端 | 日志确认 `向量库：server 模式 127.0.0.1:8001`（p31364、p33920 各一条） |
| **检索语义不变** | embedded 与 server 的 top1 文件相同、`max_score` **都是 0.6588**、hits 顺序一致 |
| **入库即时可见** | 独立进程写入 `probe/live-test.md` → 4 worker 服务**立刻**搜到，score **0.8703**，排名第一，未重启 |
| 单进程回归 | 默认 embedded 模式启动正常，top1 正确（0.6572） |

**最关键的一条是"检索语义不变"**：Chroma 的 server 与 embedded 用同一套 cosine 计算，
所以 `MIN_RELEVANCE_SCORE = 0.2` **不需要重新标定**。

⚠️ 这条结论**不能推广到换别的向量库**。换成 Qdrant / Milvus 时分数分布不保证一致，
必须重跑标定 —— 附录二里说的「换库要重标定」依然成立。

### 一个 Windows 特有的坑

`uvicorn.run()` 要传**导入字符串** `"app.api.main:app"`，不能传 app 对象，否则
`workers` 参数被忽略、静默退化成单进程。

另外 `WEB_WORKERS` 环境变量必须在调用 `uvicorn.run()` **之前**设置：每个 worker 是
spawn 出来的独立进程，会重新 import `config` 并读这个变量。设晚了子进程拿到的还是
旧值，结果是日志不分文件、连接池按 1 个 worker 算额度。

### 还没解决的

| 项 | 说明 |
|---|---|
| Chroma 服务是单点 | 它挂了所有 worker 一起挂。要真正高可用得上 Qdrant 集群，或至少给 chroma 配进程守护 + 重启 |
| 入库仍然同步 | 多 worker 解决的是「同时有多少人在问」，不解决「传一份大 PDF 要等 30 秒」。异步入库是另一件事 |
| `/api/admin/*` 仍无认证 | 多进程不改变这一点。现在仍然只能 127.0.0.1 用 |
| 没有优雅重启 | `uvicorn` 重启 worker 会中断在途请求，因为 `/api/chat` 是同步长任务（5 秒）。要支持滚动发布得先改成异步 + 可中断 |
