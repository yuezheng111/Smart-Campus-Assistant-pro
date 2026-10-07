-- =============================================================
-- 校园智能助手 · 结构化数据库
-- MySQL 5.7+ / 字符集 utf8mb4
-- =============================================================

CREATE DATABASE IF NOT EXISTS `school_agent`
    DEFAULT CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE `school_agent`;

-- -------------------------------------------------------------
-- 1. 学校文件（非结构化知识的「档案卡」）
--    RAG 负责找回原文，这张表负责回答「谁、什么时候、哪个部门发的」
-- -------------------------------------------------------------
DROP TABLE IF EXISTS `document`;
CREATE TABLE `document` (
    `id`           INT AUTO_INCREMENT PRIMARY KEY,
    `doc_no`       VARCHAR(64)  DEFAULT NULL COMMENT '发文字号',
    `title`        VARCHAR(255) NOT NULL    COMMENT '文件标题',
    `department`   VARCHAR(64)  DEFAULT NULL COMMENT '发布部门',
    `doc_type`     VARCHAR(32)  DEFAULT NULL COMMENT '文件类型：奖学金/考试/竞赛/教学/学籍...',
    `college`      VARCHAR(64)  DEFAULT NULL COMMENT '适用学院，全校填「全校」',
    `publish_date` DATE         DEFAULT NULL COMMENT '发布日期',
    `year`         INT          DEFAULT NULL COMMENT '发布年份',
    `file_path`    VARCHAR(512) DEFAULT NULL COMMENT '本地文件相对路径',
    `url`          VARCHAR(512) DEFAULT NULL COMMENT '原文链接',
    `summary`      TEXT                     COMMENT '文件摘要',
    `content_hash` CHAR(64)     DEFAULT NULL COMMENT '文件字节的 sha256，用于判断内容是否变过',

    -- ---- 生命周期 · 内容侧 --------------------------------------------------
    -- 这一组描述「这份文件本身是什么」，一次确定、之后不变，
    -- 所以它可以跟着文件走：全量重建时从文件（或侧车）重新读出来覆盖。
    `doc_family`     VARCHAR(64) DEFAULT NULL COMMENT '文档族：同一主题的政策归一族，用于版本替代',
    `version`        INT         DEFAULT NULL COMMENT '族内版本号',
    `effective_from` DATE        DEFAULT NULL COMMENT '生效日期',
    `effective_to`   DATE        DEFAULT NULL COMMENT '失效日期',

    -- ---- 生命周期 · 状态侧 --------------------------------------------------
    -- 这一组描述「这份文件现在还有效吗」，是运行时会变的状态。
    -- 它只以本表为正本：侧车里不写、重建时也不覆盖，
    -- 否则跑一次全量重建就会把「已废止」冲回「现行」。
    --
    -- 唯一的例外是下面的 file_missing_at：它同样属于状态侧，但重建时**不回填**，
    -- 而是直接置回 NULL —— 因为「这份文件正在被处理」这件事本身，
    -- 就等于「原件好好地躺在 data/raw 里」，没有比这更可靠的证据。
    `status`         VARCHAR(16) NOT NULL DEFAULT 'active'
                   COMMENT 'active 现行 / superseded 已废止',
    `superseded_by`  INT         DEFAULT NULL COMMENT '被哪份文件替代（指向本表 id）',
    `superseded_at`  DATE        DEFAULT NULL COMMENT '执行废止操作的日期',

    -- 原件存在性：和 status 是**两个正交的维度**，所以单独一个字段而不是
    -- 往 status 里再塞一个取值。理由是「已废止」和「原件被删了」可能同时成立，
    -- 共用一个枚举就会互相覆盖 —— 一份 superseded 的文件原件消失后再恢复，
    -- 就分不清它原本是废止的还是现行的了。
    `file_missing_at` DATE       DEFAULT NULL
                   COMMENT '原件从 data/raw 消失的日期；NULL 表示原件仍在',

    `created_at`   TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    KEY `idx_doc_type`  (`doc_type`),
    KEY `idx_college`   (`college`),
    KEY `idx_year`      (`year`),
    KEY `idx_publish`   (`publish_date`),
    KEY `idx_title`     (`title`),
    KEY `idx_file_path` (`file_path`(191)),
    KEY `idx_hash`      (`content_hash`),
    KEY `idx_status`    (`status`),
    KEY `idx_family`    (`doc_family`),
    KEY `idx_missing`   (`file_missing_at`)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT = '学校文件元数据';

-- -------------------------------------------------------------
-- 2. 学科竞赛（结构化）
--    这类信息字段固定、需要按年份/类别/年级聚合统计，放数据库最合适
-- -------------------------------------------------------------
DROP TABLE IF EXISTS `competition`;
CREATE TABLE `competition` (
    `id`            INT AUTO_INCREMENT PRIMARY KEY,
    `name`          VARCHAR(128) NOT NULL COMMENT '竞赛名称',
    `category`      VARCHAR(32)  DEFAULT NULL COMMENT '类别：计算机/数学/英语/电子/创新创业/综合',
    `level`         VARCHAR(32)  DEFAULT NULL COMMENT '级别：国家级/省级/校级',
    `organizer`     VARCHAR(128) DEFAULT NULL COMMENT '主办单位',
    `year`          INT          DEFAULT NULL COMMENT '届次年份',
    `signup_start`  DATE         DEFAULT NULL COMMENT '报名开始',
    `signup_end`    DATE         DEFAULT NULL COMMENT '报名截止',
    `contest_start` DATE         DEFAULT NULL COMMENT '比赛开始',
    `contest_end`   DATE         DEFAULT NULL COMMENT '比赛结束',
    `target_grade`  VARCHAR(64)  DEFAULT NULL COMMENT '面向年级',
    `target_major`  VARCHAR(128) DEFAULT NULL COMMENT '面向专业',
    `official_url`  VARCHAR(255) DEFAULT NULL,
    `description`   TEXT                    COMMENT '简介 / 参赛要求',
    KEY `idx_cat`   (`category`),
    KEY `idx_year`  (`year`),
    KEY `idx_level` (`level`),
    KEY `idx_name`  (`name`)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT = '学科竞赛信息';

-- -------------------------------------------------------------
-- 3. 考试信息（结构化）
-- -------------------------------------------------------------
DROP TABLE IF EXISTS `exam`;
CREATE TABLE `exam` (
    `id`           INT AUTO_INCREMENT PRIMARY KEY,
    `name`         VARCHAR(128) NOT NULL COMMENT '考试名称',
    `category`     VARCHAR(32)  DEFAULT NULL COMMENT '类别：英语/计算机/职业资格/学业/其他',
    `year`         INT          DEFAULT NULL,
    `signup_start` DATE         DEFAULT NULL,
    `signup_end`   DATE         DEFAULT NULL,
    `exam_date`    DATE         DEFAULT NULL COMMENT '考试日期（起始）',
    `fee`          VARCHAR(64)  DEFAULT NULL COMMENT '报名费',
    `target_grade` VARCHAR(64)  DEFAULT NULL,
    `remark`       TEXT                    COMMENT '备注',
    KEY `idx_cat`  (`category`),
    KEY `idx_year` (`year`),
    KEY `idx_name` (`name`)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT = '考试信息';

-- -------------------------------------------------------------
-- 4. 学生画像（用于第三阶段「学业规划」）
-- -------------------------------------------------------------
DROP TABLE IF EXISTS `student`;
CREATE TABLE `student` (
    `id`         INT AUTO_INCREMENT PRIMARY KEY,
    `student_no` VARCHAR(32) NOT NULL UNIQUE COMMENT '学号',
    `name`       VARCHAR(32) DEFAULT NULL,
    `grade`      VARCHAR(16) DEFAULT NULL COMMENT '年级，如 大二',
    `major`      VARCHAR(64) DEFAULT NULL COMMENT '专业',
    `college`    VARCHAR(64) DEFAULT NULL,
    `interests`  VARCHAR(255) DEFAULT NULL COMMENT '兴趣方向，逗号分隔',
    `gpa`        DECIMAL(3, 2) DEFAULT NULL
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT = '学生画像（演示用）';
