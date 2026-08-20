#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
migrate_all_to_cloud.py
将项目下所有本地 SQLite 数据库(应用主库 + 各数据处理库)迁移到云 MySQL。

设计:
- 每个源 SQLite 文件 -> 云 MySQL 中的一个独立 schema(按来源命名,避免表名冲突)。
- 应用主库 data/recognition.db -> 落到 MYSQL_DB(anime_role_detect) schema,与应用读取位置一致。
- 抽取原表结构(列/类型/约束/索引)重建为 MySQL DDL;逐表 bulk 拷贝数据;行数核对。
- 安全策略:
    * 默认 --dry-run:把数据写入本地 SQLite 暂存库(表名加 schema 前缀),验证抽取+拷贝+核对逻辑。
    * --print-sql:仅打印/导出 MySQL DDL,不连库、不写数据。
    * --apply:真正连云 MySQL 执行(需配好 .env 且本机可连)。
- 幂等:CREATE SCHEMA/TABLE 均 IF NOT EXISTS;数据拷贝前会先清空目标表(便于重跑)。

用法(在用户 Mac 本机执行,需 .venv 且能连云 MySQL):
    .venv/bin/python3 scripts/migrate_all_to_cloud.py --dry-run          # 本地验证逻辑
    .venv/bin/python3 scripts/migrate_all_to_cloud.py --print-sql         # 预览 DDL
    .venv/bin/python3 scripts/migrate_all_to_cloud.py --apply             # 真正推上云
    .venv/bin/python3 scripts/migrate_all_to_cloud.py --apply --src data/auth.db   # 只迁某个库
"""
import os
import sys
import argparse
import sqlite3
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_MAIN_DB = (PROJECT_ROOT / "data" / "recognition.db").resolve()

# ---- 源库清单(相对项目根) ----
SOURCE_RELS = [
    "data/recognition.db",            # 应用主库 -> MYSQL_DB
    "data/auth.db",
    "data/data_pipeline.db",
    "data/database/data_pipeline.db",
    "data/database/recognition.db",
    "data/database/role_images.db",
    "data/database/spider_records.db",
    "data/packages.db",
    "data/image_hashes.db",
    "data/collection.db",
    "data/role_stats.db",
    "recognition.db",
    "scripts/skillhub/data/ardc.db",
    "scripts/skillhub/test.db",
    "scripts/skillhub/ardc/api/test.db",
]


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass


def mysql_url_from_env() -> str:
    host = os.environ.get("MYSQL_HOST", "")
    port = os.environ.get("MYSQL_PORT", "3306")
    user = os.environ.get("MYSQL_USER", "")
    pw = os.environ.get("MYSQL_PASSWORD", "")
    db = os.environ.get("MYSQL_DB", "anime_role_detect")
    if not (host and user and pw):
        raise RuntimeError("MYSQL_* 环境变量未配置(读不到 .env 或 .env 缺失)")
    from urllib.parse import quote_plus
    return f"mysql+pymysql://{user}:{quote_plus(pw)}@{host}:{port}/{db}?charset=utf8mb4"


def schema_for(abs_path: Path) -> str:
    """返回目标 MySQL schema 名。应用主库落到 MYSQL_DB。"""
    if abs_path.resolve() == APP_MAIN_DB:
        return os.environ.get("MYSQL_DB", "anime_role_detect")
    rel = abs_path.relative_to(PROJECT_ROOT).as_posix()
    # data/recognition.db -> data_recognition; scripts/skillhub/test.db -> scripts_skillhub_test
    name = re.sub(r"[^a-zA-Z0-9]+", "_", rel)
    name = name.strip("_").lower()
    if not name:
        name = "migrated"
    return f"{name}_db" if not name.endswith("_db") else name


# ---------------- SQLite schema 抽取 ----------------
def extract_tables(sqlite_path: Path):
    con = sqlite3.connect(str(sqlite_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT name, type FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = []
    for row in cur.fetchall():
        tname = row["name"]
        if tname == "sqlite_sequence":
            continue
        info = cur.execute(f'PRAGMA table_info("{tname}")').fetchall()
        cols = []
        pk_cols = []
        for c in info:
            cols.append({
                "name": c["name"],
                "type": (c["type"] or "").upper(),
                "notnull": int(c["notnull"] or 0),
                "dflt": c["dflt_value"],
                "pk": int(c["pk"] or 0),
            })
            if int(c["pk"] or 0):
                pk_cols.append(c["name"])
        idxs = cur.execute(f'PRAGMA index_list("{tname}")').fetchall()
        indexes = []
        for ix in idxs:
            ix_name = ix["name"]
            if ix_name.startswith("sqlite_autoindex"):
                continue
            iinfo = cur.execute(f'PRAGMA index_info("{ix_name}")').fetchall()
            indexes.append({
                "name": ix_name,
                "unique": int(ix["unique"] or 0),
                "columns": [i["name"] for i in iinfo],
            })
        tables.append({"name": tname, "columns": cols, "pk": pk_cols, "indexes": indexes})
    con.close()
    return tables


# ---------------- 类型映射 ----------------
def mysql_type(decl: str) -> str:
    d = (decl or "").strip().upper()
    if d == "":
        return "LONGTEXT"
    # 带长度/精度,如 VARCHAR(255), DECIMAL(10,2)
    m = re.match(r"^([A-Z ]+?)\s*(\(.*\))?$", d)
    base = m.group(1).strip() if m else d
    arg = m.group(2) or "" if m else ""
    mapping_int = {"INTEGER", "INT", "BIGINT", "SMALLINT", "TINYINT", "MEDIUMINT", "UNSIGNED BIG INT", "INT2", "INT8"}
    if base in mapping_int:
        if base in ("BIGINT", "INT8", "UNSIGNED BIG INT"):
            return "BIGINT" + arg
        if base in ("SMALLINT", "TINYINT", "INT2"):
            return "SMALLINT"
        return "INT" + arg
    if base in ("REAL", "FLOAT", "DOUBLE", "DOUBLE PRECISION"):
        return "DOUBLE"
    if base in ("NUMERIC", "DECIMAL"):
        return ("DECIMAL" + arg) if arg else "DECIMAL(20,6)"
    if base in ("BOOLEAN", "BOOL"):
        return "TINYINT(1)"
    if base in ("DATE", "DATETIME", "TIMESTAMP"):
        return "DATETIME"
    if base in ("BLOB", "BINARY", "VARBINARY"):
        return "LONGBLOB"
    if base in ("TEXT", "CLOB", "STRING"):
        return "LONGTEXT"
    if base in ("VARCHAR", "NVARCHAR", "CHAR"):
        # MySQL 要求 VARCHAR/CHAR 必须带长度
        return base + (arg if arg else "(255)")
    # 未知类型,退化为 LONGTEXT 以保住数据
    return "LONGTEXT"


def is_int_type(decl: str) -> bool:
    d = (decl or "").strip().upper()
    return any(d.startswith(p) for p in ("INTEGER", "INT", "BIGINT", "SMALLINT", "TINYINT", "MEDIUMINT", "INT2", "INT8", "UNSIGNED"))


# MySQL 不允许给 TEXT/BLOB/JSON/GEOMETRY 家族列设置字面 DEFAULT 值
_NO_DEFAULT_TYPES = {"TEXT", "TINYTEXT", "MEDIUMTEXT", "LONGTEXT",
                     "BLOB", "TINYBLOB", "MEDIUMBLOB", "LONGBLOB", "JSON", "GEOMETRY"}

# 这些类型在 PRIMARY KEY / 索引中必须指定键长(如 `col(255)`)
_KEY_LEN_TYPES = {"TEXT", "TINYTEXT", "MEDIUMTEXT", "LONGTEXT",
                  "BLOB", "TINYBLOB", "MEDIUMBLOB", "LONGBLOB", "JSON", "GEOMETRY"}


def _can_have_default(mysql_typ: str) -> bool:
    base = mysql_typ.split("(")[0].strip().upper()
    return base not in _NO_DEFAULT_TYPES


def _needs_key_len(mysql_typ: str) -> bool:
    base = mysql_typ.split("(")[0].strip().upper()
    return base in _KEY_LEN_TYPES


def _col_type_map(table) -> dict:
    return {c["name"]: mysql_type(c["type"]) for c in table["columns"]}


def fmt_default(dflt):
    """把 SQLite 的 dflt_value 转成 MySQL 可用的 DEFAULT 字面量;无法安全表达则返回 None。"""
    if dflt is None:
        return None
    s = str(dflt).strip()
    # 函数表达式 / 当前时间函数,跳过(数据拷贝会写入真实值,仅影响未来插入默认值)
    if s.startswith("(") or "CURRENT_TIMESTAMP" in s.upper() or "DATETIME(" in s.upper() or "NOW(" in s.upper():
        return None
    # 数字
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return s
    # 引号字符串:去掉包裹引号
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        inner = s[1:-1].replace("''", "'")
        return "'" + inner.replace("'", "''") + "'"
    # 裸字符串(无引号)也按字符串处理
    return "'" + s.replace("'", "''") + "'"


def build_create(table, schema, dest_kind):
    """生成 CREATE TABLE DDL。dest_kind='mysql' 用 schema 限定的反引号;'sqlite' 用前缀表名。"""
    tname = table["name"]
    if dest_kind == "mysql":
        full = f"`{schema}`.`{tname}`"
    else:
        full = f"`{schema}__{tname}`"
    col_defs = []
    auto_col = None
    for c in table["columns"]:
        parts = [f"`{c['name']}`", mysql_type(c["type"])]
        if c["pk"] and len(table["pk"]) == 1 and is_int_type(c["type"]):
            if dest_kind == "mysql":
                parts.append("NOT NULL")
                parts.append("AUTO_INCREMENT")
                # MySQL 要求 AUTO_INCREMENT 列必须是键,内联 PRIMARY KEY
                parts.append("PRIMARY KEY")
            else:
                # SQLite 仅在类型为 INTEGER 且作为 PRIMARY KEY 时自动自增
                parts = [f"`{c['name']}`", "INTEGER NOT NULL PRIMARY KEY"]
            auto_col = c["name"]
        else:
            if c["notnull"]:
                parts.append("NOT NULL")
            d = fmt_default(c["dflt"])
            if d is not None:
                # MySQL 不允许 TEXT/BLOB 家族列设置字面 DEFAULT,跳过(真实数据值仍会写入)
                if dest_kind == "mysql" and not _can_have_default(mysql_type(c["type"])):
                    pass
                else:
                    parts.append(f"DEFAULT {d}")
        col_defs.append("  " + " ".join(parts))
    if table["pk"] and not auto_col:
        ctm = _col_type_map(table)
        pk_parts = []
        for p in table["pk"]:
            if _needs_key_len(ctm.get(p, "")):
                pk_parts.append(f"`{p}`(255)")
            else:
                pk_parts.append(f"`{p}`")
        pk = ", ".join(pk_parts)
        col_defs.append(f"  PRIMARY KEY ({pk})")
    col_defs_str = ",\n".join(col_defs)
    if dest_kind == "mysql":
        return f"CREATE TABLE IF NOT EXISTS {full} (\n{col_defs_str}\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
    else:
        return f"CREATE TABLE IF NOT EXISTS {full} (\n{col_defs_str}\n);"


def build_indexes(table, schema, dest_kind):
    out = []
    tname = table["name"]
    if dest_kind == "mysql":
        full = f"`{schema}`.`{tname}`"
        tref = full
    else:
        full = f"`{schema}__{tname}`"
        tref = full
    ctm = _col_type_map(table)
    for ix in table["indexes"]:
        col_parts = []
        for c in ix["columns"]:
            if _needs_key_len(ctm.get(c, "")):
                col_parts.append(f"`{c}`(255)")
            else:
                col_parts.append(f"`{c}`")
        cols = ", ".join(col_parts)
        uniq = "UNIQUE " if ix["unique"] else ""
        idx_name = f"ix_{schema}_{tname}_{ix['name']}" if dest_kind == "mysql" else f"ix_{schema}__{tname}_{ix['name']}"
        idx_name = idx_name[:64]  # MySQL 索引名最长 64 字符
        # MySQL 不支持 CREATE INDEX IF NOT EXISTS,SQLite 支持
        if_exists = "" if dest_kind == "mysql" else "IF NOT EXISTS "
        out.append({"name": idx_name, "sql": f"CREATE {uniq}INDEX {if_exists}`{idx_name}` ON {tref} ({cols});"})
    return out


# ---------------- 数据源读取 ----------------
def read_rows(sqlite_path: Path, table: str):
    con = sqlite3.connect(str(sqlite_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(f'SELECT * FROM "{table}"')
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]


# ---------------- 写入目标 ----------------
def migrate_one(src_path, schema, dest_engine, dest_kind, write, print_sql, manifest):
    tables = extract_tables(src_path)
    result = {"source": str(src_path), "schema": schema, "tables": []}
    for t in tables:
        ddl = build_create(t, schema, dest_kind)
        idx_ddl = build_indexes(t, schema, dest_kind)
        if print_sql:
            print("-- table:", t["name"], "->", schema)
            print(ddl + "\n")
            for ix in idx_ddl:
                print(ix["sql"])
            print()
        table_rec = {"name": t["name"], "columns": len(t["columns"]), "src_rows": 0, "dst_rows": 0, "status": "pending"}
        try:
            if write:
                with dest_engine.begin() as conn:
                    from sqlalchemy import text
                    if dest_kind == "mysql":
                        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS `{schema}` DEFAULT CHARACTER SET utf8mb4"))
                    conn.execute(text(ddl))
                    for ix in idx_ddl:
                        try:
                            if dest_kind == "mysql":
                                exists = conn.execute(text(
                                    "SELECT 1 FROM information_schema.STATISTICS "
                                    "WHERE TABLE_SCHEMA=:s AND TABLE_NAME=:t AND INDEX_NAME=:i"
                                ), {"s": schema, "t": t["name"], "i": ix["name"]}).scalar()
                                if exists:
                                    continue
                            conn.execute(text(ix["sql"]))
                        except Exception as e:
                            print(f"  [warn] index skip {t['name']}.{ix['name']}: {e}")
                # 清空目标表以便重跑
                dest_table = f"`{schema}`.`{t['name']}`" if dest_kind == "mysql" else f"`{schema}__{t['name']}`"
                with dest_engine.begin() as conn:
                    from sqlalchemy import text
                    conn.execute(text(f"DELETE FROM {dest_table}"))
                # 拷贝数据
                rows = read_rows(src_path, t["name"])
                table_rec["src_rows"] = len(rows)
                if rows:
                    cols = list(rows[0].keys())
                    col_sql = ", ".join(f"`{c}`" for c in cols)
                    placeholders = ", ".join(f":{c}" for c in cols)
                    insert_sql = f"INSERT INTO {dest_table} ({col_sql}) VALUES ({placeholders})"
                    with dest_engine.begin() as conn:
                        from sqlalchemy import text
                        conn.execute(text(insert_sql), rows)
                    # 修复 AUTO_INCREMENT 计数器
                    if dest_kind == "mysql" and any(c["pk"] and is_int_type(c["type"]) and len(t["pk"]) == 1 for c in t["columns"]):
                        try:
                            with dest_engine.begin() as conn:
                                from sqlalchemy import text
                                conn.execute(text(f"ALTER TABLE {dest_table} AUTO_INCREMENT = (SELECT COALESCE(MAX(`{t['pk'][0]}`),0)+1 FROM {dest_table})"))
                        except Exception:
                            pass
                with dest_engine.begin() as conn:
                    from sqlalchemy import text
                    cnt = conn.execute(text(f"SELECT COUNT(*) FROM {dest_table}")).scalar()
                    table_rec["dst_rows"] = int(cnt)
                table_rec["status"] = "ok" if table_rec["src_rows"] == table_rec["dst_rows"] else "MISMATCH"
            else:
                table_rec["src_rows"] = len(read_rows(src_path, t["name"]))
                table_rec["status"] = "dry-skipped"
        except Exception as e:
            table_rec["status"] = "error"
            table_rec["error"] = str(e)[:300]
            print(f"  [ERROR] {t['name']}: {e}")
        result["tables"].append(table_rec)
    manifest.append(result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正连云 MySQL 执行(默认仅 dry-run)")
    ap.add_argument("--print-sql", action="store_true", help="仅打印 MySQL DDL,不连库")
    ap.add_argument("--src", nargs="*", help="只处理指定相对路径(如 data/auth.db)")
    ap.add_argument("--staging", default="/tmp/migrate_staging.db", help="dry-run 暂存 SQLite 路径")
    args = ap.parse_args()

    load_env()
    if not args.print_sql and not args.apply:
        print("[mode] DRY-RUN(本地 SQLite 暂存验证)。加 --apply 推云,--print-sql 仅预览 DDL。")

    if args.src:
        rels = [s for s in SOURCE_RELS if s in args.src or any(s == x for x in args.src)]
    else:
        rels = SOURCE_RELS

    manifest = []
    if args.apply:
        from sqlalchemy import create_engine
        url = mysql_url_from_env()
        dest_engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 15})
        dest_kind = "mysql"
        write = True
        print(f"[connect] 云 MySQL: {url.split('@')[-1]}")
    elif args.print_sql:
        dest_engine = None
        dest_kind = "mysql"
        write = False
        print("[mode] PRINT-SQL(仅预览 DDL,不连库)")
    else:
        from sqlalchemy import create_engine
        if os.path.exists(args.staging):
            os.remove(args.staging)
        dest_engine = create_engine(f"sqlite:///{args.staging}")
        dest_kind = "sqlite"
        write = True
        print(f"[staging] {args.staging} (DRY-RUN 真实写入并核对)")

    if args.print_sql:
        ddl_path = PROJECT_ROOT / "deliverables" / "migrate_ddl_preview.sql"
        ddl_path.parent.mkdir(exist_ok=True)
        with open(ddl_path, "w", encoding="utf-8") as f:
            old_stdout = sys.stdout
            sys.stdout = f
            for rel in rels:
                p = (PROJECT_ROOT / rel)
                if not p.exists():
                    continue
                schema = schema_for(p.resolve())
                migrate_one(p.resolve(), schema, dest_engine, dest_kind, write, True, manifest)
            sys.stdout = old_stdout
        print(f"[ddl] DDL 预览已写入 {ddl_path}")

    for rel in rels:
        p = (PROJECT_ROOT / rel)
        if not p.exists():
            print(f"[skip] 不存在: {rel}")
            continue
        if args.print_sql:
            # DDL 已在上方 ddl 块打印过,这里仅汇总
            schema = schema_for(p.resolve())
            res = next((m for m in manifest if m["source"] == str(p.resolve())), None)
            ntab = len(res["tables"]) if res else 0
            print(f"[SQL] {rel} -> schema `{schema}`  tables={ntab}")
            continue
        schema = schema_for(p.resolve())
        try:
            res = migrate_one(p.resolve(), schema, dest_engine, dest_kind, write, args.print_sql, manifest)
        except Exception as e:
            print(f"[ERROR] 源库处理失败 {rel}: {e}")
            manifest.append({"source": str(p.resolve()), "schema": schema, "tables": [], "error": str(e)[:300]})
            continue
        total_src = sum(t["src_rows"] for t in res["tables"])
        total_dst = sum(t["dst_rows"] for t in res["tables"])
        tag = "APPLY" if args.apply else "DRY"
        n_err = sum(1 for t in res["tables"] if t["status"] in ("error", "MISMATCH"))
        print(f"[{tag}] {rel} -> schema `{schema}`  tables={len(res['tables'])} src_rows={total_src} dst_rows={total_dst}" + (f"  ⚠️{n_err}个问题" if n_err else ""))

    out_path = PROJECT_ROOT / "deliverables" / "migrate_manifest.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[manifest] {out_path}")


if __name__ == "__main__":
    main()
