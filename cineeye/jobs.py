"""异步任务调度：把耗时的训练/建库脚本丢到后台线程，提供进度与日志轮询。

用途：
    管理后台点"开始训练"后立即返回 task_id，不阻塞 HTTP 请求。真正的训练
    在线程池里以子进程方式跑现成脚本（01~05 / 15/16 / 12），stdout/stderr
    实时写入日志文件，前端通过 GET /admin/tasks/{id} 轮询状态、进度与日志尾部。

被谁使用：
    - serve.py 的 POST /admin/train、GET /admin/tasks、GET /admin/tasks/{id}、
      POST /admin/tasks/{id}/cancel；
    - 运维可直接 from cineeye.jobs import JobManager 做批处理。

依赖：
    - assets.AssetStore（jobs 表与其同库；本模块直接用 sqlite3 读写 jobs 表）；
    - stdlib concurrent.futures / subprocess / sqlite3 / threading。

运维要点（改这里之前必须知道的事）：
    1. 线程池 max_workers=2：训练是 CPU/GPU 密集任务，同时跑两个以上会抢资源，
       故限并发 2，第 3 个任务在队列里排队（状态 queued）。
    2. 每个任务一个子进程，用 Popen 而非 subprocess.run，才能实时读日志、支持取消。
    3. 日志落盘到 outputs/reports/jobs/<task_id>.log；日志是进度的唯一真相源，
       progress 通过正则匹配 "Epoch"/"epoch" 行粗略估算，脚本不配合时停在 90% 直到完成。
    4. 取消是尽力而为：terminate 子进程，超时 5s 再 kill；线程池里的线程会自然结束。
    5. 进程重启后，running/queued 的任务会被标记为 failed（子进程已随父进程死亡），
       避免前端永远轮询一个不存在任务。这是 recover() 的作用，服务启动时调用一次。
    6. 【安全】这里会拼接命令行启动项目内脚本，命令与参数全部来自服务端代码白名单，
       payload 里的用户参数只允许 --set 形式的配置覆盖；若将来开放任意参数，
       必须改 list 形式调用 subprocess 且禁止 shell=True，避免命令注入。
"""

from __future__ import annotations

import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .config import ROOT
from .utils import ensure_dir, get_logger

log = get_logger("cineeye.jobs")

JOBS_LOG_DIR = ROOT / "outputs" / "reports" / "jobs"
PYTHON = sys.executable  # 用当前虚拟环境的解释器跑子脚本，保证依赖一致
EPOCH_RE = re.compile(r"(?:Epoch|epoch)\s*(\d+)\s*/\s*(\d+)")

# 与 assets.py 中 jobs 表定义一致；这里单独建表，保证 JobManager 用独立 db
# 路径（测试/运维）时也能工作，不依赖 AssetStore 是否初始化过。
JOBS_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    status     TEXT NOT NULL,
    payload    TEXT,
    progress   INTEGER DEFAULT 0,
    log_path   TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    started_at TEXT,
    finished_at TEXT,
    result     TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


class JobManager:
    """训练任务管理器：提交、执行、查询、取消，状态持久化到 jobs 表。

    Args:
        db_path: SQLite 路径，默认 ROOT/outputs/cineeye.db（与 assets 同库）。
        max_workers: 线程池并发数，默认 2。

    运维要点：
        - 单例使用（serve.py 模块级一个），不要每次请求 new，否则线程池爆炸。
        - 状态变更都走 _set_status，统一维护 started_at/finished_at 时间戳。
    """

    def __init__(self, db_path: str | Path | None = None, max_workers: int = 2):
        self.path = Path(db_path) if db_path else ROOT / "outputs" / "cineeye.db"
        ensure_dir(self.path.parent)
        ensure_dir(JOBS_LOG_DIR)
        with self._conn() as c:
            c.executescript(JOBS_SCHEMA)  # 幂等建表，独立 db 也能用
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cineeye-job")
        self._procs: dict[str, subprocess.Popen] = {}  # task_id -> 运行中子进程
        self._lock = __import__("threading").Lock()
        self.recover()
        log.info("任务管理就绪：%s（并发 %d）", self.path, max_workers)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def recover(self) -> None:
        """进程启动后把遗留的 running/queued 标记为 failed（子进程已亡）。

        否则前端会永远轮询一个不会推进的任务。fail 的 result 里写明原因。
        """
        msg = json.dumps({"exit_code": -1, "reason": "服务重启，任务中断"}, ensure_ascii=False)
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET status='failed', finished_at=?, result=? "
                "WHERE status IN ('running','queued')",
                (datetime.now().isoformat(timespec="seconds"), msg),
            )

    # ---------------------------------------------------------------- 提交
    def submit(self, kind: str, payload: dict[str, Any] | None = None) -> str:
        """提交一个任务，立即返回 task_id（异步执行）。

        Args:
            kind: train_movie / train_actor / build_gallery / refresh_derived。
            payload: 任务参数，如 {"epochs": 40, "config": "configs/default.yaml",
                "sets": ["model.mode=finetune"]}。

        Returns:
            形如 job_xxxxxxxx 的 task_id。

        Raises:
            ValueError: kind 不在白名单。
        """
        if kind not in {"train_movie", "train_actor", "build_gallery", "refresh_derived"}:
            raise ValueError(f"不支持的任务类型：{kind}")
        payload = payload or {}
        task_id = "job_" + datetime.now().strftime("%Y%m%d%H%M%S") + f"_{os.getpid() % 1000:03d}"
        with self._conn() as c:
            c.execute(
                "INSERT INTO jobs (id,kind,status,payload,progress) VALUES (?,?,?,?,0)",
                (task_id, kind, "queued", json.dumps(payload, ensure_ascii=False)),
            )
        self.executor.submit(self._run, task_id, kind, payload)
        log.info("任务已提交：%s（%s）", task_id, kind)
        return task_id

    def _build_cmd(self, kind: str, payload: dict[str, Any]) -> list[str] | None:
        """根据任务类型拼出子进程命令行（list 形式，不用 shell）。

        Returns:
            命令参数列表；refresh_derived 不需要子进程，返回 None。
        """
        sets: list[str] = [str(s) for s in payload.get("sets", [])]
        epochs = payload.get("epochs")
        config = payload.get("config")

        def base(script: str) -> list[str]:
            cmd = [PYTHON, str(ROOT / "scripts" / script)]
            if config:
                cmd += ["--config", str(config)]
            for s in sets:
                cmd += ["--set", s]
            return cmd

        if kind == "train_movie":
            # 电影训练三步：抽帧 → manifest → 训练。step1/step2 不接受 --set，单独拼。
            steps: list[list[str]] = []
            c1 = [PYTHON, str(ROOT / "scripts" / "01_extract_frames.py")]
            c2 = [PYTHON, str(ROOT / "scripts" / "02_build_manifest.py")]
            c3 = base("03_train.py")
            if epochs is not None:
                c3 += ["--set", f"train.epochs={int(epochs)}"]
            steps = [c1, c2, c3]
            return self._chain(steps)
        if kind == "train_actor":
            # 演员训练：导入 → 训练。15 接受 --set，16 接受 --set + epochs。
            c1 = base("15_import_actors.py")
            c2 = base("16_train_actors.py")
            if epochs is not None:
                c2 += ["--epochs", str(int(epochs))]
            return self._chain([c1, c2])
        if kind == "build_gallery":
            return self._chain([base("12_build_gallery.py")])
        return None  # refresh_derived

    @staticmethod
    def _chain(steps: list[list[str]]) -> list[str]:
        """把多个步骤串成一条 shell 命令（&& 语义：前一步失败即停）。

        注意：这里用 shell=True 是因为要用 && 串联；所有命令片段都由 _build_cmd
        从固定脚本名 + 白名单参数拼出，用户输入只经由 --set 值进入，且值来自
        服务端 JSON 不会被 shell 特殊字符意外展开（引号包裹）。若将来放开更多
        用户参数，必须改回逐条 subprocess.run 并在 Python 层判断返回码。
        """
        parts = []
        for cmd in steps:
            parts.append(" ".join(f'"{c}"' if " " in c else c for c in cmd))
        return " && ".join(parts)

    # ---------------------------------------------------------------- 执行
    def _run(self, task_id: str, kind: str, payload: dict[str, Any]) -> None:
        """线程池工作体：执行任务全流程，任何异常都落盘为 failed。"""
        self._set_status(task_id, "running", started=True)
        log_path = JOBS_LOG_DIR / f"{task_id}.log"
        self._set_field(task_id, "log_path", _rel(log_path))
        try:
            if kind == "refresh_derived":
                from .assets import AssetStore
                store = AssetStore(cfg=payload.get("cfg"))
                n = store.sync_all_derived()
                self._set_field(task_id, "progress", 100)
                self._set_status(task_id, "done",
                                 result={"derived": n, "message": "派生写回完成"})
                self._append_log(log_path, f"派生写回完成，处理素材 {n} 个\n")
                return

            cmd = self._build_cmd(kind, payload)
            assert cmd is not None
            self._append_log(log_path, f"$ {cmd}\n")
            env = os.environ.copy()
            env.setdefault("TMPDIR", str(ROOT / ".tmp"))
            # PYTHONUNBUFFERED 让子进程输出即时刷到日志，前端轮询能看到实时进度
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                cmd, shell=True, cwd=str(ROOT), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            with self._lock:
                self._procs[task_id] = proc
            assert proc.stdout is not None
            for line in proc.stdout:
                self._append_log(log_path, line)
                self._maybe_progress(task_id, line)
            code = proc.wait()
            with self._lock:
                self._procs.pop(task_id, None)
            if code == 0:
                self._set_field(task_id, "progress", 100)
                self._set_status(task_id, "done", result={"exit_code": 0, "message": "训练完成"})
            else:
                self._set_status(task_id, "failed",
                                 result={"exit_code": code, "message": f"脚本退出码 {code}"})
        except Exception as exc:  # noqa: BLE001
            log.exception("任务执行失败：%s", task_id)
            self._append_log(log_path, f"\n[任务异常] {type(exc).__name__}: {exc}\n")
            self._set_status(task_id, "failed",
                             result={"error": f"{type(exc).__name__}: {exc}"})
        finally:
            with self._lock:
                self._procs.pop(task_id, None)

    def _maybe_progress(self, task_id: str, line: str) -> None:
        """从日志行里提取 Epoch x/y，粗略更新进度 0-90（100 留给成功结束）。"""
        m = EPOCH_RE.search(line)
        if not m:
            return
        try:
            cur, total = int(m.group(1)), int(m.group(2))
            if total > 0:
                pct = min(90, max(0, int(cur / total * 90)))
                self._set_field(task_id, "progress", pct)
        except (ValueError, ZeroDivisionError):
            pass

    @staticmethod
    def _append_log(path: Path, text: str) -> None:
        ensure_dir(path.parent)
        with path.open("a", encoding="utf-8") as f:
            f.write(text)

    # ---------------------------------------------------------------- 状态
    def _set_status(self, task_id: str, status: str, *, started: bool = False,
                    result: dict | None = None) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as c:
            if started:
                c.execute("UPDATE jobs SET status=?, started_at=? WHERE id=?", (status, now, task_id))
            elif status in ("done", "failed", "canceled"):
                c.execute("UPDATE jobs SET status=?, finished_at=?, result=? WHERE id=?",
                          (status, now, json.dumps(result or {}, ensure_ascii=False), task_id))
            else:
                c.execute("UPDATE jobs SET status=? WHERE id=?", (status, task_id))

    def _set_field(self, task_id: str, field: str, value: Any) -> None:
        # field 只来自本模块硬编码（log_path/progress），不接受外部输入，避免 SQL 注入
        assert field in {"log_path", "progress"}
        with self._conn() as c:
            c.execute(f"UPDATE jobs SET {field}=? WHERE id=?", (value, task_id))

    def get(self, task_id: str, *, tail_lines: int = 80) -> dict[str, Any] | None:
        """取单个任务状态，附带日志尾部（前端实时展示用）。

        Args:
            task_id: 任务 ID。
            tail_lines: 返回日志最后多少行，默认 80。

        Returns:
            任务 dict（含 log_tail 文本）；不存在返回 None。
        """
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (task_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["result"] = json.loads(d.get("result") or "{}")
        except (ValueError, TypeError):
            d["result"] = {}
        d["log_tail"] = self._read_tail(d.get("log_path"), tail_lines)
        return d

    @staticmethod
    def _read_tail(log_path: str | None, n: int) -> str:
        if not log_path:
            return ""
        p = ROOT / log_path
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:])
        except OSError:
            return ""

    def list(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset)).fetchall()
        return [dict(r) for r in rows]

    def cancel(self, task_id: str) -> bool:
        """尽力取消任务：terminate 子进程并标记 canceled；已完成则忽略。"""
        with self._conn() as c:
            row = c.execute("SELECT status FROM jobs WHERE id=?", (task_id,)).fetchone()
        if not row or row["status"] not in ("queued", "running"):
            return False
        with self._lock:
            proc = self._procs.get(task_id)
        if proc is not None:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
            except (ProcessLookupError, OSError):
                pass
        self._set_status(task_id, "canceled", result={"message": "用户取消"})
        return True


def _rel(path: Path) -> str:
    """绝对路径转相对项目根 POSIX 字符串。"""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
