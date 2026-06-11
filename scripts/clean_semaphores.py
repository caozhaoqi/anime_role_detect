"""
清理 macOS 泄漏的系统信号量（Semaphore）
在模型服务启动前调用，清理之前崩溃进程残留的信号量
"""
import subprocess
import re


def clean_semaphores():
    """清理当前用户拥有的所有 System V 信号量"""
    try:
        # 列出所有信号量
        result = subprocess.run(
            ["ipcs", "-s"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return 0

        sem_ids = []
        for line in result.stdout.split("\n"):
            # 匹配信号量行：s  65536 0x51103625 --ra------- caozhaoqi    staff
            match = re.match(r"s\s+(\d+)", line.strip())
            if match:
                sem_ids.append(match.group(1))

        if not sem_ids:
            return 0

        for sid in sem_ids:
            subprocess.run(
                ["ipcrm", "-s", sid],
                capture_output=True, timeout=3,
            )

        print(f"[CLEAN_SEM] 已清理 {len(sem_ids)} 个泄漏的信号量")
        return len(sem_ids)
    except Exception as e:
        print(f"[CLEAN_SEM] 清理失败: {e}")
        return -1


if __name__ == "__main__":
    n = clean_semaphores()
    print(f"清理完成: {n} 个信号量")