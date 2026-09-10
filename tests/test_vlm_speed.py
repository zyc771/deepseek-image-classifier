"""限速器与连接复用测试（不发起网络请求）"""
import threading
import time

from app.vlm import RateLimiter, _session


class TestRateLimiter:
    def test_interval_from_rpm(self):
        rl = RateLimiter(60)
        assert abs(rl.interval - 1.0) < 1e-6
        rl2 = RateLimiter(120)
        assert abs(rl2.interval - 0.5) < 1e-6

    def test_zero_rpm_does_not_crash(self):
        rl = RateLimiter(0)
        assert rl.interval >= 1.0  # 兜底为最小速率

    def test_first_acquire_immediate(self):
        rl = RateLimiter(60)
        start = time.monotonic()
        rl.acquire()
        assert time.monotonic() - start < 0.2

    def test_second_acquire_respects_interval(self):
        rl = RateLimiter(120)  # 间隔 0.5s
        rl.acquire()
        start = time.monotonic()
        rl.acquire()
        assert time.monotonic() - start >= 0.35

    def test_concurrent_acquires_are_serialized_by_slots(self):
        """10 个线程各取一次令牌，总耗时至少 (n-1)*interval"""
        rl = RateLimiter(600)  # 间隔 0.1s
        errors = []

        def worker():
            try:
                rl.acquire()
            except Exception as e:
                errors.append(e)

        start = time.monotonic()
        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - start
        assert not errors
        assert elapsed >= 0.4  # 5 * 0.1s 的下限（留少许余量）


class TestSessionReuse:
    def test_same_session_per_thread(self):
        s1 = _session()
        s2 = _session()
        assert s1 is s2

    def test_different_threads_get_different_sessions(self):
        holder = {}

        def worker():
            holder["s"] = _session()

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert holder["s"] is not _session()
