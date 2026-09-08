from __future__ import annotations

import asyncio
import threading

import httpx
import pytest
from app import main
from app.models import ComplianceReport, PipelineState, Transaction
from app.scenarios import scenario_low_risk


def _success(transaction: Transaction) -> PipelineState:
    return PipelineState(
        transaction=transaction,
        report=ComplianceReport(transaction=transaction),
        processing_steps=["completed"],
    )


async def _request(client: httpx.AsyncClient, custom: bool) -> httpx.Response:
    if custom:
        return await client.post(
            "/api/analyze/custom", json=scenario_low_risk().model_dump(mode="json")
        )
    return await client.post("/api/analyze", json={"scenario_key": "low_risk"})


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test")


class _BlockedPipeline:
    def __init__(self, *, fail: bool = False) -> None:
        self.loop = asyncio.get_running_loop()
        self.started: asyncio.Queue[Transaction] = asyncio.Queue()
        self.release = threading.Event()
        self.timed_out = threading.Event()
        self.calls = 0
        self.lock = threading.Lock()
        self.fail = fail

    def __call__(self, transaction: Transaction) -> PipelineState:
        with self.lock:
            self.calls += 1
        self.loop.call_soon_threadsafe(self.started.put_nowait, transaction)
        if not self.release.wait(timeout=3):
            self.timed_out.set()
        if self.fail:
            raise RuntimeError("controlled failure after cancellation")
        return _success(transaction)

    async def wait_started(self) -> None:
        await asyncio.wait_for(self.started.get(), timeout=5)


@pytest.mark.parametrize("custom", [False, True])
def test_health_responds_while_analysis_is_blocked(
    monkeypatch: pytest.MonkeyPatch, custom: bool
) -> None:
    async def exercise() -> None:
        pipeline = _BlockedPipeline()
        monkeypatch.setattr(main, "run_compliance_pipeline", pipeline)
        async with _client() as client:
            task = asyncio.create_task(_request(client, custom))
            try:
                await pipeline.wait_started()
                response = await asyncio.wait_for(client.get("/api/health"), timeout=1)
                assert response.json() == {"status": "healthy", "version": "1.0.0"}
                assert not pipeline.timed_out.is_set()
                assert not task.done()
            finally:
                pipeline.release.set()
                await task

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel_outcome", ["none", "success", "error"])
def test_routes_share_four_slots_until_actual_work_finishes(
    monkeypatch: pytest.MonkeyPatch, cancel_outcome: str
) -> None:
    async def exercise() -> None:
        pipeline = _BlockedPipeline(fail=cancel_outcome == "error")
        monkeypatch.setattr(main, "run_compliance_pipeline", pipeline)
        async with _client() as client:
            tasks = [
                asyncio.create_task(_request(client, custom))
                for custom in [False, True, False, True]
            ]
            try:
                for _ in tasks:
                    await pipeline.wait_started()
                assert not pipeline.timed_out.is_set()
                if cancel_outcome != "none":
                    tasks[0].cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await tasks[0]
                for custom in [False, True]:
                    response = await asyncio.wait_for(_request(client, custom), timeout=1)
                    assert response.status_code == 503
                assert pipeline.calls == 4
            finally:
                pipeline.release.set()
                await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(exercise())

    # asyncio.run waits for worker shutdown, including the canceled request's work.
    async def reuse() -> None:
        pipeline = _BlockedPipeline()
        monkeypatch.setattr(main, "run_compliance_pipeline", pipeline)
        async with _client() as client:
            tasks = [
                asyncio.create_task(_request(client, custom))
                for custom in [False, True, False, True]
            ]
            try:
                for _ in tasks:
                    await pipeline.wait_started()
                assert not pipeline.timed_out.is_set()
            finally:
                pipeline.release.set()
                responses = await asyncio.gather(*tasks)
            assert all(response.status_code == 200 for response in responses)

    asyncio.run(reuse())


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("outcome", ["success", "error", "missing_report"])
def test_completion_preserves_responses_and_releases_capacity(
    monkeypatch: pytest.MonkeyPatch, custom: bool, outcome: str
) -> None:
    transactions: list[Transaction] = []

    def pipeline(transaction: Transaction) -> PipelineState:
        transactions.append(transaction)
        if outcome == "error":
            raise RuntimeError("controlled failure")
        if outcome == "missing_report":
            return PipelineState(transaction=transaction)
        return _success(transaction)

    monkeypatch.setattr(main, "run_compliance_pipeline", pipeline)

    async def exercise() -> None:
        async with _client() as client:
            for _ in range(6):
                response = await _request(client, custom)
                if outcome == "success":
                    assert response.status_code == 200
                    payload = response.json()
                    assert set(payload) == {
                        "success",
                        "processing_time_ms",
                        "report",
                        "executive_summary",
                        "graph_trace",
                    }
                    assert payload["success"] is True
                    assert payload["executive_summary"] == "Processing incomplete."
                    assert payload["graph_trace"] == ["completed"]
                    assert payload["report"]["transaction"]["transaction_id"] == (
                        transactions[-1].transaction_id
                    )
                else:
                    assert response.status_code == 500
                    assert response.json()["detail"] == (
                        "Pipeline error — see server logs for details"
                        if outcome == "error"
                        else "Pipeline completed but no report was generated"
                    )
            response = await client.post("/api/analyze", json={"scenario_key": "unknown"})
            assert response.status_code == 400
            assert len(transactions) == 6

    asyncio.run(exercise())
