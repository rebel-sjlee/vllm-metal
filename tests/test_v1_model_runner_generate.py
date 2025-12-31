# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import vllm_metal.v1.model_runner as mr


def _make_runner(mr_module):
    runner = mr_module.MetalModelRunner.__new__(mr_module.MetalModelRunner)
    runner.model = object()
    runner.tokenizer = object()
    return runner


def test_generate_accumulates_streamed_segments(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_stream_generate(model, tokenizer, prompt, max_tokens=256, **kwargs):
        captured["prompt"] = prompt
        captured["max_tokens"] = max_tokens
        captured["kwargs"] = kwargs
        yield SimpleNamespace(text="hello")
        yield SimpleNamespace(text=" ")
        yield SimpleNamespace(text="world")

    monkeypatch.setattr(mr, "stream_generate", fake_stream_generate)

    runner = _make_runner(mr)
    out = runner.generate("p", max_tokens=3, temperature=0.0)

    assert out == "hello world"
    assert captured["prompt"] == "p"
    assert captured["max_tokens"] == 3
    assert captured["kwargs"].get("temp") == 0.0


def test_generate_falls_back_to_temp_if_sampler_rejected(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_stream_generate(model, tokenizer, prompt, max_tokens=256, **kwargs):
        captured["kwargs"] = kwargs
        assert "temp" in kwargs
        yield SimpleNamespace(text="a")
        yield SimpleNamespace(text="b")

    monkeypatch.setattr(mr, "stream_generate", fake_stream_generate)

    runner = _make_runner(mr)
    out = runner.generate("p", max_tokens=2, temperature=0.0)

    assert out == "ab"
    assert "temp" in captured["kwargs"]
