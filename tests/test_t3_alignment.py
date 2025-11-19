import torch

from chatterbox.models.t3.t3 import T3


class DummyAnalyzer:
    def __init__(self, delta: float = 1.0):
        self.delta = delta
        self.calls = []

    def step(self, logits, next_token=None):
        self.calls.append((logits.clone(), next_token))
        return logits + self.delta


def test_apply_alignment_analyzers_noop_when_none():
    logits = torch.randn(2, 4)
    generated = torch.ones(2, 1, dtype=torch.long)

    result = T3._apply_alignment_stream_analyzers(
        logits.clone(),
        None,
        generated,
        finished_mask=torch.zeros(2, dtype=torch.bool),
    )

    assert torch.equal(result, logits)


def test_apply_alignment_analyzers_single():
    analyzer = DummyAnalyzer(delta=0.5)
    logits = torch.zeros(1, 3)
    generated = torch.tensor([[10, 11]], dtype=torch.long)

    result = T3._apply_alignment_stream_analyzers(
        logits.clone(),
        analyzer,
        generated,
        finished_mask=torch.zeros(1, dtype=torch.bool),
    )

    assert torch.allclose(result, torch.full((1, 3), 0.5))
    assert analyzer.calls, "Analyzer was not invoked"
    assert torch.equal(analyzer.calls[0][0], logits)
    assert analyzer.calls[0][1].item() == 11


def test_apply_alignment_analyzers_batch_list():
    analyzers = [DummyAnalyzer(delta=idx + 1) for idx in range(3)]
    logits = torch.zeros(3, 2)
    generated = torch.tensor(
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ],
        dtype=torch.long,
    )

    result = T3._apply_alignment_stream_analyzers(
        logits.clone(),
        analyzers,
        generated,
        finished_mask=torch.zeros(3, dtype=torch.bool),
    )

    expected = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ]
    )
    assert torch.allclose(result, expected)

    for idx, analyzer in enumerate(analyzers):
        assert analyzer.calls, f"Analyzer {idx} was not invoked"
        logged_logits, token = analyzer.calls[0]
        assert torch.equal(logged_logits, torch.zeros(1, 2))
        assert token.item() == generated[idx, -1].item()


def test_apply_alignment_analyzers_raises_on_mismatch():
    analyzers = [DummyAnalyzer(delta=1.0)]
    logits = torch.zeros(2, 2)
    generated = torch.zeros(2, 1, dtype=torch.long)

    try:
        T3._apply_alignment_stream_analyzers(
            logits,
            analyzers,
            generated,
            finished_mask=torch.zeros(2, dtype=torch.bool),
        )
        assert False, "Expected ValueError for mismatched analyzer count"
    except ValueError:
        pass


def test_apply_alignment_analyzers_skips_finished_rows():
    analyzer = DummyAnalyzer(delta=2.0)
    logits = torch.zeros(1, 2)
    generated = torch.ones(1, 1, dtype=torch.long)
    finished = torch.tensor([True])

    result = T3._apply_alignment_stream_analyzers(
        logits.clone(),
        analyzer,
        generated,
        finished_mask=finished,
    )

    assert torch.equal(result, logits)
    assert not analyzer.calls, "Analyzer should not run for finished samples"

