import torch

from mjlab.tasks.velocity.rl.amp.replay_buffer import ReplayBuffer


def test_replay_buffer_insert_and_sample():
    device = "cpu"
    buf = ReplayBuffer(obs_dim=8, buffer_size=10, device=device)

    # Insert less than capacity
    s = torch.randn(7, 8)
    ns = torch.randn(7, 8)
    buf.insert(s, ns)
    assert len(buf) == 7

    # Two mini-batches without replacement (total <= len)
    gen = buf.feed_forward_generator(num_mini_batch=2, mini_batch_size=3, allow_replacement=False)
    batches = list(gen)
    assert len(batches) == 2
    for b_s, b_ns in batches:
        assert b_s.shape == (3, 8)
        assert b_ns.shape == (3, 8)

    # Insert more to wrap-around and fill capacity
    s2 = torch.randn(10, 8)
    ns2 = torch.randn(10, 8)
    buf.insert(s2, ns2)
    assert len(buf) == 10  # capped at capacity

    # Request more total than stored -> samples with replacement
    gen2 = buf.feed_forward_generator(num_mini_batch=5, mini_batch_size=4, allow_replacement=True)
    batches2 = list(gen2)
    assert len(batches2) == 5
    for b_s, b_ns in batches2:
        assert b_s.shape == (4, 8)
        assert b_ns.shape == (4, 8)