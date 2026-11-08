"""Fine-tuning batch/mask test. Run: python -m tests.test_finetune"""
import torch

from src.finetune import make_batch


def test_mask_covers_exactly_the_answer_tokens():
    # ids: 3 prompt tokens (10,11,12) then 2 answer tokens (20,21) then <eot>=0
    ex = [([10, 11, 12, 20, 21, 0], 3), ([10, 12, 20, 0], 2)]
    x, y, m = make_batch(ex, pad_id=1, device="cpu")
    assert x.shape == y.shape == m.shape == (2, 5)
    # positions where the mask is on must be exactly the answer tokens + eot in y
    assert y[0][m[0].bool()].tolist() == [20, 21, 0]
    assert y[1][m[1].bool()].tolist() == [20, 0]
    assert x[1, 3:].tolist() == [1, 1] and m[1, 3:].sum() == 0              # padding never counted


if __name__ == "__main__":
    test_mask_covers_exactly_the_answer_tokens()
    print("PASS test_finetune")
