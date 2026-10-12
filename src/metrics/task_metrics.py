def classification_accuracy(predictions, targets):
    if len(predictions) != len(targets) or not targets:
        raise ValueError("Predictions and targets must have the same non-zero length")
    return sum(int(a == b) for a, b in zip(predictions, targets)) / len(targets)


def unavailable(reason="not_configured"):
    return {"status": "not_configured", "reason": reason}
