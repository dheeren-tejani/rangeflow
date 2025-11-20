from .backend import get_backend
xp = get_backend()

def robust_cross_entropy(out_range, targets):
    l, h = out_range.decay()
    # Worst case: Max for wrong, Min for correct
    worst = h.copy()
    rows = xp.arange(len(targets))
    worst[rows, targets] = l[rows, targets]
    
    # LogSoftmax
    shift = xp.max(worst, axis=1, keepdims=True)
    z = worst - shift
    log_probs = z - xp.log(xp.sum(xp.exp(z), axis=1, keepdims=True))
    return -xp.mean(log_probs[rows, targets])