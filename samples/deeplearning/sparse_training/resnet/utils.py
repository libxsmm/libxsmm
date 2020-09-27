"""
Returns a list of prune_ratios that specifies the
prune rate for each epoch so that the end sparsity matches
the target prune_rate
"""
def prune_scheduler(
    prune_rate, epochs,
    start_epoch=0, end_epoch=0,
    increment_profile="natural"):
    end_epoch = epochs
    num_increments = end_epoch - start_epoch

    # (1-r)^n = prune_rate

    r = 1 - (prune_rate)**(1/num_increments)

    pruning_rate = []

    for i in range(epochs):
        if i < start_epoch:
            _r = 1.
        else:
            _r  = r

        pruning_rate.append(_r)
    return pruning_rate
