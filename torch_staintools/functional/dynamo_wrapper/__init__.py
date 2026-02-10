from torch import _dynamo as torchdynamo

# mark_dynamic_ = torchdynamo.mark_dynamic

def mark_dynamic_(*args, **kwargs): ...
