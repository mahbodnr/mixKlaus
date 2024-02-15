import torch.nn.functional as F
import torch

def debugger_wrapper(func):
    def wrapper(*args, **kwargs):
        tensor_args = {k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
        # add args to tensor_args
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                tensor_args[f"arg{i}"] = arg
        output= func(*args, **kwargs)
        tensor_args["output"] = output
        errors = check_tensors(tensor_args)
        if errors:
            raise ValueError(
                f"Error in {func.__name__}:\n"
                + "\n\n".join(errors)
                + "\n\n".join([
            f"Tensor {name}: shape {tensor.shape}, min {tensor.min()}, max {tensor.max()}, mean {tensor.mean()}, std {tensor.std()}, sum {tensor.sum()}, nans {tensor.isnan().sum()}, infs {tensor.isinf().sum()}" for name, tensor in tensor_args.items()
        ])
                )
        return output
    return wrapper

for func in [d for d in dir(F) if (
    callable(getattr(F, d))
    and not d.startswith('_') 
    and 'function' in type(getattr(F, d)).__name__
    )]:
    exec(f"{func} = debugger_wrapper(F.{func})")

def check_tensors(tensors):
    if not tensors:
        return
    errors = []
    for name, tensor in tensors.items():
        if tensor.isnan().any():
            errors.append(f"Tensor {name} has NaN values.")
        elif not tensor.isfinite().all():
            errors.append(f"Tensor {name} has infinite values.")
    return errors


    
if __name__ == "__main__":
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    y[0, 0] = float('inf')
    prelu(x, y)
