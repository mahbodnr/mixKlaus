import torch.nn.functional as F
import torch
import os
from datetime import datetime

BASE_PATH = "./debug/logs/"
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(torch.randint(0, 100000, (1,)).item())

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
            # make a folder for the run
            run_path = BASE_PATH + RUN_ID
            os.makedirs(run_path, exist_ok=True)
            # save tensors to file
            for name, tensor in tensor_args.items():
                torch.save(tensor, os.path.join(run_path, f"{name}.pt"))
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
