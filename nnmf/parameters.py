import torch

class NonNegativeParameter(torch.nn.Parameter):
    """
    A parameter that is constrained to be non-negative.
    """
    def __new__(cls, data):
        if torch.any(data < 0):
            raise ValueError("Negative values are not allowed in the parameter data.")
        return super(NonNegativeParameter, cls).__new__(cls, data, requires_grad=True)

    def _check_negative_values(self):
        if torch.any(self.data < 0):
            raise ValueError("Negative values are not allowed in the parameter data.")

    def __setattr__(self, name, value):
        if name == 'data':
            self._check_negative_values()
            super(NonNegativeParameter, self).__setattr__(name, value)
        else:
            super(NonNegativeParameter, self).__setattr__(name, value)

    def __setitem__(self, key, value):
        self._check_negative_values()
        super(NonNegativeParameter, self).__setitem__(key, value)

    def __delitem__(self, key):
        self._check_negative_values()
        super(NonNegativeParameter, self).__delitem__(key)

    def __setstate__(self, state):
        self._check_negative_values()
        super(NonNegativeParameter, self).__setstate__(state)

class ParameterList(torch.nn.ParameterList):
    def requires_grad_(self, requires_grad=True):
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def size(self, idx=None):
        if idx is None:
            return [p.size() for p in self.parameters()]
        return self.parameters[idx].size()