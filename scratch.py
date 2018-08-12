import torch

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.B = torch.nn.Parameter(torch.Tensor())

class A(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = M()

import myutils
import pandas

pd = myutils.get_gaga_as_pandas_datasets()
pd[0].to_csv("gaga_pandas_train.csv")
pd[1].to_csv("gaga_pandas_test.csv")

