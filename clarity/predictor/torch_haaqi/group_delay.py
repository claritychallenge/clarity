# from __future__ import annotations
#
# import torch
# from torch.autograd import Function
# from torch import tensor
#
#
# class GroupDelay(Function):
#     @staticmethod
#     def forward(ctx, system: tuple[tensor, tensor], w=512) -> tensor:
#         """
#         Calculate the group delay of a system using the Hilbert transform.
#
#         Parameters
#         ----------
#         system : tuple[tensor, tensor]
#             A tuple of the numerator and denominator of the transfer function.
#         w : int, optional
#             The number of frequencies to evaluate the group delay at. The default is 512.
#
#         Returns
#         -------
#         tensor
#             The group delay of the system.
#
#         """
#         b, a = system
#         w, h = freqz(b, a, w=w)
#         gd = -torch.imag(torch.log(h)) / (2 * torch.pi * torch.real(w))
#         ctx.save_for_backward(gd)
#         return gd
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         (result,) = ctx.saved_tensors
#         return grad_output * result


import torch


def group_delay(b, a, w):
    # Calculate the frequency response of the filter
    z = torch.exp(1j * w)
    num = torch.sum(
        b.reshape(-1, 1) * z.pow(torch.arange(b.shape[0]).reshape(-1, 1)), dim=0
    )
    den = torch.sum(
        a.reshape(-1, 1) * z.pow(torch.arange(a.shape[0]).reshape(-1, 1)), dim=0
    )
    h = num / den

    # Calculate the phase response of the filter
    phase = torch.atan2(h.imag, h.real)

    # Calculate the group delay of the filter
    gd = torch.diff(phase) / torch.diff(w)[0]

    return gd


if __name__ == "__main__":
    from scipy import signal

    b, a = signal.iirdesign(0.1, 0.3, 5, 50, ftype="cheby1")
    w, gd = signal.group_delay((b, a))

    gd_2 = group_delay(torch.tensor(b), torch.tensor(a), torch.tensor(w))

    print(gd - gd_2.numpy())
