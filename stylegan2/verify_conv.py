"""Native-vs-adapted convolution first/second derivative checks, CPU float64."""
import torch
from . import models, conv_gradfix
from torch_utils.ops import conv2d_gradfix as legacy


def main():
    torch.set_num_threads(4)
    torch.manual_seed(81)
    for transpose in (False, True):
        for groups in (1, 2):
            x = torch.randn(2, 4, 7, 7, dtype=torch.double, requires_grad=True)
            shape = (4, 6//groups, 3, 3) if transpose else (6, 4//groups, 3, 3)
            w = torch.randn(shape, dtype=torch.double, requires_grad=True)
            b = torch.randn(6, dtype=torch.double, requires_grad=True)
            native = torch.nn.functional.conv_transpose2d if transpose else torch.nn.functional.conv2d
            custom = conv_gradfix.conv_transpose2d if transpose else conv_gradfix.conv2d
            kw = dict(stride=2, padding=1, groups=groups)
            if transpose: kw['output_padding'] = 1
            with conv_gradfix.regularization():
                a = custom(x,w,b,**kw)
                z = native(x,w,b,**kw)
                torch.testing.assert_close(a,z,rtol=0,atol=0)
                ga = torch.autograd.grad(a.square().mean(),(x,w,b),create_graph=True)
                gz = torch.autograd.grad(z.square().mean(),(x,w,b),create_graph=True)
                for u,v in zip(ga,gz): torch.testing.assert_close(u,v,rtol=1e-10,atol=1e-10)
                ha = torch.autograd.grad(sum(t.square().sum() for t in ga),(x,w,b))
                hz = torch.autograd.grad(sum(t.square().sum() for t in gz),(x,w,b))
                for u,v in zip(ha,hz): torch.testing.assert_close(u,v,rtol=1e-10,atol=1e-10)
                assert torch.autograd.gradgradcheck(lambda xx,ww: custom(xx,ww,None,**kw),
                    (x,w),fast_mode=True,atol=1e-5,rtol=1e-3)
                # R1/PL input differentiation with direct weight gradient omitted.
                a = custom(x,w,b,**kw);z = native(x,w,b,**kw)
                with legacy.no_weight_gradients():
                    ga = torch.autograd.grad(a.square().sum(),x,create_graph=True)[0]
                    gz = torch.autograd.grad(z.square().sum(),x,create_graph=True)[0]
                ha = torch.autograd.grad(ga.square().sum(),w)[0]
                hz = torch.autograd.grad(gz.square().sum(),w)[0]
                torch.testing.assert_close(ha,hz,rtol=1e-10,atol=1e-8)
            print('PASS convolution and regularization derivatives',transpose,groups,flush=True)


if __name__ == '__main__': main()
