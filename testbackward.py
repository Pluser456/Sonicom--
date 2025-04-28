import torch
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
mu = torch.randn(10, requires_grad=True)
c = torch.randn(10, requires_grad=True)
true_w = torch.tensor([2.0],dtype=torch.float32)
true_b = torch.tensor([1.0],dtype=torch.float32)
true_mu = torch.tensor([3.0]*10,dtype=torch.float32)
true_c = torch.tensor([4.0]*10,dtype=torch.float32)

for i in range(10):
    x =torch.randn(1)
    z = torch.randn(10)
    y = w*x + b
    for j in range(5):
        latter_y = mu[2*j:2*j+2]*z[2*j:2*j+2] + c[2*j:2*j+2]
        midtensor = torch.cat((y, latter_y), dim=0)
        loss = (2*x + 1 + (3*z[2*j:2*j+2] + 2).sum() - midtensor.sum())**2
        retain_graph = (j < 4)  # 除了最后一次迭代，其他都保留计算图
        loss.backward(retain_graph=retain_graph)
        with torch.no_grad():
            w -= 0.1 * w.grad
            b -= 0.1 * b.grad
            mu -= 0.1 * mu.grad
            c -= 0.1 * c.grad
            print("w的梯度:", w.grad, "w的值:", w)
            print("b的梯度:", b.grad, "b的值:", b)
            print("mu的梯度:\n", mu.grad, "mu的值:\n", mu)
            print("c的梯度:\n", c.grad, "c的值:\n", c)
            print("loss:", loss.item())
        w.grad.zero_()
        b.grad.zero_()
        mu.grad.zero_()
        c.grad.zero_()

    
    # 为非标量输出提供梯度参数
    # 这里提供一个全1张量，相当于对所有元素求和
    # grad_tensor = torch.ones_like(vector_loss)
    # vector_loss.backward(grad_tensor)
    # print("x的梯度:", x.grad)  # x的梯度
    # print("w的梯度:", w.grad)  # w的梯度
    # print("b的梯度:", b.grad)  # b的梯度