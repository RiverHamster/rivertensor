import rivertensor as rt

a = rt.randn([2, 2], requires_grad=True)
b = rt.randn([2, 2], requires_grad=True)
c = rt.sigmoid(a + b)
d = rt.relu(a + b)
e = c + d
e.backward()

print(a.grad)
print(b.grad)
print(c.grad)
