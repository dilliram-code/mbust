from network import Network

xor_input=[[0,0],[1,0],[0,1],[1,1]]
xor_ideal=[[0],[1],[1],[0]]
network=Network(2,3,1,0.7,0.9)
for epoch in range(10000):
    for x,y in zip(xor_input,xor_ideal):
        network.compute_outputs(x)
        network.calc_error(y)
        network.learn()
print("Recall")
for x in xor_input:
    print(x,"->",round(network.compute_outputs(x)[0],4))