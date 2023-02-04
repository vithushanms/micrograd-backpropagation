import random

class Neuron:
  #nin  is number of inputs to the nurons
  def __init__(self, nin):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))

  def __call__(self,x):
    act = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b
    return act.tanh()
   #return the parameters in the neurons
  def parameters(self):
    return self.w + [self.b]


class Layer:
  #here nout is the number of neurons we want in a layer
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]

  def __call__(self,x):
    out = [n(x) for n in self.neurons]
    return out[0] if len(out) == 1 else out

  #return the parameters in the layer from all the neurons
  def parameters(self):
    parameters = []
    for neuron in self.neurons:
      parameters.extend(neuron.parameters())
    return parameters
  

class MLP:
  #here the nouts is a list of numbers in each layers
  def __init__(self, nin, nouts):
    #appeand the number of inputs to the layers's first element
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)

    return x

  #get all the parameters of MLP from all the layers 
  def parameters(self):
    parameters = []
    for layer in self.layers:
      parameters.extend(layer.parameters())
    return parameters