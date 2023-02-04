#copy the final value function from previous notebook
class Value:
  def __init__(self, data, _children = (), _op='', label = ''):
    self.data = data
    self._prev = set(_children)
    self._op = _op
    self.label = label
    self.grad = 0.0
    self._backward = lambda : None 

  #printing format
  def __repr__(self):
    return f"Value(data={self.data})"

  #addition function {Value Object + (Value Object or a number)}
  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), "+")
    
    def backward_fun():
        self.grad += 1.0 * out.grad
        other.grad += 1.0 * out.grad
    out._backward = backward_fun

    return out

  #substraction function {Value Object - (Value Object or a number)}
  def __sub__(self, other):
    return self + (-other)

  #substraction function {number - Value Object}
  def __rsub__(self, other):
    return self + (-other)

  #multiplication function {Value Object * (Value Object or a number)}
  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*") 
    
    def backward_fun():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
    out._backward = backward_fun

    return out

  #reverse add function {number * Value Object}
  def __radd__(self, other): 
    return self + other

  #reverse multiplication function {number * Value Object}
  def __rmul__(self, other):
    return self * other

  #division function {(Value object or number) / (Value object or number)}
  def __truediv__(self, other):
    return self * other**-1

  #power function {Value object ^ number}
  def __pow__(self, other):
    assert isinstance(other, (int, float)) #allow only the int and float as other for power values
    out = Value(self.data ** other, (self, ), f'**{other}')

    def backward_fun():
      self.grad += (other * (self.data ** (other -1))) * out.grad
    out._backward = backward_fun

    return out;

  #tanh fumction on the current Value object
  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')

    def backward_fun():
        out.grad += 1.0
        self.grad += (1 - t**2) * out.grad 
    out._backward = backward_fun

    return out

  #exponential on the current value object
  def exp(self):
    out = Value(math.exp(self.data), (self,), 'exp')

    def backward_fun():
      self.grad += out.data * out.grad
    out._backward = backward_fun

    return out

  #triggering Backpropagation from the last node
  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()