import numpy.linalg as la
import numpy as np
from numbers import Number


class FAD:
  def __init__(self, f, grad):
      self.f = f
      self.grad = np.copy(grad)

  ### String representation
  def __str__(self):
      return "FAD(val={}, deriv={})" .format(self.f, self.grad)

  ### Dimension of gradient
  def __len__(self):
      return len(self.grad)

  ########################################################################
  # Elementary arithmetic operations (FAD on left, FAD or Number on right
  ########################################################################

  ### Overloaded unary minus (-FAD)
  def __neg__(self):
      return FAD(-(self.f), -(self.grad))

  ### Overloaded addition (FAD + [Number|FAD])
  def __add__(self, other):
      if isConstant(other):
          return FAD(self.f+other, self.grad)
      if isinstance(other, FAD):
          return FAD(self.f+other.f, self.grad + other.grad)
      FAD.badarg(other, '__add__')

  ### Overloaded subtraction (FAD - [Number|FAD])
  def __sub__(self, other):
      if isConstant(other):
          return FAD(self.f - other, self.grad)
      if isinstance(other, FAD):
          return FAD(self.f - other.f, self.grad - other.grad)
      FAD.badarg(other, '__sub__')

  ### Overloaded multiplication (FAD * [Number|FAD])
  def __mul__(self, other):
      if isConstant(other):
          return FAD(other*self.f, other*self.grad)
      if isinstance(other, FAD):
          return FAD(other.f*self.f,
                     other.f*self.grad + other.grad*self.f)
      FAD.badarg(other, '__mul__')

  ### Overloaded division (FAD / [Number|FAD])
  def __truediv__(self, other):
      if isConstant(other):
          return FAD(self.f/other, self.grad/other)
      if isinstance(other, FAD):
          return FAD(self.f/other.f,
                     (other.f*self.grad - self.f*other.grad)/(other.f*other.f))
      FAD.badarg(other, '__div__')

  ########################################################################
  # Elementary arithmetic operations, with FAD argument on right
  ########################################################################

  ### Division: number/FAD
  def __rtruediv__(self, other):
      if isConstant(other):
          return FAD(other/self.f,
                     -other*self.grad/(self.f*self.f))
      FAD.badarg(other, '__rdiv__')

  ### Addition: number+FAD
  def __radd__(self, other):
      return self.__add__(other)
      FAD.badarg(other, '__rdiv__')

  ### Subtraction: number-FAD
  def __rsub__(self, other):
      return -self.__sub__(other)
      FAD.badarg(other, '__rsub__')

  ### Multiplication: number*FAD
  def __rmul__(self, other):
      return self.__mul__(other)
      FAD.badarg(other, '__rmul__')

  ########################################################################
  # Error handling helpers
  ########################################################################
  @staticmethod
  def badarg(x, f):
      raise TypeError('argument %r to function %r should be a Number or a FAD' % (x,f))

  @staticmethod
  def badarg2(x, y, f):
      raise TypeError('arguments %r and %r to function %r should be Numbers or FADs' % (x,y,f))


############################################################################
# Exponential, logs, and powers
############################################################################

def exp(x):
    if isConstant(x):
        return np.exp(x)
    if isinstance(x, FAD):
        ex = np.exp(x.f)
        return FAD(ex, ex*x.grad)
    FAD.badarg(x, 'exp')


def log(x):
    if isConstant(x):
        return np.log(x)
    if isinstance(x, FAD):
        return FAD(np.log(x.f), x.grad/x.f)
    FAD.badarg(x, 'log')

def log10(x):
    if isConstant(x):
        return np.log10(x)
    if isinstance(x, FAD):
        return FAD(np.log10(x.f), x.grad/x.f/np.log(10.0))
    FAD.badarg(x, 'log10')

def pow(x, y):
    if isConstant(x) and isConstant(y):
        return np.pow(x, y)
    if isinstance(x, FAD) and isConstant(y):
        return pow(x, FAD(y,0))
    if isConstant(x) and isinstance(y, FAD):
        return pow(FAD(x,0), y)
    if isinstance(x, FAD) and isinstance(y, FAD):
        return exp(y*log(x))
    FAD.badarg2(x, y, 'pow')

def sqrt(x):
    if isConstant(x):
        return np.sqrt(x)
    if isinstance(x, FAD):
        sx = np.sqrt(x.f)
        return FAD(sx, 0.5*x.grad/sx)
    FAD.badarg(x, 'sqrt')

############################################################################
# Trigonometry
############################################################################

def cos(x):
    if isConstant(x):
        return np.cos(x)
    if isinstance(x, FAD):
        return FAD(np.cos(x.f), -np.sin(x.f)*x.grad)
    FAD.badarg(x, 'cos')

def sin(x):
    if isConstant(x):
        return np.sin(x)
    if isinstance(x, FAD):
        return FAD(np.sin(x.f), np.cos(x.f)*x.grad)
    FAD.badarg(x, 'sin')

def tan(x):
    if isConstant(x):
        return np.tan(x)
    if isinstance(x, FAD):
        c = np.cos(x.f)
        return FAD(np.tan(x.f), x.grad/c/c)
    FAD.badarg(x, 'tan')

def atan(x):
    if isConstant(x):
        return np.arctan(x)
    if isinstance(x, FAD):
        return FAD(np.arctan(x.f), x.grad/(1.0 + x.f**2))
    FAD.badarg(x, 'atan')

def asin(x):
    if isConstant(x):
        return np.arcsin(x)
    if isinstance(x, FAD):
        return FAD(np.arcsin(x.f), x.grad/np.sqrt(1.0-x.f**2))
    FAD.badarg(x, 'asin')

def acos(x):
    if isConstant(x):
        return np.arccos(x)
    if isinstance(x, FAD):
        return FAD(np.arcsin(x.f), -x.grad/np.sqrt(1.0-x.f**2))
    FAD.badarg(x, 'asin')

def atan2(y, x):
    if isConstant(x) and isConstant(y):
        return np.arctan2(y, x)
    else:
        xx = getFADVal(x)
        yy = getFADVal(y)
        if xx==0.0 and yy==0.0:
            raise ValueError('0/0 in arctan2')
        if xx>0.0:
            return 2.0*atan(y/(sqrt(x*x + y*y)+x))
        if xx <= 0.0 and yy != 0.0:
            return 2.0*atan((sqrt(x*x + y*y)-x)/y)
        if xx < 0.0 and yy == 0.0:
            val = np.pi
            if isinstance(y,FAD):
                grad = getFADGrad(y)/getFADVal(x)
            else:
                grad = 0.0*getFADGrad(x)
            return FAD(val, grad)








############################################################################
# Hyperbolic functions
############################################################################

def cosh(x):
    if isConstant(x):
        return np.cosh(x)
    if isinstance(x, FAD):
        return FAD(np.cosh(x.f), np.sinh(x.f)*x.grad)
    FAD.badarg(x, 'cosh')

def sinh(x):
    if isConstant(x):
        return np.sinh(x)
    if isinstance(x, FAD):
        return FAD(np.sinh(x.f), np.cosh(x.f)*x.grad)
    FAD.badarg(x, 'sinh')

def tanh(x):
    if isConstant(x):
        return np.tanh(x)
    if isinstance(x, FAD):
        c = np.cosh(x.f)
        return FAD(np.tanh(x.f), x.grad/c/c)
    FAD.badarg(x, 'tanh')

def atanh(x):
    if isConstant(x):
        return np.arctanh(x)
    if isinstance(x, FAD):
        return FAD(np.arctanh(x.f), x.grad/(1.0 - x.f**2))
    FAD.badarg(x, 'atanh')

def asinh(x):
    if isConstant(x):
        return np.arcsinh(x)
    if isinstance(x, FAD):
        return FAD(np.arcsinh(x.f), x.grad/np.sqrt(1.0+x.f**2))
    FAD.badarg(x, 'asinh')

def acosh(x):
    if isConstant(x):
        return np.arccosh(x)
    if isinstance(x, FAD):
        return FAD(np.arccosh(x.f), x.grad/np.sqrt(x.f**2-1.0))
    FAD.badarg(x, 'asinh')



############################################################################
# Miscellaneous math
############################################################################

def fabs(x):
    if isConstant(x):
        return np.fabs(x)
    if isinstance(x, FAD):
        return FAD(np.fabs(x.f), x.grad*x.f/np.fabs(x.f))
    FAD.badarg(x, 'fabs')


############################################################################
# Non-member constructors
############################################################################

def FADVariable(val, dir, dim=3):
    grad = np.zeros(dim)
    grad[dir]=1
    return FAD(val, grad)

def FADConstant(c, dim=3):
    grad = np.array([0,0,0])
    return FAD(c, grad)

def FADVector(vec):
    if not isConstant(vec):
        rtn = []
        for i in range(len(vec)):
            rtn.append(FADVariable(vec[i], i, len(vec)))
        return rtn
    else:
        return FADVariable(vec, 0, 1)

def getFADVal(x):
    if isConstant(x):
        return x
    if isinstance(x, FAD):
        return x.f
    FAD.badarg(x, 'getFADVal')

def getFADGrad(x):
    if isConstant(x):
        return 0 # Should be a vector of zeros
    if isinstance(x, FAD):
        return x.grad
    FAD.badarg(x, 'getFADGrad')


def isConstant(x):

    if isinstance(x, Number):
        return True
    if isinstance(x, np.ndarray):
        if len(x)==1:
            return True
    return False

def FDCheck(func, XVec, h=1.0e-2):
    print('-' * 80)
    print('computing FD check of function ', func)
    gradFD = np.zeros(len(XVec))
    X0 = np.array(XVec)
    fadVars = FADVector(X0)
    f0 = func(X0)
    dx = np.array([-4, -3, -2, -1, 1, 2, 3, 4])*h
    w = np.array([1.0/280.0, -4.0/105.0, 1.0/5.0, -4.0/5.0,
                  4.0/5.0, -1.0/5.0, 4.0/105.0, -1.0/280.0])/h
    for i in range(len(XVec)):
        grad_i = 0.0
        delta = np.zeros(len(XVec))
        delta[i]=1.0
        for dx_j,w_j in zip(dx, w):
            Xj = X0 + dx_j*delta
            Fj = func(Xj)
            grad_i += w_j*Fj

        gradFD[i] = grad_i

    F = func(fadVars)
    fVal = getFADVal(F)
    gradF = getFADGrad(F)
    tol = 1.0e-11
    print('FD check: f={}, FAD.val={}'.format(f0,fVal))
    print('FD check: FD grad f={}, FAD.grad={}'.format(gradFD, gradF))
    print('val err=%25.15g' % np.abs(f0-fVal))
    print('grad err=%25.15g'% la.norm(gradFD-gradF))

    if np.abs(f0-fVal) > tol:
        print('ERROR in value calculation: ', np.abs(f0-fVal))
    if la.norm(gradFD-gradF) > tol:
        print('ERROR in derivative calculation: ', la.norm(gradFD-gradF))


def CartToPolar(X):
    if len(X)==2:
        rho = sqrt(X[0]*X[0] + X[1]*X[1])
        theta = atan2(X[1], X[0])
        return (rho,theta)
    if len(X)==3:
        r = sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2])
        theta = acos(z/r)
        phi = atan2(X[1], X[0])
        return (r,theta, phi)
    else:
        raise ValueError('dimension must be 2 or 3 in CartToPolar')





if '__main__'==__name__:

    from sys import exit

    # An independent variable x has dual representation (x,1)
    x = FADVariable(3,0)
    y = FADVariable(5,1)
    print('x=', x, ' y=', y)
    xy = x*y
    print('x*y=', xy)
    xpy = x+y
    print('x+y=', xpy)
    sinxy = sin(x*y)
    print('sin(x*y)=', sin(x*y))



    #exit(0)


    def f1(X):
        return X[0]

    def f2(X):
        return X[0]*X[0] + X[1]*X[1] + X[2]*X[2]

    def f3(X):
        return sqrt(f2(X))

    X = np.array([1.0, np.pi, np.sqrt(2.0)])

    FDCheck(f1, X)
    FDCheck(f2, X)
    FDCheck(f3, X)


    x = np.array([0.5])
    y = np.array([0.25])
    for f in (sqrt, exp, log, log10, cos, sin, tan, acos, asin, atan,
             cosh, sinh, tanh, asinh, atanh):
        print('function = ', f)
        FDCheck(f, x)

    x[0] = 2.0
    print('function = acosh')
    FDCheck(acosh, x)

    p = FADVariable(x[0], 0, 1)
    q = FADVariable(y[0], 0, 1)
    print('x/y=', p/q)

    n = 180
    for i in range(n):
        t = -np.pi + FADVariable(2*i*np.pi/n, 0, 1)
        x = cos(t)
        y = sin(t)
        print('t={} arctan2(y,x)={}'.format(getFADVal(t), atan2(y, x)))
