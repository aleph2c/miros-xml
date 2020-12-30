# successful experiment
from decimal import Decimal
from functools import singledispatch

@singledispatch
def fun(arg, verbose=True):
  if verbose:
    print("Let me just say,", end=" ")
  print(arg)

@fun.register(int)
def _(arg, verbose=False):
  if verbose:
    print("Strength in numbers, eh?", end=" ")
  print(arg)

@fun.register(list)
def _(arg, verbose=False):
  def _(arg, verbose=False):
    if verbose:
      print("Enumerate this:")
    for i, elem in enumerate(arg):
      print(i, elem)

def nothing(arg, verbose=False):
  print("nothing")

fun.register(type(None), nothing)

@fun.register(float)
@fun.register(Decimal)
def fun_num(arg, verbose=False):
  if verbose:
    print("Half of your number:", end=" ")
  print(arg / 3)

fun("hello, world.")
fun("test.", verbose=True)
fun(42, verbose=True)
fun(None, verbose=True)
