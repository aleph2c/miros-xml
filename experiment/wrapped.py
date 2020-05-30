from functools import wraps

def decorator(fn):

  @wraps(fn)
  def _decorator(chart, *args):
    status = fn(chart, *args)
    return status
  return _decorator

@decorator
def some_function(chart, e):
  return e

def function_that_takes_a_function(fn):
  fn1 = fn.__wrapped__
  return fn1('c', 4)

some_function('a', 5)
not_wrapped1 = some_function.__wrapped__
not_wrapped1('b', 3)

bob = some_function
not_wrapped_2 = bob.__wrapped__
not_wrapped_2('c', 4)

function_that_takes_a_function(some_function)
