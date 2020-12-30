import re
import time
import logging
from functools import partial
from collections import deque
from collections import namedtuple

from miros import Event
from miros import spy_on
from miros import signals
from miros import ActiveObject
from miros import return_status

def empty_entry_decorator(fn):
  def _wrapper(r, e):
    status = fn(r, e)
    print("testing for entry")
    if status == return_status.UNHANDLED:
      if(e.signal == signals.ENTRY_SIGNAL):
        print('entry found')
        status = return_status.UNHANDLED
    return status
  return _wrapper

def empty_exit_decorator(fn):
  def _wrapper(r, e):
    status = fn(r, e)
    print("testing for exit")
    if status == return_status.UNHANDLED:
      if(e.signal == signals.EXIT_SIGNAL):
        print('exit found')
        status = return_status.UNHANDLED
    return status
  return _wrapper

@empty_exit_decorator
@empty_entry_decorator
def state_function_1(r, e):
  status = return_status.UNHANDLED
  print("user defined")
  return status

state_function_1('region', Event(signal=signals.ENTRY_SIGNAL))
print("")
state_function_1('region', Event(signal=signals.ANYTHING))
