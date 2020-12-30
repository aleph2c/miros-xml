# failed experiment
import types
from miros import signals
from functools import singledispatch

def __init__(self, signal, payload=None):
  self.signal = signal
  if payload is not None:
    self.payload = payload

def payload(self):
  return self.payload

def signal(self):
  return self.signal


cls_dict = {
  '__init__': __init__,
  'payload': payload,
  'signal': signal,
}

ENTRY_EVENT = types.new_class(
  'ENTRY_EVENT',
   (),
   {},
   lambda ns: ns.update(cls_dict)
)
ENTRY_EVENT.__module__ = __name__

EXIT_EVENT = types.new_class(
  'EXIT_EVENT',
   (),
   {},
   lambda ns: ns.update(cls_dict)
)
EXIT_EVENT.__module__ = __name__

entry_event = ENTRY_EVENT(signal=signals.ENTRY_SIGNAL)
exit_event = EXIT_EVENT(signal=signals.EXIT_SIGNAL)

@singledispatch
def fun(e):
  print(e.__class__)

fun.register(ENTRY_EVENT)
def _(e):
  print('entry')

fun.register(EXIT_EVENT)
def _(e):
  print('exit')

print(type(entry_event) == ENTRY_EVENT)
print(type(exit_event) == EXIT_EVENT)

fun(entry_event)  # fails, calls fun default function
fun(exit_event)   # fails, calls fun default function
