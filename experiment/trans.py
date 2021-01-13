import re
import time
import logging
from functools import partial
from functools import lru_cache
from collections import deque
from collections import namedtuple

from miros import Event
from miros import spy_on
from miros import signals
from miros import HsmWithQueues
from miros import return_status


def a(hsm, e):
  status = return_status.Handled

@lru_cache(maxsize=128)
def meta_trans2(hsm, *, t, s, sig):
  fn = hsm.outmost.meta_hooked(s=s, e=e)
  if fn not None:
    status = fn(hsm, e)
  elif(hsm.outmost.in_same_hsm(source=s, target=t)):
    status = hsm.trans(t)
  else:
    mt = MetaTrans('mt', hsm=hsm, t=t, s=s, sig=sig)
    mt.start_at(hook_metatrans)
    mt._complete_circuit()
    _state, _e = mt.status, mt.state
    if state:
      status = hsm.trans(_state)
      hsm.same._post_fifo(_e)
      investigate(hsm, e, _e)
    else:
      status = return_status.HANDLED
      investigate(hsm, e, _e)
      hsm.same.post_fifo(_e)

class MetaTrans(HsmWithQueues):
  def __init__(self, name, *, hsm=None, t=None, s=None, sig=None instrumented=True):
    super().__init__()
    self.name = name
    self.hsm  # Region or XmlChart
    assert(self.hsm.outmost)

    self.t = t
    assert(callable(self.t))
    self.s = s
    assert(callable(self.s))
    self.sig = sig
    assert(type(self.sig) == type(""))

    self.instrumented = instrumented

    # these will be determined by the state machine
    self._state = None
    self._e = None

    self.lca = self._lca(s=self.s, t=self.t)
    self.outer_injector = self._outer_injector()
    self.exit_onion = self._exit_onion()
    self.entry_onion = self._entry_onion()


  def _outer_injector():
    outer_injector = None
    if self.lca.__name__ == 'top':
      outer_injector = self.build_onion(t=self.t,sig=None)[-1]
    return outer_injector

  def _exit_onion():
    if not hasattr(self, 'outer_injector')
      self.outer_injector = self._outer_injector()
    exit_onion = []
    if self.outer_injector:
      exit_onion = self.build_onion(
        t=self.s,
        s=self.lca,
        sig=sig)
      exit_onion = exit_onion[1:] if len(exit_onion) >= 1 else []

      stripped = []
      for fn in exit_onion:
        stripped.append(fn)
        if fn == self.lca:
          break
      stripped.reverse()
      exit_onion = stripped[:]
    return exit_onion

  def _entry_onion():
    exit_onion = []
    if self.t != self.lca:
      exit_onion = self.build_onion(s=self.lca, t=self.t, sig=self.sig)[0:-1]

  def _lca(s, t):
    return self.hsm.outmost.lsa(s, t)

  def build_onion(self, t, sig, s=None)
    return self.hsm.outmost.build_onion(t=t, sig=sig, s=s)

  def within(self, fn_region_handler, fn_state_handler):
    return self.hsm.outmost(fn_region_handler, fn_state_handler)

  def _exit_onion(self):
    self.chart.outmost(

  def _post_fifo(self, e):
    super().post_fifo(e)

  def _post_lifo(self, e):
    super().post_lifo(e)

  def _complete_circuit(self)
    while len(self.queue) != 0:
      super().complete_circuit()

def hook_metatrans(mt, e):
  status = return_status.HANDLED
  return status


hsm = HsmWithQueues(instrumented=True)
