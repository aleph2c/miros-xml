


import re
import time
import logging
from functools import partial
from collections import namedtuple

from miros import pp
from miros import Event
from miros import spy_on
from miros import signals
from miros import ActiveObject
from miros import return_status

@spy_on
def Start(self, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    self.scribble('Hello from \"start\"')
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.SCXML_INIT_SIGNAL):
    status = self.trans(Work)
  elif(e.signal == signals.EXIT_SIGNAL):
    status = return_state.HANDLED
  else:
    self.temp.fun = self.top
    status = return_status.SUPER
  return status

@spy_on
def Work(self, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    self.scribble('Hello from \'work\'')
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.SCXML_INIT_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    status = return_state.HANDLED
  else:
    self.temp.fun = self.top
    status = return_status.SUPER
  return status



class ScxmlChart(ActiveObject):
  def __init__(self, name):
    super().__init__(name)

ao = ScxmlChart("Scxml")
ao.live_spy = True

ao.start_at(Start)
