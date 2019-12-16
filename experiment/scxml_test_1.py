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

@spy_on
def Start(chart, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    chart.scribble("Hello from 'start'")
    status = chart.trans(Work)
  elif(e.signal == signals.scxml_immediate):
    status = chart.trans(Work)
  else:
    chart.temp.fun = chart.top
    status = return_status.SUPER
  return status

@spy_on
def Work(chart, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    chart.scribble("Hello from 'work'")
    status = return_status.HANDLED
  else:
    chart.temp.fun = chart.top
    status = return_status.SUPER
  return status

if __name__ == '__main__':
  ao = ActiveObject('scxml')
  ao.live_spy = True
  ao.start_at(Start)
  ao.post_fifo(Event(signal=signals.scxml_immediate))
  time.sleep(0.1)
