import re
import time
import dill
import logging
from functools import partial
from collections import deque

from miros import Event
from miros import spy_on
from miros import signals
from miros import ActiveObject
from miros import return_status

def _build_payload():
  from collections import namedtuple
  Payload = namedtuple('Payload', ['proof'])
  payload = Payload(proof=True)
  return payload

dill.dump(_build_payload, open("build_payload.p", "wb"))
#fun_from_binary('hello world')

@spy_on
def Start(chart, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    chart.scribble("Hello from 'start'")
    status = chart.trans(Work)
  elif(e.signal == signals.build_payload_and_trans):
    pickled_function = dill.load(open("build_payload.p", "rb"))
    payload = pickled_function()
    chart.post_fifo(Event(signal=signals.evidence, payload=payload))
    status = return_status.HANDLED
  elif(e.signal == signals.evidence):
    if e.payload.proof == True:
      output = "---------- IT WORKED!! :) ---------"
    else:
      output = "---------- it didn't work :( ----------"
    chart.scribble(output)
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
  ao = ActiveObject('dill_in_chart')
  ao.live_spy = True
  ao.start_at(Start)
  ao.post_fifo(Event(signal=signals.build_payload_and_trans))
  time.sleep(0.1)

