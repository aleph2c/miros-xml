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
from miros import ActiveObject
from miros import return_status
"""
<!--
Adjusted from: test208 of the SCXML spec
-->
<scxml initial="s0" version="1.0">
  <state id="s0">
    <onentry>
      <send id="foo" event="event1" delay="1"/>
      <send event="event2" delay="1.5"/>
      <cancel sendid="foo"/>
    </onentry>
    <transition event="event2" target"_pass"/>
    <transition event="*" target="_fail"/>
  </state>
  <state id="_pass">
  <state id="_fail">
</scxml>
"""

@spy_on
def s0(self, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    self.post_fifo_with_sendid(Event(signal="timeout.banana"),
      sendid='foo',
      times=1,
      period=3.0,
      deferred=True)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    status = self.trans(s01)
  elif(self.token_match(e.signal_name, 'timeout')):
    status = self.trans(_fail)
  elif(self.token_match(e.signal_name, 'event1')):
    status = self.trans(_fail)
  elif(self.token_match(e.signal_name, 'event2')):
    status = self.trans(_pass)
  elif(e.signal==signals.EXIT_SIGNAL):
    #self.cancel_all(Event(signal='bob'))
    self.cancel_with_sendid(sendid="foo")
    status = return_status.HANDLED
  else: 
    self.temp.fun = self.top
    status = return_status.SUPER
  return status

@spy_on
def s01(self, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    self.post_fifo(Event(signal="event1"),
      period=2.0,
      times=1,
      deferred=True)
    status = return_status.HANDLED
  elif(self.token_match(e.signal_name, 'event1')):
    status = self.trans(s02)
  elif(signals.is_inner_signal(e.signal)):
    self.temp.fun = s0
    status = return_status.SUPER
  else: # "*"
    status = self.trans(_fail)
  return status

@spy_on
def s02(self, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    self.post_fifo(Event(signal="event2.token5"),
      times=1,
      period=5.0,
      deferred=True)
    status = return_status.HANDLED
  elif(self.token_match(e.signal_name, 'event2')):
    status = self.trans(_pass)
  elif(e.signal == signals.INIT_SIGNAL):
    status = return_status.HANDLED
  elif(self.token_match(e.signal_name, 'timeout')):
    status = self.trans(_fail)
  else:
    self.temp.fun = self.top
    status = return_status.SUPER
  return status

@spy_on
def _pass(self, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    self.scribble("\n|--> PASS!!! <--|\n")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    status = return_status.HANDLED
  else:
    self.temp.fun = self.top
    status = return_status.SUPER
  return status

@spy_on
def _fail(self, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    self.scribble("\n|--> FAIL!!! <--|\n")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    status = return_status.HANDLED
  else:
    self.temp.fun = self.top
    status = return_status.SUPER
  return status

class InstrumentedActiveObject(ActiveObject):
  def __init__(self, name, log_file):
    super().__init__(name)

    self.log_file = log_file

    logging.basicConfig(
      format='%(asctime)s %(levelname)s:%(message)s',
      filemode='w',
      filename=self.log_file,
      level=logging.DEBUG)
    self.register_live_spy_callback(partial(self.spy_callback))
    self.register_live_trace_callback(partial(self.trace_callback))

  def trace_callback(self, trace):
    '''trace without datetimestamp'''
    trace_without_datetime = re.search(r'(\[.+\]) (\[.+\].+)', trace).group(2)
    print(trace_without_datetime)
    logging.debug("T: " + trace_without_datetime)

  def spy_callback(self, spy):
    '''spy with machine name pre-pending'''
    print(spy)
    logging.debug("S: [%s] %s" % (self.name, spy))

STXRef = namedtuple('SendPostCrossReference', ['send_id', 'thread_id'])

class ScxmlChart(InstrumentedActiveObject):
  def __init__(self, name, log_file):
    super().__init__(name, log_file)
    self.shot_lookup = {}

  def start(self):
    self.start_at(s0)

  @lru_cache(maxsize=32)
  def tockenize(self, signal_name):
    return set(signal_name.split("."))

  @lru_cache(maxsize=32)
  def token_match(self, resident, other):
    alien_set = self.tockenize(other)
    resident_set = self.tockenize(resident)
    result = True if len(resident_set.intersection(alien_set)) >= 1 else False
    return result

  def post_fifo_with_sendid(self, e, sendid, period=None, times=None, deferred=None):
    thread_id = self.post_fifo(e, period, times, deferred)
    if thread_id is not None:
      self.shot_lookup[e.signal_name] = \
        STXRef(thread_id=thread_id,send_id=sendid)

  def post_lifo_with_sendid(self, e, sendid, period=None, times=None, deferred=None):
    thread_id = super().post_lifo(e, period, times, deferred)
    if thread_id is not None:
      self.shot_lookup[e.signal_name] = \
        STXRef(thread_id=thread_id,send_id=sendid)

  def cancel_with_sendid(self, sendid):
    thread_id = None
    for k, v in self.shot_lookup.items():
      if v.send_id == sendid:
        thread_id = v.thread_id
        break
    if thread_id is not None:
      self.cancel_event(thread_id)

  def cancel_all(self, e):
    token = e.signal_name
    for k, v in self.shot_lookup.items():
      if self.token_match(token, k):
        self.cancel_events(Event(signal=k))
        break

if __name__ == '__main__':
  ao = ScxmlChart('test403', "/mnt/c/github/xml/experiment/test_403.log")
  ao.live_spy = True
  ao.start()
  #ao.post_fifo(Event(signal=signals.event4))
  time.sleep(100.01)
  assert ao.token_match("bob.mary.john", "john.murphy.muff")

