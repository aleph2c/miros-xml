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
Adjusted from:
https://github.com/alexzhornyak/SCXML-tutorial/blob/master/Doc/transition.md
-->
<scxml xmlns="http://www.w3.org/2005/07/scxml" initial="s0" version="1.0">
  <state id="s0" initial="s01">
    <onentry>
      <!--  catch the failure case  -->
      <send event="timeout.token1.token2" delay="1s"/>
    </onentry>
    <transition event="timeout" target="_fail"/>
    <transition event="event1" target="_fail"/>
    <transition event="event2" target="_pass"/>
    <state id="s01">
      <onentry>
        <!--
          this should be caught by the first transition in this state, taking us to S02 
        -->
        <raise event="token3.event1.token4"/>
      </onentry>
      <transition event="event1" target="s02"/>
      <transition event="*" target="_fail"/>
    </state>
    <state id="s02">
      <onentry>
        <!--
          since the local transition has a cond that evaluates to false this
          should be caught by a transition in the parent state, taking us to
          pass 
        -->
        <raise event="event2.token5"/>
      </onentry>
      <transition event="event1" target="_pass"/>
      <transition event="event2" cond="false" target="_fail"/>
    </state>
    <state id="_pass">
      <onentry>
        <log expr="\n|--> PASS!!! <--|\n"/>
      </onentry>
    </state>
    <state id="_fail">
      <onentry>
        <log expr="\n|--> FAIL!!! <--|\n"/>
      </onentry>
    </state>
  </state>
</scxml>
"""

@spy_on
def s0(self, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    self.post_fifo(Event(signal="timeout.token1.token2"),
      times=1,
      period=1.0,
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
  else: 
    self.temp.fun = self.top
    status = return_status.SUPER
  return status

@spy_on
def s01(self, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    self.post_fifo(Event(signal="token3.event1.token4"))
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
    self.post_fifo(Event(signal="event2.token5"))
    status = return_status.HANDLED
  elif(self.token_match(e.signal_name, 'event1')):
    status = self.trans(_pass)
  elif(self.token_match(e.signal_name, 'event2')):
    if False:
      status = self.trans(_fail)
  elif(e.signal == signals.INIT_SIGNAL):
    status = return_status.HANDLED
  else:
    self.temp.fun = s0
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

class ScxmlChart(InstrumentedActiveObject):
  def __init__(self, name, log_file):
    super().__init__(name, log_file)

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

if __name__ == '__main__':
  ao = ScxmlChart('test403', "/mnt/c/github/xml/experiment/test_403.log")
  ao.live_spy = True
  ao.start()
  ao.post_fifo(Event(signal=signals.event4))
  time.sleep(1.01)
  assert ao.token_match("bob.mary.john", "john.murphy.muff")

