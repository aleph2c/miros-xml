import re
import dill
import time
import logging
from pathlib import Path
from functools import wraps
from functools import partial
from functools import lru_cache
from collections import namedtuple
from collections import OrderedDict

from miros import Event
from miros import spy_on
from miros import signals
from miros import ActiveObject
from miros import return_status
from miros import HsmWithQueues

def p_spy_on(fn):
  '''wrapper for the parallel regions states
    **Note**:
       It hide any hidden state from appearing in the instrumentation
    **Args**:
       | ``fn`` (function): the state function
    **Returns**:
       (function): wrapped function
    **Example(s)**:
      
    .. code-block:: python
       
       @p_spy_on
       def example(p, e):
        status = return_status.UNHANDLED
        return status
  '''
  @wraps(fn)
  def _pspy_on(chart, *args):
    if chart.instrumented:
      status = spy_on(fn)(chart, *args)
      for line in list(chart.rtc.spy):
        m1 = re.search(r'hidden_region', str(line))
        m2 = re.search(r'START', str(line))
        if not m1 and not m2:
          chart.outer.live_spy_callback(
            "{}::{}".format(chart.name, line))
      chart.rtc.spy.clear()
    else:
      e = args[0] if len(args) == 1 else args[-1]
      status = fn(chart, e)
    return status
  return _pspy_on

class Region(HsmWithQueues):
  def __init__(self, name, starting_state, outer, final_event, instrumented=True):
    super().__init__()
    self.name = name
    self.outer = outer
    self.final_event = final_event
    self.instrumented = instrumented
    self.starting_state = starting_state

    self.final = False
    self.regions = []

  def post_p_final_to_outer_if_ready(self):
    ready = False if self.regions is None and len(self.regions) < 1 else True
    for region in self.regions:
      ready &= True if region.final else False
    if ready:
      self.outer.post_fifo(self.final_event)

  @lru_cache(maxsize=32)
  def tockenize(self, signal_name):
    return set(signal_name.split("."))

  @lru_cache(maxsize=32)
  def token_match(self, resident, other):
    alien_set = self.tockenize(other)
    resident_set = self.tockenize(resident)
    result = True if len(resident_set.intersection(alien_set)) >= 1 else False
    return result

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

  def clear_log(self):
    with open(self.log_file, "w") as fp:
      fp.write("")

class Regions():
  '''Replaces long-winded boiler plate code like this:

    self.p_regions.append(
      Region(
        name='s1_r',
        starting_state=p_p_s1_r_hidden_region,
        outer=self,
        final_event=Event(signal=signals.p_final)
      )
    )
    self.p_regions.append(
      Region(
        name='s2_r',
        starting_state=p_s2_r_hidden_region,
        outer=self,
        final_event=Event(signal=signals.p_final)
      )
    )

    # link all regions together
    for region in self.p_regions:
      for _region in self.p_regions:
        region.regions.append(_region)

  With this:

    self.p_regions = Regions(name='p', outer=self).add('s1_r').add('s2_r').regions

  '''
  def __init__(self, name, outer):
    self.name = name
    self.outer = outer
    self._regions = []
    self.final_signal_name = name + "_final"

  def add(self, region_name):
    '''
    This code basically provides this feature:

      self.p_regions.append(
        Region(
          name='s2_r',
          starting_state=p_s2_r_hidden_region,
          outer=self,
          final_event=Event(signal=signals.p_final)
        )
      )
    Where to 'p_s2_r_hidden_region', 'p_final' are inferred based on conventions
    and outer was provided to the Regions __init__ method

    '''
    self._regions.append(
      Region(
        name=region_name,
        starting_state=eval(region_name+"_hidden_region"),
        outer=self.outer,
        final_event = Event(signal=self.final_signal_name)
      ) 
    )
    return self

  def link(self):
    '''This property provides this basic feature:

      # link all regions together
      for region in self.p_regions:
        for _region in self.p_regions:
          region.regions.append(_region)

    We want to link all of the different regions together after we have
    added the regions, so this is why we put it in the property getter.  A
    client will want to get the regions array after they have finished adding
    each region.
    '''

    #for region in self._regions:
    #  for _region in self._regions:
    #    region.regions.append(_region)
    for region in self._regions:
      for _region in self._regions:
        if  not _region in region.regions:
          region.regions.append(_region)
    return self

  def post_fifo(self, e):
    [region.post_fifo(e) for region in self._regions]
    [region.complete_circuit() for region in self._regions]

  def start(self):
    for region in self._regions:
      region.start_at(region.starting_state)

  @property
  def instrumented(self):
    instrumented = True
    for region in self._regions:
      instrumented &= region.instrumented
    return instrumented

  @instrumented.setter
  def instrumented(self, _bool):
    for region in self._regions:
      region.instrumented = _bool



STXRef = namedtuple('STXRef', ['send_id', 'thread_id'])

class ScxmlChart(InstrumentedActiveObject):
  def __init__(self, name, log_file, live_spy=None, live_trace=None):
    super().__init__(name, log_file)

    if not live_spy is None:
      self.live_spy = live_spy

    if not live_trace is None:
      self.live_trace = live_trace

    self.shot_lookup = {}
    self.regions = {}
    self.regions['p'] = Regions(name='p',
        outer=self).add('p_s1_r').add('p_s2_r').link()

  def start(self):
    #for region in self.p_regions:
    #  region.start_at(region.starting_state)
    if self.live_spy:
      self.regions['p'].instrumented = self.instrumented
    else:
      self.regions['p'].instrumented = False
    self.regions['p'].start()

    self.start_at(outer_state)

  def instrumented(self, _bool):
    super().instrumented = _bool
    self.region['p'].instrumented = _bool

  @lru_cache(maxsize=32)
  def tockenize(self, signal_name):
    return set(signal_name.split("."))

  @lru_cache(maxsize=32)
  def token_match(self, resident, other):
    alien_set = self.tockenize(other)
    resident_set = self.tockenize(resident)
    result = True if len(resident_set.intersection(alien_set)) >= 1 else False
    return result

  def post_fifo_with_sendid(self, sendid, e, period=None, times=None, deferred=None):
    thread_id = self.post_fifo(e, period, times, deferred)
    if thread_id is not None:
      self.shot_lookup[e.signal_name] = \
        STXRef(thread_id=thread_id,send_id=sendid)

  def post_lifo_with_sendid(self, sendid, e, period=None, times=None, deferred=None):
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

@p_spy_on
def p_s1_r_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(r.token_match(e.signal_name, "enter_region")):
    status = r.trans(p_s1_region)
  else:
    r.temp.fun = r.top
    status = return_status.SUPER
  return status

@p_spy_on
def p_s1_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    status = r.trans(p_s11)
  elif(r.token_match(e.signal_name, "region_exit")):
    status = r.trans(p_s1_r_hidden_region)
  else:
    r.temp.fun = p_s1_r_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_s11(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name,"e4")):
    status = r.trans(p_s12)
  elif(e.signal == signals.EXIT_SIGNAL):
    status = return_status.HANDLED
  else:
    r.temp.fun = p_s1_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_s12(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e1")):
    status = r.trans(p_s1_region_final)
  elif(e.signal == signals.EXIT_SIGNAL):
    status = return_status.HANDLED
  else:
    r.temp.fun = p_s1_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_s1_region_final(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    r.final = True
    r.post_p_final_to_outer_if_ready()
  elif(e.signal == signals.ENTRY_SIGNAL):
    r.final = False
  else:
    r.temp.fun = p_s1_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_s2_r_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(r.token_match(e.signal_name, "enter_region")):
    status = r.trans(p_s2_region)
  else:
    r.temp.fun = r.top
    status = return_status.SUPER
  return status

@p_spy_on
def p_s2_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    status = r.trans(p_s21)
  elif(e.signal == signals.EXIT_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.region_exit):
    status = r.trans(p_s2_r_hidden_region)
  else:
    r.temp.fun = p_s2_r_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_s21(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name,"e1")):
    status = r.trans(p_s22)
  elif(e.signal == signals.EXIT_SIGNAL):
    status = return_status.HANDLED
  else:
    r.temp.fun = p_s2_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_s22(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e2")):
    status = r.trans(p_s2_region_final)
  elif(e.signal == signals.EXIT_SIGNAL):
    status = return_status.HANDLED
  else:
    r.temp.fun = p_s2_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_s2_region_final(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    r.final = True
    r.post_p_final_to_outer_if_ready()
  elif(e.signal == signals.ENTRY_SIGNAL):
    r.final = False
  else:
    r.temp.fun = p_s2_region
    status = return_status.SUPER
  return status

@spy_on
def outer_state(self, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(self.token_match(e.signal_name, "to_p")):
    status = self.trans(p)
  else:
    self.temp.fun = self.top
    status = return_status.SUPER
  return status

@spy_on
def some_other_state(self, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  else:
    self.temp.fun = outer_state
    status = return_status.SUPER
  return status

@spy_on
def p(self, e):
  status = return_status.UNHANDLED
  # enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    if self.live_spy and self.instrumented:
      self.live_spy_callback("{}:p".format(e.signal_name))
    self.regions['p'].post_fifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  # any event handled within there regions must be pushed from here
  elif(self.token_match(e.signal_name, "e1") or
      self.token_match(e.signal_name, "e2") or 
      self.token_match(e.signal_name, "e4")
      ):
    if self.live_spy and self.instrumented:
      self.live_spy_callback("{}:p".format(e.signal_name))
    self.regions['p'].post_fifo(e)
    status = return_status.HANDLED
  elif(self.token_match(e.signal_name, self.regions['p'].final_signal_name)):
    status = self.trans(some_other_state)
  elif(self.token_match(e.signal_name, "to_outer")):
    status = self.trans(outer_state)
  else:
    self.temp.fun = outer_state
    status = return_status.SUPER
  return status

if __name__ == '__main__':
  example = ScxmlChart(
    name='parallel',
    log_file="/mnt/c/github/xml/experiment/parallel_example_2.log",
    live_trace=True,
    live_spy=True,
  )
  #example.instrumented = True
  example.start()
  example.post_fifo(Event(signal=signals.to_p))
  example.post_fifo(Event(signal=signals.e4))
  example.post_fifo(Event(signal=signals.e1))
  example.post_fifo(Event(signal=signals.to_outer))
  example.post_fifo(Event(signal=signals.to_p))
  example.post_fifo(Event(signal=signals.e4))
  example.post_fifo(Event(signal=signals.e1))
  example.post_fifo(Event(signal=signals.e2))
  time.sleep(0.1)


