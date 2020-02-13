import re
import time
import logging
from functools import wraps
from functools import partial
from functools import lru_cache
from collections import namedtuple

from miros import Event
from miros import spy_on
from miros import signals
from miros import ActiveObject
from miros import return_status
from miros import HsmWithQueues


META_SIGNAL_PAYLOAD = namedtuple("META_SIGNAL_PAYLOAD", ['event', 'state', 'source_event', 'region'])
import pprint
def pp(item):
  pprint.pprint(item)

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
        m1 = re.search(r'.*hidden_region', str(line))
        m2 = re.search(r'START|SEARCH_FOR_SUPER_SIGNAL|region_exit', str(line))
        if not m1 and not m2:
          chart.outmost.live_spy_callback(
            "{}::{}".format(chart.name, line))
      chart.rtc.spy.clear()
    else:
      e = args[0] if len(args) == 1 else args[-1]
      status = fn(chart, e)
    return status
  return _pspy_on

Reflections = []

class Region(HsmWithQueues):
  def __init__(self, name, starting_state, outmost, final_event, instrumented=True, outer=None):
    super().__init__()
    self.name = name
    self.outmost = outmost
    self.final_event = final_event
    self.instrumented = instrumented
    self.starting_state = starting_state
    self.bottom = self.top
    self.outer = outer

    self.final = False
    self.regions = []

  def post_p_final_to_outmost_if_ready(self):
    ready = False if self.regions is None and len(self.regions) < 1 else True
    for region in self.regions:
      ready &= True if region.final else False
    if ready:
      self.outmost.post_fifo(self.final_event)

  @lru_cache(maxsize=32)
  def tockenize(self, signal_name):
    return set(signal_name.split("."))

  @lru_cache(maxsize=32)
  def token_match(self, resident, other):
    alien_set = self.tockenize(other)
    resident_set = self.tockenize(resident)
    result = True if len(resident_set.intersection(alien_set)) >= 1 else False
    return result

  def init_stack(self, e):
    result = (None, None)
    if len(self.queue) >= 1 and \
      self.queue[0].signal == signals.INIT_META:
      _e = self.queue.popleft()
      result = (_e.payload.event, _e.payload.state)
    return result

  def has_a_child(self, fn_state_handler):
    old_temp = self.temp.fun
    old_fun = self.state.fun

    current_state = self.temp.fun
    self.temp.fun = fn_state_handler

    result = False
    super_e = Event(signal=signals.SEARCH_FOR_SUPER_SIGNAL)
    while(True):
      if(self.temp.fun == current_state):
        result = True
        r = return_status.IGNORED
      else:
        r = self.temp.fun(self, super_e)
      if r == return_status.IGNORED:
        break;
    self.temp.fun = old_temp
    self.state.fun = old_fun
    return result

  def get_region(self, fun=None):

    if fun is None:
      current_state = self.temp.fun
    else:
      current_state = fun

    old_temp = self.temp.fun
    old_fun = self.state.fun

    self.temp.fun = current_state

    result = ''
    super_e = Event(signal=signals.SEARCH_FOR_SUPER_SIGNAL)
    while(True):
      if 'region' in self.temp.fun.__name__:
        result = self.temp.fun.__name__
        r = return_status.IGNORED
      elif 'top' in self.temp.fun.__name__:
        r = return_status.IGNORED
      else:
        r = self.temp.fun(self, super_e)
      if r == return_status.IGNORED:
        break;
    self.temp.fun = old_temp
    self.state.fun = old_fun
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
    #self.print(trace_without_datetime)
    logging.debug("T: " + trace_without_datetime)

  def spy_callback(self, spy):
    '''spy with machine name pre-pending'''
    #self.print(spy)
    logging.debug("S: [%s] %s" % (self.name, spy))

  def clear_log(self):
    with open(self.log_file, "w") as fp:
      fp.write("")

class Regions():
  '''Replaces long-winded boiler plate code like this:

    self.p_regions.append(
      Region(
        name='s1_r',
        starting_state=p_r2_under_hidden_region,
        outmost=self,
        final_event=Event(signal=signals.p_final)
      )
    )
    self.p_regions.append(
      Region(
        name='s2_r',
        starting_state=p_r2_under_hidden_region,
        outmost=self,
        final_event=Event(signal=signals.p_final)
      )
    )

    # link all regions together
    for region in self.p_regions:
      for _region in self.p_regions:
        region.regions.append(_region)

  With this:

    self.p_regions = Regions(name='p', outmost=self).add('s1_r').add('s2_r').regions

  '''
  def __init__(self, name, outmost):
    self.name = name
    self.outmost = outmost
    self._regions = []
    self.final_signal_name = name + "_final"

  def add(self, region_name, outer=None):
    '''
    This code basically provides this feature:

      self.p_regions.append(
        Region(
          name='s2_r',
          starting_state=p_r2_under_hidden_region,
          outmost=self,
          final_event=Event(signal=signals.p_final)
          outer=self,
        )
      )
    Where to 'p_r2_under_hidden_region', 'p_final' are inferred based on conventions
    and outmost was provided to the Regions __init__ method and 'outer' is needed
    for the META_EXIT signal.

    '''
    self._regions.append(
      Region(
        name=region_name,
        starting_state=eval(region_name+"_under_hidden_region"),
        outmost=self.outmost,
        final_event = Event(signal=self.final_signal_name),
        outer=outer
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
    self._post_fifo(e)
    [region.complete_circuit() for region in self._regions]

  def _post_fifo(self, e):
    [region.post_fifo(e) for region in self._regions]

  def post_lifo(self, e):
    self._post_lifo(e)
    [region.complete_circuit() for region in self._regions]

  def _post_lifo(self, e):
    [region.post_lifo(e) for region in self._regions]

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

  def region(self, name):
    result = None
    for region in self._regions:
      if name == region.name:
        result = region
        break
    return result

STXRef = namedtuple('STXRef', ['send_id', 'thread_id'])

class ScxmlChart(InstrumentedActiveObject):
  def __init__(self, name, log_file, live_spy=None, live_trace=None):
    super().__init__(name, log_file)

    if not live_spy is None:
      self.live_spy = live_spy

    if not live_trace is None:
      self.live_trace = live_trace

    self.bottom = self.top

    self.shot_lookup = {}
    self.regions = {}

    # TODO: once you have a minimal viable chart, look into this:
    # Here you have messed up your terminalology, a region means something
    # within a dotted line, but this construct, it means the outer orthogonal
    # component.
    outer = self
    self.regions['p'] = Regions(
      name='p',
      outmost=self)\
    .add('p_r1', outer=outer)\
    .add('p_r2', outer=outer).link()

    outer = self.regions['p']
    self.regions['p_p11'] = Regions(
      name='p_p11',
      outmost=self)\
    .add('p_p11_r1', outer=outer)\
    .add('p_p11_r2', outer=outer).link()

    outer = self.regions['p']
    self.regions['p_p12'] = Regions(
      name='p_p12',
      outmost=self)\
    .add('p_p12_r1', outer=outer)\
    .add('p_p12_r2', outer=outer).link()

    outer = self.regions['p_p12']
    self.regions['p_p12_p11'] = Regions(
      name='p_p12_p11',
      outmost=self)\
    .add('p_p12_p11_r1', outer=outer)\
    .add('p_p12_p11_r2', outer=outer).link()

    outer = self.regions['p']
    self.regions['p_p22'] = Regions(
      name='p_p22',
      outmost=self)\
    .add('p_p22_r1', outer=outer)\
    .add('p_p22_r2', outer=outer).link()

  def start(self):
    if self.live_spy:
      for key in self.regions.keys():
        self.regions[key].instrumented = self.instrumented
    else:
      for key in self.regions.keys():
        self.regions[key].instrumented = False

    for key in self.regions.keys():
      self.regions[key].start()

    self.start_at(outer_state)

  def instrumented(self, _bool):
    super().instrumented = _bool
    for key in self.regions.keys():
      self.regions[key].instrumented = _bool

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

  def init_stack(self, e):
    result = (None, None)
    if len(self.queue.deque) >= 1 and \
      self.queue.deque[0].signal == signals.INIT_META:
      _e = self.queue.deque.popleft()
      result = (_e.payload.event, _e.payload.state)
    return result

  def active_states(self):

    parallel_state_names = self.regions.keys()

    def recursive_get_states(name):
      states = []
      if name in parallel_state_names:
        for region in self.regions[name]._regions:
          if region.state_name in parallel_state_names:
            _states = recursive_get_states(region.state_name)
            states.append(_states)
          else:
            states.append(region.state_name)
      else:
        states.append(self.state_name)
      return states

    states = recursive_get_states(self.state_name)
    return states



################################################################################
#                                   p region                                   #
################################################################################
# * define hidden for each region in p
# * define region state for each region in p
# * define all substates
# * define event horizon (def p)
# * in ScxmlChart add a region
# * in ScxmlChart start, add start to region
# * figure out the exits
@p_spy_on
def p_r1_under_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(r.token_match(e.signal_name, "enter_region")):
    status = r.trans(p_r1_region)
  else:
    r.temp.fun = r.top
    status = return_status.SUPER
  return status

@p_spy_on
def p_r1_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = r.init_stack(e) # search for INIT_META
    # if _state is a child of this state then transition to it

    if _state is None or not r.has_a_child(_state):
      status = r.trans(p_p11)
    else:
      status = r.trans(_state)
      if not _e is None:
        r.post_fifo(_e)

  elif(e.signal == signals.region_exit):
    status = r.trans(p_r1_under_hidden_region)
  elif(e.signal == signals.INIT_META):
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r1_under_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_r1_over_hidden_type(r, e):
  status = return_status.UNHANDLED
  if(e.signal==signals.force_region_init):
    status = r.trans(p_r1_region)
  else:
    r.temp.fun = p_r1_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11(r, e):
  outmost = r.outmost
  status = return_status.UNHANDLED
  # enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    if outmost.live_spy and outmost.instrumented:
      outmost.live_spy_callback("{}:p_p11".format(e.signal_name))
    (_e, _state) = r.init_stack(e) # search for INIT_META
    if _state:
      outmost.regions['p_p11']._post_fifo(_e)
    outmost.regions['p_p11'].post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  # any event handled within there regions must be pushed from here
  elif(outmost.token_match(e.signal_name, "e1") or
       outmost.token_match(e.signal_name, "e2") or
       outmost.token_match(e.signal_name, "e4") or
       outmost.token_match(e.signal_name, "A") or
       outmost.token_match(e.signal_name, "F1") or
       outmost.token_match(e.signal_name, "G3")
      ):
    if outmost.live_spy and outmost.instrumented:
      outmost.live_spy_callback("{}:p_p11".format(e.signal_name))
      outmost.regions['p_p11'].post_fifo(e)
      status = return_status.HANDLED
  elif(outmost.token_match(e.signal_name, outmost.regions['p_p11'].final_signal_name)):
    if outmost.live_spy and outmost.instrumented:
      outmost.live_spy_callback("{}:p_p11".format(e.signal_name))
    status = r.trans(p_p12)
  elif outmost.token_match(e.signal_name, "C0"):
    status = r.trans(p_p12)
  elif(e.signal == signals.META_EXIT):
    region1 = r.get_region()
    region2 = r.get_region(e.payload.state)
    if region1 == region2:
      status = r.trans(e.payload.state)
    else:
      status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL or e.signal == signals.region_exit):
    if outmost.live_spy and outmost.instrumented:
      outmost.live_spy_callback("{}:p_p11".format(Event(signal=signals.region_exit)))
    outmost.regions['p_p11'].post_lifo(Event(signal=signals.region_exit))
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r1_over_hidden_type
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_r1_under_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(rr.token_match(e.signal_name, "enter_region")):
    status = rr.trans(p_p11_r1_region)
  else:
    rr.temp.fun = rr.top
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_r1_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = rr.init_stack(e) # search for INIT_META
    # if _state is a child of this state then transition to it
    if _state is None or not rr.has_a_child(_state):
      status = rr.trans(p_p11_s11)
    else:
      status = rr.trans(_state)
      if not _e is None:
        rr.post_fifo(_e)
  elif(e.signal == signals.region_exit):
    status = rr.trans(p_p11_r1_under_hidden_region)
  elif(e.signal == signals.INIT_META):
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p11_r1_under_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_r1_over_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal==signals.force_region_init):
    status = rr.trans(p_p11_r1_region)
  else:
    rr.temp.fun = p_p11_r1_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_s11(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e4")):
    status = rr.trans(p_p11_s12)
  else:
    rr.temp.fun = p_p11_r1_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_s12(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e1")):
    status = rr.trans(p_p11_r1_final)
  else:
    rr.temp.fun = p_p11_r1_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_r1_final(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    rr.final = True
    rr.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    rr.final = False
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p11_r1_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_r2_under_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(rr.token_match(e.signal_name, "enter_region")):
    status = rr.trans(p_p11_r2_region)
  else:
    rr.temp.fun = rr.top
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_r2_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = rr.init_stack(e) # search for INIT_META
    # if _state is a child of this state then transition to it
    if _state is None or not rr.has_a_child(_state):
      status = rr.trans(p_p11_s21)
    else:
      status = rr.trans(_state)
      if not _e is None:
        rr.post_fifo(_e)
  else:
    rr.temp.fun = p_p11_r2_under_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_r2_over_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal==signals.force_region_init):
    status = rr.trans(p_p11_r2_region)
  else:
    rr.temp.fun = p_p11_r2_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_s21(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e1")):
    status = rr.trans(p_p11_s22)
  else:
    rr.temp.fun = p_p11_r2_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_s22(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e2")):
    status = rr.trans(p_p11_r2_final)
  elif(rr.token_match(e.signal_name, "e1")):
    status = rr.trans(p_p11_s21)
  else:
    rr.temp.fun = p_p11_r2_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_r2_final(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    rr.final = True
    rr.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    rr.final = False
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p11_r2_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_r1_final(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    r.final = True
    r.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r.final = False
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r1_over_hidden_type
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12(r, e):
  outmost = r.outmost
  status = return_status.UNHANDLED
  # enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    if outmost.live_spy and outmost.instrumented:
      outmost.live_spy_callback("{}:p_p12".format(e.signal_name))
    (_e, _state) = r.init_stack(e) # search for INIT_META
    if _state:
      outmost.regions['p_p12']._post_fifo(_e)
    outmost.regions['p_p12'].post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  # any event handled within there regions must be pushed from here
  elif(outmost.token_match(e.signal_name, "e1") or
      outmost.token_match(e.signal_name, "e2") or
      outmost.token_match(e.signal_name, "e4")
      #outmost.token_match(e.signal_name, "G3")
      ):
    if outmost.live_spy and outmost.instrumented:
      outmost.live_spy_callback("{}:p_p12".format(e.signal_name))
    outmost.regions['p_p12'].post_fifo(e)
    status = return_status.HANDLED
  # final token match
  elif(outmost.token_match(e.signal_name, outmost.regions['p_p12'].final_signal_name)):
    if outmost.live_spy and outmost.instrumented:
      outmost.live_spy_callback("{}:p_p12".format(e.signal_name))
    status = r.trans(p_r1_final)
  elif(outmost.token_match(e.signal_name, "e5")):
    status = r.trans(p_r1_final)
  elif(e.signal == signals.META_EXIT):
    region1 = r.get_region()
    region2 = r.get_region(e.payload.state)
    if region1 == region2:
      status = r.trans(e.payload.state)
    else:
      status = return_status.HANDLED
  # exit signals
  elif(e.signal == signals.EXIT_SIGNAL or e.signal == signals.region_exit):
    if outmost.live_spy and outmost.instrumented:
      outmost.live_spy_callback("{}:p_p12".format(Event(signal=signals.region_exit)))
    outmost.regions['p_p12'].post_lifo(Event(signal=signals.region_exit))
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r1_over_hidden_type
    status = return_status.SUPER
  return status

# inner parallel
@p_spy_on
def p_p12_r1_under_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(rr.token_match(e.signal_name, "enter_region")):
    status = rr.trans(p_p12_r1_region)
  else:
    rr.temp.fun = rr.top
    status = return_status.SUPER
  return status

# inner parallel
@p_spy_on
def p_p12_r1_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = rr.init_stack(e) # search for INIT_META
    # if _state is a child of this state then transition to it
    if _state is None or not rr.has_a_child(_state):
      status = rr.trans(p_p12_p11)
    else:
      status = rr.trans(_state)
      if not _e is None:
        rr.post_fifo(_e)
  # can we get rid of region_exit?
  elif(e.signal == signals.region_exit):
    status = rr.trans(p_p12_r1_under_hidden_region)
  elif(e.signal == signals.INIT_META):
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p12_r1_under_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_r1_over_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal==signals.force_region_init):
    status = rr.trans(p_p12_r1_region)
  else:
    rr.temp.fun = p_p12_r1_region
    status = return_status.SUPER
  return status


# inner parallel
@p_spy_on
def p_p12_p11(rr, e):
  outmost = rr.outmost
  status = return_status.UNHANDLED
  # enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    if outmost.live_spy and outmost.instrumented:
      outmost.live_spy_callback("{}:p_p12_s11".format(e.signal_name))
    (_e, _state) = rr.init_stack(e) # search for INIT_META
    if _state:
      outmost.regions['p_p12_p11']._post_fifo(_e)
    outmost.regions['p_p12_p11'].post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  # any event handled within there regions must be pushed from here
  elif(outmost.token_match(e.signal_name, "G1") or
       outmost.token_match(e.signal_name, outmost.regions['p_p12_p11'].final_signal_name)
      ):
    if outmost.live_spy and outmost.instrumented:
      outmost.live_spy_callback("{}:p_p12_p11".format(e.signal_name))
    outmost.regions['p_p12_p11'].post_fifo(e)
    status = return_status.HANDLED
  elif(outmost.token_match(e.signal_name, outmost.regions['p_p12_p11'].final_signal_name)):
    if outmost.live_spy and outmost.instrumented:
      outmost.live_spy_callback("{}:p_p12_p11".format(e.signal_name))
    status = rr.trans(p_p12_s12)
  elif(rr.token_match(e.signal_name, "e4")):
    status = rr.trans(p_p12_s12)
  else:
    rr.temp.fun = p_p12_r1_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_p11_r1_under_hidden_region(rrr, e):
  status = return_status.UNHANDLED
  if(rrr.token_match(e.signal_name, "enter_region")):
    status = rrr.trans(p_p12_p11_r1_region)
  else:
    rrr.temp.fun = rrr.top
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_p11_r1_region(rrr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = rrr.init_stack(e) # search for INIT_META
    # if _state is a child of this state then transition to it
    if _state is None or not rrr.has_a_child(_state):
      status = rrr.trans(p_p12_p11_s11)
    else:
      status = rrr.trans(_state)
      if not _e is None:
        rrr.post_fifo(_e)
  # can we get rid of region_exit?
  elif(e.signal == signals.region_exit):
    status = rrr.trans(p_p12_p11_r1_under_hidden_region)
  elif(e.signal == signals.INIT_META):
    status = return_status.HANDLED
  else:
    rrr.temp.fun = p_p12_p11_r1_under_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_p11_r1_over_hidden_region(rrr, e):
  status = return_status.UNHANDLED
  if(e.signal==signals.force_region_init):
    status = rrr.trans(p_p12_p11_r1_region)
  else:
    rrr.temp.fun = p_p12_p11_r1_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_p11_s11(rrr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  else:
    rrr.temp.fun = p_p12_p11_r1_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_p11_r2_under_hidden_region(rrr, e):
  status = return_status.UNHANDLED
  if(rrr.token_match(e.signal_name, "enter_region")):
    status = rrr.trans(p_p12_p11_r2_region)
  else:
    rrr.temp.fun = rrr.top
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_p11_r2_region(rrr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = rrr.init_stack(e) # search for INIT_META
    # if _state is a child of this state then transition to it
    if _state is None or not rrr.has_a_child(_state):
      status = rrr.trans(p_p12_p11_s21)
    else:
      status = rrr.trans(_state)
      if not _e is None:
        rrr.post_fifo(_e)
  # can we get rid of region_exit?
  elif(e.signal == signals.region_exit):
    status = rrr.trans(p_p12_p11_r2_under_hidden_region)
  elif(e.signal == signals.INIT_META):
    status = return_status.HANDLED
  else:
    rrr.temp.fun = p_p12_p11_r2_under_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_p11_r2_over_hidden_region(rrr, e):
  status = return_status.UNHANDLED
  if(e.signal==signals.force_region_init):
    status = rrr.trans(p_p12_p11_r2_region)
  else:
    rrr.temp.fun = p_p12_p11_r2_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_p11_s21(rrr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  else:
    rrr.temp.fun = p_p12_p11_r2_over_hidden_region
    status = return_status.SUPER
  return status

# inner parallel
@p_spy_on
def p_p12_s12(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e1")):
    status = rr.trans(p_p12_r1_final)
  else:
    rr.temp.fun = p_p12_r1_over_hidden_region
    status = return_status.SUPER
  return status

# inner parallel
@p_spy_on
def p_p12_r1_final(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    rr.final = True
    rr.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    rr.final = False
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p12_r1_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_r2_under_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(rr.token_match(e.signal_name, "enter_region")):
    status = rr.trans(p_p12_r2_region)
  else:
    rr.temp.fun = rr.top
    status = return_status.SUPER
  return status

# inner parallel
@p_spy_on
def p_p12_r2_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = rr.init_stack(e) # search for INIT_META
    # if _state is a child of this state then transition to it
    if _state is None or not rr.has_a_child(_state):
      status = rr.trans(p_p12_s21)
    else:
      status = rr.trans(_state)
      if not _e is None:
        rr.post_fifo(_e)
  # can we get rid of region_exit?
  elif(e.signal == signals.region_exit):
    status = rr.trans(p_p12_r2_under_hidden_region)
  elif(e.signal == signals.INIT_META):
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p12_r2_under_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_r2_over_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal==signals.force_region_init):
    status = rr.trans(p_p12_r2_region)
  else:
    rr.temp.fun = p_p12_r2_region
    status = return_status.SUPER
  return status

# inner parallel
@p_spy_on
def p_p12_s21(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  #elif(rr.token_match(e.signal_name, "G3")):
  #  source_event = e.signal_name
  #  rr.outer._post_lifo(
  #    Event(signal=signals.META_EXIT,
  #      payload=META_SIGNAL_PAYLOAD(
  #        event=None,
  #        state=p_s11,
  #        source_event=source_event,
  #        region=None,
  #      )
  #    )
  #  )
  #  rr.outmost.complete_circuit()
  #  #status = rr.trans(p_p12_r2_under_hidden_region)
  #  status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e1")):
    status = rr.trans(p_p12_s22)
  else:
    rr.temp.fun = p_p12_r2_over_hidden_region
    status = return_status.SUPER
  return status

# inner parallel
@p_spy_on
def p_p12_s22(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e2")):
    status = rr.trans(p_p12_r2_final)
  elif(rr.token_match(e.signal_name, "e1")):
    status = rr.trans(p_p12_s21)
  else:
    rr.temp.fun = p_p12_r2_over_hidden_region
    status = return_status.SUPER
  return status

# inner parallel
@p_spy_on
def p_p12_r2_final(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    rr.final = True
    rr.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    rr.final = False
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p12_r2_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_r2_under_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(r.token_match(e.signal_name, "enter_region")):
    status = r.trans(p_r2_region)
  else:
    r.temp.fun = r.top
    status = return_status.SUPER
  return status

@p_spy_on
def p_r2_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = r.init_stack(e) # search for INIT_META
    if _state is None or not r.has_a_child(_state):
      status = r.trans(p_s21)
    else:
      status = r.trans(_state)
      if not _e is None:
        r.post_fifo(_e)
  elif(e.signal == signals.EXIT_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.region_exit):
    status = r.trans(p_r2_under_hidden_region)
  elif(e.signal == signals.INIT_META):
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r2_under_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_r2_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal==signals.force_region_init):
    status = r.trans(p_r2_region)
  else:
    r.temp.fun = p_r2_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_s21(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name,"C0")):
    status = r.trans(p_p22)
  elif(e.signal == signals.EXIT_SIGNAL):
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r2_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22(r, e):
  outmost = r.outmost
  status = return_status.UNHANDLED
  # enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    if outmost.live_spy and outmost.instrumented:
      outmost.live_spy_callback("{}:p_p22".format(e.signal_name))
    (_e, _state) = r.init_stack(e) # search for INIT_META
    if _state:
      outmost.regions['p_p22']._post_fifo(_e)
    outmost.regions['p_p22'].post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  # any event handled within there regions must be pushed from here
  elif(outmost.token_match(e.signal_name, "e1") or
      outmost.token_match(e.signal_name, "e2") or
      outmost.token_match(e.signal_name, "e4")
      #outmost.token_match(e.signal_name, "G3")
      ):
    if outmost.live_spy and outmost.instrumented:
      outmost.live_spy_callback("{}:p_p22".format(e.signal_name))
    outmost.regions['p_p22'].post_fifo(e)
    status = return_status.HANDLED
  # final token match
  elif(outmost.token_match(e.signal_name, outmost.regions['p_p22'].final_signal_name)):
    if outmost.live_spy and outmost.instrumented:
      outmost.live_spy_callback("{}:p_p22".format(e.signal_name))
    status = r.trans(p_r2_final)
  elif(e.signal == signals.META_EXIT):
    region1 = r.get_region()
    region2 = r.get_region(e.payload.state)
    if region1 == region2:
      status = r.trans(e.payload.state)
    else:
      status = return_status.HANDLED
  # exit signals
  elif(e.signal == signals.EXIT_SIGNAL or e.signal == signals.region_exit):
    if outmost.live_spy and outmost.instrumented:
      outmost.live_spy_callback("{}:p_p22".format(Event(signal=signals.region_exit)))
    outmost.regions['p_p22'].post_lifo(Event(signal=signals.region_exit))
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r2_over_hidden_region
    status = return_status.SUPER
  return status

# inner parallel
@p_spy_on
def p_p22_r1_under_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(rr.token_match(e.signal_name, "enter_region")):
    status = rr.trans(p_p22_r1_region)
  else:
    rr.temp.fun = rr.top
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_r1_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = rr.init_stack(e) # search for INIT_META
    # if _state is a child of this state then transition to it
    if _state is None or not rr.has_a_child(_state):
      status = rr.trans(p_p22_s11)
    else:
      status = rr.trans(_state)
      if not _e is None:
        rr.post_fifo(_e)
  # can we get rid of region_exit?
  elif(e.signal == signals.region_exit):
    status = rr.trans(p_p22_r1_under_hidden_region)
  elif(e.signal == signals.INIT_META):
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p22_r1_under_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_r1_over_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal==signals.force_region_init):
    status = rr.trans(p_p22_r1_region)
  else:
    rr.temp.fun = p_p22_r1_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_s11(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e4")):
    status = rr.trans(p_p22_s12)
  else:
    rr.temp.fun = p_p22_r1_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_s12(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e1")):
    status = rr.trans(p_p22_r1_final)
  else:
    rr.temp.fun = p_p22_r1_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_r1_final(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    r.final = True
    r.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r.final = False
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p22_r1_over_hidden_region
    status = return_status.SUPER
  return status

# inner parallel
@p_spy_on
def p_p22_r2_under_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(rr.token_match(e.signal_name, "enter_region")):
    status = rr.trans(p_p22_r2_region)
  else:
    rr.temp.fun = rr.top
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_r2_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = rr.init_stack(e) # search for INIT_META
    # if _state is a child of this state then transition to it
    if _state is None or not rr.has_a_child(_state):
      status = rr.trans(p_p22_s21)
    else:
      status = rr.trans(_state)
      if not _e is None:
        rr.post_fifo(_e)
  # can we get rid of region_exit?
  elif(e.signal == signals.region_exit):
    status = rr.trans(p_p22_r2_under_hidden_region)
  elif(e.signal == signals.INIT_META):
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p22_r2_under_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_r2_over_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal==signals.force_region_init):
    status = rr.trans(p_p22_r2_region)
  else:
    rr.temp.fun = p_p22_r2_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_s21(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e1")):
    status = rr.trans(p_p22_s22)
  else:
    rr.temp.fun = p_p22_r2_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_s22(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e2")):
    status = rr.trans(p_p22_r2_final)
  else:
    rr.temp.fun = p_p22_r2_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_r2_final(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    r.final = True
    r.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r.final = False
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p22_r2_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_r2_final(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    r.final = True
    r.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r.final = False
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r2_over_hidden_region
    status = return_status.SUPER
  return status

@spy_on
def outer_state(self, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(self.token_match(e.signal_name, "to_p")):
    if self.live_spy and self.instrumented:
      self.live_spy_callback("{}:outer_state".format(e.signal_name))
    status = self.trans(p)
  elif(self.token_match(e.signal_name, "E0")):
    if self.live_spy and self.instrumented:
      self.live_spy_callback("{}:outer_state".format(e.signal_name))

    eeee = Event(
      signal=signals.INIT_META,
      payload=META_SIGNAL_PAYLOAD(
        event=None, state=p_p11_s21, source_event=e, region=None)
    )
    eee = Event(
      signal=signals.INIT_META,
      payload=META_SIGNAL_PAYLOAD(event=eeee, state="p_p11_s21", source_event=e,
        region=None)
    )
    ee = Event(
      signal=signals.INIT_META,
      payload=META_SIGNAL_PAYLOAD(event=eee, state=p_p11, source_event=e,
        region=None)
    )
    _e = Event(
      signal=signals.INIT_META,
      payload=META_SIGNAL_PAYLOAD(event=ee, state="p_p11", source_event=e,
        region=None)
    )
    self.post_fifo(_e)
    status = self.trans(p)
  #elif(self.token_match(e.signal_name, "E3")):
  #  if self.live_spy and self.instrumented:
  #    self.live_spy_callback("{}:outer_state".format(e.signal_name))

  #  eeee = Event(
  #    signal=signals.INIT_META,
  #    payload=META_SIGNAL_PAYLOAD(event=None, state=p_p12, source_event=e,
  #      region=None)
  #  )
  #  eee = Event(
  #    signal=signals.INIT_META,
  #    payload=META_SIGNAL_PAYLOAD(event=eeee, state="p_p12", source_event=e,
  #      region=None)
  #  )
  #  ee = Event(
  #    signal=signals.INIT_META,
  #    payload=META_SIGNAL_PAYLOAD(event=eee, state=p_p12, source_event=e,
  #      region=None)
  #  )
  #  _e = Event(
  #    signal=signals.INIT_META,
  #    payload=META_SIGNAL_PAYLOAD(event=ee, state="p", source_event=e,
  #      region=None)
  #  )
  #  self.post_fifo(_e)
  #  status = self.trans(p)
  else:
    self.temp.fun = self.bottom
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
    (_e, _state) = self.init_stack(e) # search for INIT_META
    if _state:
      self.regions['p']._post_fifo(_e)
    self.regions['p'].post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  # any event handled within there regions must be pushed from here
  elif(self.token_match(e.signal_name, "e1") or
      self.token_match(e.signal_name, "e2") or
      self.token_match(e.signal_name, "e3") or
      self.token_match(e.signal_name, "e4") or
      self.token_match(e.signal_name, "e5") or
      self.token_match(e.signal_name, "C0") or
      # self.token_match(e.signal_name, "G3") or
      self.token_match(e.signal_name, self.regions['p_p11'].final_signal_name) or
      self.token_match(e.signal_name, self.regions['p_p12'].final_signal_name) or
      self.token_match(e.signal_name, self.regions['p_p22'].final_signal_name)
      ):
    if self.live_spy and self.instrumented:
      self.live_spy_callback("{}:p".format(e.signal_name))
    self.regions['p'].post_fifo(e)
    status = return_status.HANDLED
  #elif(self.token_match(e.signal_name, "E1")):
  #  if self.live_spy and self.instrumented:
  #    self.live_spy_callback("{}:p".format(e.signal_name))

  #  # this breaks the code... p_r2 should not be re-initialized
  #  self.regions['p']._post_lifo(Event(signal=signals.force_region_init))
  #  #self.regions['p'].post_fifo(Event(signal=signals.region_exit))
  #  eee = Event(
  #    signal=signals.INIT_META,
  #    payload=META_SIGNAL_PAYLOAD(event=None, state=p_p12_s12, source_event=e,
  #      region=None)
  #  )

  #  ee = Event(
  #    signal=signals.INIT_META,
  #    payload=META_SIGNAL_PAYLOAD(event=eee, state="p_p12", source_event=e,
  #      region=None)
  #  )

  #  _e = Event(
  #    signal=signals.INIT_META,
  #    payload=META_SIGNAL_PAYLOAD(event=ee, state=p_p12, source_event=e,
  #      region=None)
  #  )

  #  self.regions['p'].post_fifo(_e)
  #  status = return_status.HANDLED
  # final token match
  elif(self.token_match(e.signal_name, self.regions['p'].final_signal_name)):
    if self.live_spy and self.instrumented:
      self.live_spy_callback("{}:p".format(Event(signal=signals.region_exit)))
    self.regions['p'].post_fifo(Event(signal=signals.region_exit))
    status = self.trans(some_other_state)
  elif(self.token_match(e.signal_name, "to_outer")):
    status = self.trans(outer_state)
  elif(e.signal == signals.META_EXIT):
    if hasattr(e.payload, "state"):
      self.regions['p']._post_lifo(Event(signal=signals.force_region_init))
      state = e.payload.state
      source_event = e.payload.source_event
      _e = Event(
        signal=signals.INIT_META,
        payload=META_SIGNAL_PAYLOAD(event=None, state=state,
          source_event=source_event, region=None)
      )
      self.regions['p'].post_fifo(_e)
    status = return_status.HANDLED
  # exit
  elif(e.signal == signals.EXIT_SIGNAL):
    if self.live_spy and self.instrumented:
      self.live_spy_callback("{}:p".format(Event(signal=signals.region_exit)))
    self.regions['p'].post_lifo(Event(signal=signals.region_exit))
    status = return_status.HANDLED
  else:
    self.temp.fun = outer_state
    status = return_status.SUPER
  return status


if __name__ == '__main__':
  active_states = None
  example = ScxmlChart(
    name='parallel',
    log_file="/mnt/c/github/miros-xml/experiment/parallel_example_4.log",
    live_trace=True,
    live_spy=True,
  )
  #example.instrumented = True
  example.start()
  time.sleep(0.01)

  example.post_fifo(Event(signal=signals.to_p))
  time.sleep(0.01)
  active_states = example.active_states()
  print("{:>10} -> {}".format("to_p", active_states))
  assert active_states == [['p_p11_s11', 'p_p11_s21'], 'p_s21']

  example.post_fifo(Event(signal=signals.e4))
  time.sleep(0.01)
  active_states = example.active_states()
  print("{:>10} -> {}".format("e4", active_states))
  assert active_states == [['p_p11_s12', 'p_p11_s21'], 'p_s21']

  example.post_fifo(Event(signal=signals.e1))
  time.sleep(0.02)
  active_states = example.active_states()
  print("{:>10} -> {}".format("e1", active_states))
  assert active_states == [['p_p11_r1_final', 'p_p11_s22'], 'p_s21']

  example.post_fifo(Event(signal=signals.e2))
  time.sleep(0.02)
  active_states = example.active_states()
  print("{:>10} -> {}".format("e2", active_states))
  assert active_states == [[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_s21' ]

  example.post_fifo(Event(signal=signals.C0))
  time.sleep(0.02)
  active_states = example.active_states()
  print("{:>10} -> {}".format("C0", active_states))
  assert active_states == [[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], ['p_p22_s11', 'p_p22_s21']]

  example.post_fifo(Event(signal=signals.e4))
  time.sleep(0.01)
  active_states = example.active_states()
  print("{:>10} -> {}".format("e4", active_states))
  assert active_states == [['p_p12_s12', 'p_p12_s21'], ['p_p22_s12', 'p_p22_s21']]

  example.post_fifo(Event(signal=signals.e1))
  time.sleep(0.02)
  active_states = example.active_states()
  print("{:>10} -> {}".format("e1", active_states))
  assert active_states == [['p_p12_r1_final', 'p_p12_s22'], ['p_p22_r1_final', 'p_p22_s22']]

  example.post_fifo(Event(signal=signals.e2))
  time.sleep(0.04)
  active_states = example.active_states()
  print("{:>10} -> {}".format("e2", active_states))
  assert active_states == ['some_other_state']

  example.post_fifo(Event(signal=signals.to_p))
  time.sleep(0.01)
  active_states = example.active_states()
  print("{:>10} -> {}".format("to_p", active_states))
  assert active_states == [['p_p11_s11', 'p_p11_s21'], 'p_s21']

  example.post_fifo(Event(signal=signals.e4))
  time.sleep(0.01)
  active_states = example.active_states()
  print("{:>10} -> {}".format("e4", active_states))
  assert active_states == [['p_p11_s12', 'p_p11_s21'], 'p_s21']

  example.post_fifo(Event(signal=signals.e1))
  time.sleep(0.01)
  active_states = example.active_states()
  print("{:>10} -> {}".format("e1", active_states))
  assert active_states == [['p_p11_r1_final', 'p_p11_s22'], 'p_s21']

  example.post_fifo(Event(signal=signals.to_outer))
  time.sleep(0.01)
  active_states = example.active_states()
  print("{:>10} -> {}".format("to_outer", active_states))
  assert active_states == ['outer_state']

  example.post_fifo(Event(signal=signals.E0))
  time.sleep(0.01)
  active_states = example.active_states()
  print("{:>10} -> {}".format("E0", active_states))
  assert active_states == [['p_p11_s11', 'p_p11_s21'], 'p_s21']
