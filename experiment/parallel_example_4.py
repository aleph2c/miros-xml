import re
import time
import inspect
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

FrameData = namedtuple('FrameData', [
  'filename',
  'line_number',
  'function_name',
  'lines',
  'index'])

def payload_string(e):
  tabs = ""
  result = ""

  if e.payload is None:
    result = "{}".format(e.signal_name)
  else:
    while(True):
      result += "{}{} {} ->".format(tabs,
        e.signal_name,
        e.payload.state)
      if e.payload.event is None:
        break
      else:
        e = e.payload.event
        tabs += "  "
  return result

def pprint(value):
  #pass
   print(value)

def p_spy_on(fn):
  '''spy wrapper for the parallel regions states

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
        m = re.search(r'SEARCH_FOR_SUPER_SIGNAL', str(line))
        if not m:
          if hasattr(chart, "outmost"):
            chart.outmost.live_spy_callback(
              "[{}] {}".format(chart.name, line))
          else:
            chart.live_spy_callback(
              "[{}] {}".format(chart.name, line))
      chart.rtc.spy.clear()
    else:
      e = args[0] if len(args) == 1 else args[-1]
      status = fn(chart, e)
    return status
  return _pspy_on

Reflections = []

class Region(HsmWithQueues):
  def __init__(self,
    name, starting_state, outmost, final_event,
    under_hidden_state_function, region_state_function,
    over_hidden_state_function,  instrumented=True, outer=None):

    super().__init__()
    self.name = name
    self.starting_state = starting_state
    self.outmost = outmost
    self.final_event = final_event
    self.fns = {}
    self.fns['under_hidden_state_function'] = under_hidden_state_function
    self.fns['region_state_function'] = region_state_function
    self.fns['over_hidden_state_function'] = over_hidden_state_function
    #self.under_hidden_state_function = under_hidden_state_function,
    #self.region_state_function = region_state_function,
    #self.over_hidden_state_function = over_hidden_state_function,
    self.instrumented = instrumented
    self.bottom = self.top
    self.outer = outer

    assert callable(self.fns['under_hidden_state_function'])
    assert callable(self.fns['region_state_function'])
    assert callable(self.fns['over_hidden_state_function'])

    self.final = False

    # this will be populated by the 'link' command have each
    # region has been added to the regions object
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

  def peel_meta(self, e):
    result = (None, None)
    if len(self.queue) >= 1 and \
      self.queue[0].signal == signals.META_INIT:
      _e = self.queue.popleft()
      result = (_e.payload.event, _e.payload.state)
    return result

  def has_a_child(self, fn_region_handler, fn_state_handler):
    old_temp = self.temp.fun
    old_fun = self.state.fun

    current_state = fn_region_handler
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
        break
    self.temp.fun = old_temp
    self.state.fun = old_fun
    return result

  @lru_cache(maxsize=32)
  def has_state(self, state):
    result = False

    old_temp = self.temp.fun
    old_fun = self.state.fun

    self.temp.fun = state
    super_e = Event(signal=signals.SEARCH_FOR_SUPER_SIGNAL)
    while(True):
      if(self.temp.fun == self.fns['region_state_function']):
        result = True
        r = return_status.IGNORED
      else:
        r = self.temp.fun(self, super_e)
      if r == return_status.IGNORED:
        break
    self.temp.fun = old_temp
    self.state.fun = old_fun
    return result

  # TODO: refactor to use fns members
  # only change this once you have a test in place
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
      if 'under' in self.temp.fun.__name__:
        result = self.temp.fun.__name__
        r = return_status.IGNORED
      elif 'top' in self.temp.fun.__name__:
        r = return_status.IGNORED
      else:
        r = self.temp.fun(self, super_e)
      if r == return_status.IGNORED:
        break
    self.temp.fun = old_temp
    self.state.fun = old_fun
    return result

  @lru_cache(maxsize=64)
  def lca(self, s, t):
    def build_hierachy_chain(state):
      old_temp, old_fun = self.temp.fun, self.state.fun
      self.temp.fun = state
      chain = [self.temp.fun]
      super_e = Event(signal=signals.SEARCH_FOR_SUPER_SIGNAL)
      while(True):
        r = self.temp.fun(self, super_e)
        chain.append(self.temp.fun)
        if self.temp.fun == self.top:
          pass
        if r == return_status.IGNORED:
          self.temp.fun, self.state.fun = old_temp, old_fun
          break
      return chain
    s_chain = set(build_hierachy_chain(s))
    t_chain = set(build_hierachy_chain(t))
    return list(s_chain.intersection(t_chain))[0]

  def function_name(self):
    previous_frame = inspect.currentframe().f_back
    fdata = FrameData(*inspect.getframeinfo(previous_frame))
    function_name = fdata.function_name
    return function_name

class InstrumentedActiveObject(ActiveObject):
  def __init__(self, name, log_file):
    super().__init__(name)

    self.log_file = log_file
    self.old_states = None

    logging.basicConfig(
      format='%(asctime)s %(levelname)s:%(message)s',
      filemode='w',
      filename=self.log_file,
      level=logging.DEBUG)
    self.register_live_spy_callback(partial(self.spy_callback))
    self.register_live_trace_callback(partial(self.trace_callback))

  def trace_callback(self, trace):
    '''trace without datetimestamp'''
    # trace_without_datetime = re.search(r'(\[.+\]) (\[.+\].+)', trace).group(2)
    signal_name = re.search(r'->(.+)?\(', trace).group(1)
    new_states = self.active_states()
    old_states = "['bottom']" if self.old_states is None else self.old_states
    trace = "{}<-{} == {}".format(old_states, signal_name, new_states)

    #self.print(trace_without_datetime)
    logging.debug("T: " + trace)
    self.old_states = new_states

  def spy_callback(self, spy):

    '''spy with machine name pre-pending'''
    #self.print(spy)
    logging.debug("S: [%s] %s" % (self.name, spy))

  def report(self, message):
    logging.debug("R:%s" % message)

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
    self.lookup = {}

  def add(self, region_name, outer):
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
    under_hidden_state_function = eval(region_name + "_under_hidden_region")
    region_state_function = eval(region_name + "_region")
    over_hidden_state_function = eval(region_name + "_over_hidden_region")

    assert callable(under_hidden_state_function)
    assert callable(region_state_function)
    assert callable(over_hidden_state_function)

    region =\
      Region(
        name=region_name,
        starting_state=under_hidden_state_function,
        outmost=self.outmost,
        final_event = Event(signal=self.final_signal_name),
        under_hidden_state_function = under_hidden_state_function,
        region_state_function = region_state_function,
        over_hidden_state_function = over_hidden_state_function,
        outer=outer
      )
    self._regions.append(region)
    self.lookup[region_state_function] = region
    return self

  def get_obj_for_fn(self, fn):
    result = self._regions[fn] if fn in self._regions else None
    return result

  def link(self):
    '''Link each region back to this regions object.

    Each region will be an othogonal component, but it will need to be able post
    events and drive these events through itself and the other regions that have
    been added to this object via the add method.   A call to 'link' should be
    made once all of the region objects have been added to this regions object.

    **Example(s)**:

    .. code-block:: python

      outer = self.regions['p']
      self.regions['p_p11'] = Regions(
        name='p_p11',
        outmost=self)\
      .add('p_p11_r1', outer=outer)\
      .add('p_p11_r2', outer=outer).link()

    '''
    for region in self._regions:
      for _region in self._regions:
        if _region not in region.regions:
          region.regions.append(_region)
    return self

  def post_fifo(self, e, outmost=None, ignored=None):
    self._post_fifo(e, ignored=ignored)
    [region.complete_circuit() for region in self._regions]

  def _post_fifo(self, e, ignored=None):
    regions = self._regions
    if ignored:
      if not type(ignored) is list:
        ignored = [ignored]
      regions = list(set(regions) - set(ignored))
    [region.post_fifo(e) for region in regions]

  def post_lifo(self, e, ignored=None):
    self._post_lifo(e, ignored=ignored)
    [region.complete_circuit() for region in self._regions]

  def _post_lifo(self, e, ignored=None):
    regions = self._regions
    if ignored:
      if not type(ignored) is list:
        ignored = [ignored]
      regions = list(set(regions) - set(ignored))
    [region.post_lifo(e) for region in self._regions]

  def _complete_circuit(self, ignored=None):
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

  def region(self, name):
    result = None
    for region in self._regions:
      if name == region.name:
        result = region
        break
    return result


STXRef = namedtuple('STXRef', ['send_id', 'thread_id'])

class XmlChart(InstrumentedActiveObject):
  def __init__(self, name, log_file, live_spy=None, live_trace=None):
    super().__init__(name, log_file)

    if live_spy is not None:
      self.live_spy = live_spy

    if live_trace is not None:
      self.live_trace = live_trace

    self.bottom = self.top

    self.shot_lookup = {}
    self.regions = {}

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

  def ref(self):
    previous_frame = inspect.currentframe().f_back
    fdata = FrameData(*inspect.getframeinfo(previous_frame))
    function_name = fdata.function_name
    line_number   = fdata.line_number

    #fn_ref_len = len(function_name) + 10 + len(str(line_number))
    width = 78
    print("")

    loc_and_number_report = ">>>> {} {} <<<<".format(function_name, line_number)
    additional_space =  width - len(loc_and_number_report)
    print("{}{}".format(loc_and_number_report, "<" * additional_space))
    for name, regions in self.regions.items():
      output = ""
      for region in regions._regions:
        output = region.name
        _len = len(region.queue)
        for event in region.queue:
          output += ", {}, q_len={}:".format(region.name, _len)
          print("{}".format(output))
          for e in region.queue:
            print(payload_string(e))
        if _len >= 1:
          print("-" * int(width / 2))
    print("<" * width)
    print("")

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
        STXRef(thread_id=thread_id, send_id=sendid)

  def post_lifo_with_sendid(self, sendid, e, period=None, times=None, deferred=None):
    thread_id = super().post_lifo(e, period, times, deferred)
    if thread_id is not None:
      self.shot_lookup[e.signal_name] = \
        STXRef(thread_id=thread_id, send_id=sendid)

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

  def peel_meta(self, e):
    result = (None, None)
    if len(self.queue.deque) >= 1 and \
      self.queue.deque[0].signal == signals.META_INIT:
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

  def _post_lifo(self, e, outmost=None):
    self.post_lifo(e)

  def _post_fifo(self, e, outmost=None):
    self.post_fifo(e)

  @lru_cache(maxsize=64)
  def meta_init(self, t, sig, s=None):
    '''Build target and meta event for the state.  The meta event will be a
    recursive META_INIT event for a given WTF signal and return a target for
    it's first pass off.

    **Note**:
       see `e0-wtf-event
       <https://aleph2c.github.io/miros-xml/recipes.html#e0-wtf-event>`_ for
       details about and why a META_INIT is constructed and why it is needed.

    **Args**:
       | ``t`` (state function): target state
       | ``sig`` (string): event signal name
       | ``s=None`` (state function): source state

    **Returns**:
       (Event): recursive Event

    **Example(s)**:

    .. code-block:: python

       target, onion = example.meta_init(p_p11_s21, "E0")

       assert onion.payload.state == p
       assert onion.payload.event.payload.state == p_r1_region
       assert onion.payload.event.payload.event.payload.state == p_p11
       assert onion.payload.event.payload.event.payload.event.payload.state == p_p11_r2_region
       assert onion.payload.event.payload.event.payload.event.payload.event.payload.state == p_p11_s21
       assert onion.payload.event.payload.event.payload.event.payload.event.payload.event == None

    '''
    region = None
    onion_states = []
    onion_states.append(t)

    def find_fns(state):
      '''For a given state find (region_state_function,
      outer_function_that_holds_the_region, region_object)

      **Args**:
         | ``state`` (state_function): the target of the WTF event given to
         |                             meta_init


      **Returns**:
         | (tuple): (region_state_function,
         |           outer_function_that_holds_the_region, region_object)


      **Example(s)**:

      .. code-block:: python

         a, b, c = find_fns(p_p11_s21)
         assert a == p_p11_r2_region
         assert b == p_p11
         assert c.name == 'p_p11_r2'

      '''
      outer_function_state_holds_the_region = None
      region_obj = None
      assert callable(state)
      for k, rs in self.regions.items():
        for r in rs._regions:
          if r.has_state(state):
            outer_function_state_holds_the_region = eval(rs.name)
            region_obj = r
            break
      if region_obj:
        region_state_function = region_obj.fns['region_state_function']
        assert callable(outer_function_state_holds_the_region)
        return region_state_function, outer_function_state_holds_the_region, region_obj
      else:
        return None, None, None

    target_state, region_holding_state, region = find_fns(t)
    onion_states += [target_state, region_holding_state]

    #inner_region = False
    while region and hasattr(region, 'outer'):
      target_state, region_holding_state, region = \
        find_fns(region_holding_state)
      if s is not None and region_holding_state == s:
        #inner_region = True
        break
      if target_state:
        onion_states += [target_state, region_holding_state]

    event = None
    layers = onion_states[:]
    for inner_target in layers:
      event = Event(
        signal=signals.META_INIT,
        payload=META_SIGNAL_PAYLOAD(
          event=event,
          state=inner_target,
          source_event=sig,
          region=None)
      )
    return event

  def build_onion(self, t, sig, s=None):
    region = None
    onion_states = []
    onion_states.append(t)

    def find_fns(state):
      '''For a given state find (region_state_function,
      outer_function_that_holds_the_region, region_object)

      **Args**:
         | ``state`` (state_function): the target of the WTF event given to
         |                             meta_init


      **Returns**:
         | (tuple): (region_state_function,
         |           outer_function_that_holds_the_region, region_object)


      **Example(s)**:

      .. code-block:: python

         a, b, c = find_fns(p_p11_s21)
         assert a == p_p11_r2_region
         assert b == p_p11
         assert c.name == 'p_p11_r2'

      '''
      outer_function_state_holds_the_region = None
      region_obj = None
      assert callable(state)
      for k, rs in self.regions.items():
        for r in rs._regions:
          if r.has_state(state):
            outer_function_state_holds_the_region = eval(rs.name)
            region_obj = r
            break
      if region_obj:
        region_state_function = region_obj.fns['region_state_function']
        assert callable(outer_function_state_holds_the_region)
        return region_state_function, outer_function_state_holds_the_region, region_obj
      else:
        return None, None, None

    target_state, region_holding_state, region = find_fns(t)
    onion_states += [target_state, region_holding_state]

    #inner_region = False
    while region and hasattr(region, 'outer'):
      target_state, region_holding_state, region = \
        find_fns(region_holding_state)
      if s is not None and region_holding_state == s:
        #inner_region = True
        break
      if target_state:
        onion_states += [target_state, region_holding_state]
    return onion_states

  @lru_cache(maxsize=64)
  def meta_trans(self, s, t, sig):
    lca = self.lca(_t=t, _s=s)
    # the meta init will be expressed from the lca to the target
    # 1) p_r1_region
    # 2) p_r2_region
    m_i_e = self.meta_init(t=t, s=lca, sig=sig)
    m_e_e = Event(
      signal=signals.META_EXIT,
      payload=META_SIGNAL_PAYLOAD(
      event=m_i_e,
      state=lca,
      source_event=sig,
      region=None,
      )
    )
    return m_e_e

  #@lru_cache(maxsize=64)
  def meta_trans2(self, s, t, sig):
    lca = self.lca(_t=t, _s=s)
    # the meta init will be expressed from the lca to the target
    # 1) p_r1_region
    # 2) p_r2_region
    m_i_e = self.meta_init(t=t, s=lca, sig=sig)
    # the meta init will be expressed from the lca to the target
    return m_i_e

  def lca(self, _s, _t):

    #region = None
    # get reversed onion
    s_onion = self.build_onion(_s, sig=None)[::-1]
    t_onion = self.build_onion(_t, sig=None)[::-1]
    for (a, b) in zip(s_onion, t_onion):
      _lca = a if a == b else self.bottom
      if a != b:
        break
    return _lca

# this will be wrapped into the class at some point
@lru_cache(maxsize=32)
def outmost_workers(region, region_name):
  def scribble(string):
    if region.outmost.live_spy and region.outmost.instrumented:
      region.outmost.live_spy_callback("[{}] {}".format(region_name, string))
  return (region.outmost.token_match, scribble)

# this will be wrapped into the class at some point
@lru_cache(maxsize=32)
def regions_workers(region, regions_name):
  return (region.outmost.regions[regions_name].post_fifo,
    region.outmost.regions[regions_name]._post_fifo,
    region.outmost.regions[regions_name].post_lifo,
    region.outmost.regions[regions_name]._post_lifo)

################################################################################
#                                   p region                                   #
################################################################################
# * define hidden for each region in p
# * define region state for each region in p
# * define all substates
# * define event horizon (def p)
# * in XmlChart add a region
# * in XmlChart start, add start to region
# * figure out the exits
@p_spy_on
def p_r1_under_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(r.token_match(e.signal_name, "enter_region")):

    status = r.trans(p_r1_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_r1_under_hidden_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = r.bottom
    status = return_status.SUPER
  return status

@p_spy_on
def p_r1_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_r1_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = r.peel_meta(e)  # search for META_INIT
    # if _state is a child of this state then transition to it
    if _state is None or not r.has_a_child(p_r1_region, _state):
      status = r.trans(p_p11)
    else:
      #pprint(payload_string(_e))
      #print("p_r1_region init {}".format(_state))
      status = r.trans(_state)
      if _e is not None:
        r.post_fifo(_e)
  elif(e.signal == signals.region_exit):
    status = r.trans(p_r1_under_hidden_region)
  elif(e.signal == signals.META_INIT):
    status = return_status.HANDLED
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_r1_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_r1_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r1_under_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_r1_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.force_region_init):
    status = r.trans(p_r1_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_r1_over_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_r1_over_hidden_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r1_region
    status = return_status.SUPER
  return status

# p_r1
@p_spy_on
def p_p11(r, e):
  '''
  r is either p_r1, p_r2 region
  r.outer = p
  '''
  status = return_status.UNHANDLED
  outmost = r.outmost
  # By convention this state function's name is also the key name for the outer
  # Regions collection which holds the post_fifo..._post_lifo methods to give
  # access to all of our sibling orthogonal components (parallel states)
  # contained within that Region's collection
  this_state_functions_name = r.function_name()
  (post_fifo,
   _post_fifo,
   post_lifo,
   _post_lifo) = regions_workers(r, this_state_functions_name)
  # We pass in the function_name to make the scribble log easier to read
  (token_match, scribble) = outmost_workers(r, this_state_functions_name)

  # enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11")
    scribble(e.signal_name)
    (_e, _state) = r.peel_meta(e)  # search for META_INIT
    if _state:
      _post_fifo(_e)
    post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED

  # any event handled within there regions must be pushed from here
  elif(token_match(e.signal_name, "e1") or
       token_match(e.signal_name, "e2") or
       token_match(e.signal_name, "e4") or
       token_match(e.signal_name, "A") or
       #token_match(e.signal_name, "F1") or
       token_match(e.signal_name, "G3")
       ):
    scribble(e.signal_name)
    post_fifo(e)
    status = return_status.HANDLED
  elif token_match(e.signal_name, outmost.regions['p_p11'].final_signal_name):
    scribble(e.signal_name)
    status = r.trans(p_p12)
  elif token_match(e.signal_name, "C0"):
    status = r.trans(p_p12)
  elif e.signal == signals.META_EXIT:
    region1 = r.get_region()
    region2 = r.get_region(e.payload.state)
    if region1 == region2:
      if e.payload.event is None:
        status = r.trans(e.payload.state)
      else:
        r.outer.post_fifo(e.payload.event)
    else:
      r.outer._post_lifo(e)
      r.outmost.complete_circuit()
    status = return_status.HANDLED
  elif token_match(e.signal_name, "G0"):
    status = return_status.HANDLED
  elif e.signal == signals.region_exit:
    scribble(Event(e.signal_name))
    #post_lifo(Event(signal=signals.region_exit))
    status = r.trans(p_r1_under_hidden_region)
  elif e.signal == signals.EXIT_SIGNAL:
    #scribble(Event(e.signal_name))
    post_lifo(Event(signal=signals.region_exit))
    pprint("exit p_p11")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r1_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_r1_under_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(rr.token_match(e.signal_name, "enter_region")):
    status = rr.trans(p_p11_r1_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_r1_under_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_r1_under_hidden_region")
    status = return_status.HANDLED
  else:
    rr.temp.fun = rr.bottom
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_r1_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_r1_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = rr.peel_meta(e)  # search for META_INIT
    # if _state is a child of this state then transition to it
    if _state is None or not rr.has_a_child(p_p11_r1_region, _state):
      status = rr.trans(p_p11_s11)
    else:
      status = rr.trans(_state)
      if _e is not None:
        rr.post_fifo(_e)
  elif(e.signal == signals.region_exit):
    status = rr.trans(p_p11_r1_under_hidden_region)
  elif(e.signal == signals.META_INIT):
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_r1_region")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p11_r1_under_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_r1_over_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.force_region_init):
    status = rr.trans(p_p11_r1_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_r1_over_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_r1_over_hidden_region")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p11_r1_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_s11(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_s11")
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e4")):
    status = rr.trans(p_p11_s12)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_s11")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p11_r1_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_s12(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_s12")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_s12")
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
    pprint("enter p_p11_r1_final")
    rr.final = True
    rr.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_r1_final")
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
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_r2_under_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_r2_under_hidden_region")
    status = return_status.HANDLED
  else:
    rr.temp.fun = rr.bottom
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_r2_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_r2_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = rr.peel_meta(e)  # search for META_INIT
    # if _state is a child of this state then transition to it
    if _state is None or not rr.has_a_child(p_p11_r2_region, _state):
      status = rr.trans(p_p11_s21)
    else:
      status = rr.trans(_state)
      if _e is not None:
        rr.post_fifo(_e)
  elif(e.signal == signals.region_exit):
    status = rr.trans(p_p11_r2_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_r2_region")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p11_r2_under_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_r2_over_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.force_region_init):
    status = rr.trans(p_p11_r2_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_r2_over_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_r2_over_hidden_region")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p11_r2_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_s21(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_s21")
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e1")):
    status = rr.trans(p_p11_s22)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_s21")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p11_r2_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_s22(rr, e):
  '''
  rr is p_p11_r1, p_p11_r1 and outer is p (which is wrong, I'm expecting it to be p_p11)
  '''
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_s22")
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "G3")):
    _e = rr.outmost.meta_trans(t=p_s21, s=p_p11_s22, sig=e.signal_name)

    rr.outer.post_lifo(_e)
    rr.outmost.complete_circuit()
    #status = rr.trans(rr.fns['under_hidden_state_function'])
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "F2")):
    _e = rr.outmost.meta_trans(t=p_p12_p11_s12, s=p_p11_s22, sig=e.signal_name)
    rr.outer.post_lifo(_e)
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e2")):
    status = rr.trans(p_p11_r2_final)
  elif(rr.token_match(e.signal_name, "e1")):
    status = rr.trans(p_p11_s21)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_s22")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p11_r2_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p11_r2_final(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_r2_final")
    rr.final = True
    rr.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_r2_final")
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
    pprint("enter p_r1_final")
    r.final = True
    r.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_r1_final")
    r.final = False
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r1_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12(r, e):
  status = return_status.UNHANDLED
  outmost = r.outmost

  # By convention this state function's name is also the key name for the outer
  # Regions collection which holds the post_fifo..._post_lifo methods to give
  # access to all of our sibling orthogonal components (parallel states)
  # contained within that Region's collection
  this_state_functions_name = r.function_name()
  (post_fifo,
   _post_fifo,
   post_lifo,
   _post_lifo) = regions_workers(r, this_state_functions_name)
  # We pass in the function_name to make the scribble log easier to read
  (token_match, scribble) = outmost_workers(r, this_state_functions_name)

  # enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12")
    scribble(e.signal_name)
    (_e, _state) = r.peel_meta(e)  # search for META_INIT
    if _state:
      _post_fifo(_e)
    post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  # any event handled within there regions must be pushed from here
  elif(outmost.token_match(e.signal_name, "e1") or
       outmost.token_match(e.signal_name, "e2") or
       outmost.token_match(e.signal_name, "e4") or
       outmost.token_match(e.signal_name, "G3")
       ):
    scribble(e.signal_name)
    post_fifo(e)
    status = return_status.HANDLED
  # final token match
  elif(outmost.token_match(e.signal_name, outmost.regions['p_p12'].final_signal_name)):
    scribble(e.signal_name)
    status = r.trans(p_r1_final)
  elif(outmost.token_match(e.signal_name, "e5")):
    status = r.trans(p_r1_final)
  elif outmost.token_match(e.signal_name, "E2"):
    scribble(e.signal_name)
    _e = outmost.meta_init(t=p_p12_p11_s12, s=p_p12, sig=e.signal_name)
    # this force_region_init might be a problem
    _post_lifo(Event(signal=signals.force_region_init))
    post_fifo(_e)
    status = return_status.HANDLED
  # exit signals
  elif(e.signal == signals.region_exit):
    scribble(e.signal_name)
    #post_lifo(Event(signal=signals.region_exit))
    status = r.trans(p_r1_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    scribble('region_exit')
    post_lifo(Event(signal=signals.region_exit))
    pprint("exit p_p12")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r1_over_hidden_region
    status = return_status.SUPER
  return status

# inner parallel
@p_spy_on
def p_p12_r1_under_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(rr.token_match(e.signal_name, "enter_region")):
    status = rr.trans(p_p12_r1_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_r1_under_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_r1_under_hidden_region")
    status = return_status.HANDLED
  else:
    rr.temp.fun = rr.bottom
    status = return_status.SUPER
  return status

# inner parallel
@p_spy_on
def p_p12_r1_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_r1_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = rr.peel_meta(e)  # search for META_INIT
    # if _state is a child of this state then transition to it
    if _state is None or not rr.has_a_child(p_p12_r1_region, _state):
      status = rr.trans(p_p12_p11)
    else:
      status = rr.trans(_state)
      if _e is not None:
        rr.post_fifo(_e)
  # can we get rid of region_exit?
  elif(e.signal == signals.region_exit):
    #status = rr.trans(p_p12_r1_under_hidden_region)
    status = rr.trans(p_p12_r1_under_hidden_region)
  elif(e.signal == signals.META_INIT):
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_r1_region")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p12_r1_under_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_r1_over_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.force_region_init):
    status = rr.trans(p_p12_r1_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_r1_over_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_r1_over_hidden_region")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p12_r1_region
    status = return_status.SUPER
  return status


# inner parallel
@p_spy_on
def p_p12_p11(rr, e):
  status = return_status.UNHANDLED
  outmost = rr.outmost

  # By convention this state function's name is also the key name for the outer
  # Regions collection which holds the post_fifo..._post_lifo methods to give
  # access to all of our sibling orthogonal components (parallel states)
  # contained within that Region's collection
  this_state_functions_name = rr.function_name()
  (post_fifo,
   _post_fifo,
   post_lifo,
   _post_lifo) = regions_workers(rr, this_state_functions_name)
  # We pass in the function_name to make the scribble log easier to read
  (token_match, scribble) = outmost_workers(rr, this_state_functions_name)

  # enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_p11")
    scribble(e.signal_name)
    (_e, _state) = rr.peel_meta(e)  # search for META_INIT
    if _state:
      _post_fifo(_e)
    post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  # any event handled within there regions must be pushed from here
  #elif(outmost.token_match(e.signal_name, "G2") or
  #     outmost.token_match(e.signal_name, outmost.regions['p_p12_p11'].final_signal_name)
  #    ):
  #  if outmost.live_spy and outmost.instrumented:
  #    outmost.live_spy_callback("{}:p_p12_p11".format(e.signal_name))
  #  outmost.regions['p_p12_p11'].post_fifo(e)
  #  status = return_status.HANDLED
  elif(outmost.token_match(e.signal_name, outmost.regions['p_p12_p11'].final_signal_name)):
    scribble(e.signal_name)
    status = rr.trans(p_p12_s12)
  elif(rr.token_match(e.signal_name, "e4")):
    status = rr.trans(p_p12_s12)
  elif(e.signal == signals.region_exit):
    scribble(e.signal_name)
    status = rr.trans(p_p12_r1_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    post_lifo(Event(signal=signals.region_exit))
    pprint("exit p_p12_p11")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p12_r1_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_p11_r1_under_hidden_region(rrr, e):
  status = return_status.UNHANDLED
  if(rrr.token_match(e.signal_name, "enter_region")):
    status = rrr.trans(p_p12_p11_r1_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_p11_r1_under_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_p11_r1_under_hidden_region")
    status = return_status.HANDLED
  else:
    rrr.temp.fun = rrr.bottom
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_p11_r1_region(rrr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_p11_r1_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = rrr.peel_meta(e)  # search for META_INIT
    # if _state is a child of this state then transition to it
    if _state is None or not rrr.has_a_child(p_p12_p11_r1_region, _state):
      status = rrr.trans(p_p12_p11_s11)
    else:
      status = rrr.trans(_state)
      if _e is not None:
        rrr.post_fifo(_e)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_p11_r1_region")
    status = return_status.HANDLED
  # can we get rid of region_exit?
  elif(e.signal == signals.region_exit):
    status = rrr.trans(p_p12_p11_r1_under_hidden_region)
  elif(e.signal == signals.META_INIT):
    status = return_status.HANDLED
  else:
    rrr.temp.fun = p_p12_p11_r1_under_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_p11_r1_over_hidden_region(rrr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.force_region_init):
    status = rrr.trans(p_p12_p11_r1_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_p11_r1_over_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_p11_r1_over_hidden_region")
    status = return_status.HANDLED
  else:
    rrr.temp.fun = p_p12_p11_r1_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_p11_s11(rrr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_p11_s11")
    status = return_status.HANDLED
  elif(e.signal == signals.e1):
    status = rrr.trans(p_p12_p11_s12)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_p11_s11")
    status = return_status.HANDLED
  else:
    rrr.temp.fun = p_p12_p11_r1_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_p11_s12(rrr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_p11_s12")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_p11_s12")
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
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_p11_r2_under_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_p11_r2_under_hidden_region")
    status = return_status.HANDLED
  else:
    rrr.temp.fun = rrr.bottom
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_p11_r2_region(rrr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_p11_r2_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = rrr.peel_meta(e)  # search for META_INIT
    # if _state is a child of this state then transition to it
    if _state is None or not rrr.has_a_child(p_p12_p11_r2_region, _state):
      status = rrr.trans(p_p12_p11_s21)
    else:
      status = rrr.trans(_state)
      if _e is not None:
        rrr.post_fifo(_e)
  elif(e.signal == signals.region_exit):
    status = rrr.trans(p_p12_p11_r2_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_p11_r2_region")
    status = return_status.HANDLED
  elif(e.signal == signals.META_INIT):
    status = return_status.HANDLED
  else:
    rrr.temp.fun = p_p12_p11_r2_under_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_p11_r2_over_hidden_region(rrr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.force_region_init):
    status = rrr.trans(p_p12_p11_r2_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_p11_r2_over_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_p11_r2_over_hidden_region")
    status = return_status.HANDLED
  else:
    rrr.temp.fun = p_p12_p11_r2_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_p11_s21(rrr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_p11_s21")
    status = return_status.HANDLED
  elif(e.signal == signals.G1):
    onion = rrr.outmost.build_onion(t=p_p22_s11, s=p_p12_p11_s21, sig=e)
    lca = rrr.outmost.lca(t=p_p22_s11, s=p_p12_p11_s21)
    import pdb; pdb.set_trace()

    _e = rrr.outermost.meta_trans(t=p_p22_s11, s=p_p12_p11_s21, sig=e)
    rrr.outer.post_fifo(_e)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_p11_s21")
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
    pprint("enter p_p12_s12")
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e1")):
    status = rr.trans(p_p12_r1_final)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_s12")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p12_r1_over_hidden_region
    status = return_status.SUPER
  return status

# inner parallel
@p_spy_on
def p_p12_r1_final(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_r1_final")
    rr.final = True
    rr.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_r1_final")
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
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_r2_under_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_r2_under_hidden_region")
    status = return_status.HANDLED
  else:
    rr.temp.fun = rr.bottom
    status = return_status.SUPER
  return status

# inner parallel
@p_spy_on
def p_p12_r2_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_r2_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = rr.peel_meta(e)  # search for META_INIT
    # if _state is a child of this state then transition to it
    if _state is None or not rr.has_a_child(p_p12_r2_region, _state):
      status = rr.trans(p_p12_s21)
    else:
      status = rr.trans(_state)
      if _e is not None:
        rr.post_fifo(_e)
  elif(e.signal == signals.region_exit):
    status = rr.trans(p_p12_r2_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_r2_region")
    status = return_status.HANDLED
  elif(e.signal == signals.META_INIT):
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p12_r2_under_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p12_r2_over_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.force_region_init):
    status = rr.trans(p_p12_r2_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_r2_over_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_r2_over_hidden_region")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p12_r2_region
    status = return_status.SUPER
  return status

# inner parallel
@p_spy_on
def p_p12_s21(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_s21")
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e1")):
    status = rr.trans(p_p12_s22)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_s21")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p12_r2_over_hidden_region
    status = return_status.SUPER
  return status

# inner parallel
@p_spy_on
def p_p12_s22(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_s22")
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e2")):
    status = rr.trans(p_p12_r2_final)
  elif(rr.token_match(e.signal_name, "e1")):
    status = rr.trans(p_p12_s21)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_s22")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p12_r2_over_hidden_region
    status = return_status.SUPER
  return status

# inner parallel
@p_spy_on
def p_p12_r2_final(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_r2_final")
    rr.final = True
    rr.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_r2_final")
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
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_r2_under_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_r2_under_hidden_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = r.bottom
    status = return_status.SUPER
  return status

@p_spy_on
def p_r2_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_r2_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    status = return_status.HANDLED
    (_e, _state) = r.peel_meta(e) # search for META_INIT
    if _state is None or not r.has_a_child(p_r2_region, _state):
      status = r.trans(p_s21)
    else:
      status = r.trans(_state)
      #print("p_r2_region init {}".format(_state))
      if not _e is None:
        r.post_fifo(_e)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_r2_region")
    status = return_status.HANDLED
  elif(e.signal == signals.region_exit):
    status = r.trans(p_r2_under_hidden_region)
  elif(e.signal == signals.META_INIT):
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
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_r2_over_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_r2_over_hidden_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r2_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_s21(r, e):
  status = return_status.UNHANDLED
  outmost = r.outmost

  # p_p21 is not an injector function, so we must know the name of
  # the outer Regions object that contains our sibling othogonal components
  # (parallel states).
  (post_fifo,
   _post_fifo,
   post_lifo,
   _post_lifo) = regions_workers(r, 'p')

  (token_match, scribble) = outmost_workers(r, 'p')

  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_s21")
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "C0")):
    status = r.trans(p_p22)
  elif(r.token_match(e.signal_name, "F1")):
    _e = r.outmost.meta_trans2(t=p_p22_s21, s=p_s21, sig=e.signal_name)
    r.post_lifo(Event(signal=signals.force_region_init))
    #pprint(payload_string(_e)) 
    r.outer.post_lifo(_e)
    #r.outmost.post_fifo(Event(signal=signals.flush))
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "G0")):
    _e = outmost.meta_trans2(t=p_p12_p11_s21, s=p_s21, sig=e.signal_name)
    status = r.trans(p_r2_under_hidden_region)
    r.outer.post_lifo(_e)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_s21")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r2_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22(r, e):
  status = return_status.UNHANDLED
  outmost = r.outmost

  # By convention this state function's name is also the key name for the outer
  # Regions collection which holds the post_fifo..._post_lifo methods to give
  # access to all of our sibling orthogonal components (parallel states)
  # contained within that Region's collection
  this_state_functions_name = r.function_name()
  (post_fifo,
   _post_fifo,
   post_lifo,
   _post_lifo) = regions_workers(r, this_state_functions_name)
  # We pass in the function_name to make the scribble log easier to read
  (token_match, scribble) = outmost_workers(r, this_state_functions_name)

  # enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22")
    scribble(e.signal_name)
    (_e, _state) = r.peel_meta(e) # search for META_INIT
    if _state:
      _post_fifo(_e)
    post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  # any event handled within there regions must be pushed from here
  elif(token_match(e.signal_name, "e1") or
       token_match(e.signal_name, "e2") or
       token_match(e.signal_name, "e4") or
       token_match(e.signal_name, "E2")
      ):
    scribble(e.signal_name)
    post_fifo(e)
    status = return_status.HANDLED
  # final token match
  elif(outmost.token_match(e.signal_name, outmost.regions['p_p22'].final_signal_name)):
    scribble(e.signal_name)
    status = r.trans(p_r2_final)
  #elif(e.signal == signals.META_EXIT):
  #  region1 = r.get_region()
  #  region2 = r.get_region(e.payload.state)
  #  if region1 == region2:
  #    status = r.trans(e.payload.state)
  #  else:
  #    status = return_status.HANDLED
  # exit signals
  elif(e.signal == signals.EXIT_SIGNAL):
    post_lifo(Event(signal=signals.region_exit))
    pprint("exit p_p22")
    status = return_status.HANDLED
  elif(e.signal == signals.region_exit):
    scribble(e.signal_name)
    status = r.trans(p_r2_under_hidden_region)
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
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_r1_under_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_r1_under_hidden_region")
    status = return_status.HANDLED
  else:
    rr.temp.fun = rr.bottom
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_r1_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_r1_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = rr.peel_meta(e) # search for META_INIT
    # if _state is a child of this state then transition to it
    if _state is None or not rr.has_a_child(p_p22_r1_region, _state):
      status = rr.trans(p_p22_s11)
    else:
      status = rr.trans(_state)
      if not _e is None:
        rr.post_fifo(_e)
  # can we get rid of region_exit?
  elif(e.signal == signals.region_exit):
    status = rr.trans(p_p22_r1_under_hidden_region)
  elif(e.signal == signals.META_INIT):
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_r1_region")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p22_r1_under_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_r1_over_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal==signals.force_region_init):
    status = rr.trans(p_p22_r1_under_hidden_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_r1_over_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_r1_over_hidden_region")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p22_r1_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_s11(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_s11")
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e4")):
    status = rr.trans(p_p22_s12)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_s11")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p22_r1_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_s12(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_s12")
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e1")):
    status = rr.trans(p_p22_r1_final)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_s12")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p22_r1_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_r1_final(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_r1_final")
    status = return_status.HANDLED
    r.final = True
    r.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_r1_final")
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
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_r2_under_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_r2_under_hidden_region")
    status = return_status.HANDLED
  else:
    rr.temp.fun = rr.bottom
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_r2_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_r2_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = rr.peel_meta(e) # search for META_INIT
    # if _state is a child of this state then transition to it
    if _state is None or not rr.has_a_child(p_p22_r2_region, _state):
      status = rr.trans(p_p22_s21)
    else:
      status = rr.trans(_state)
      if not _e is None:
        rr.post_fifo(_e)
  elif(e.signal == signals.region_exit):
    status = rr.trans(p_p22_r2_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_r2_region")
    status = return_status.HANDLED
  elif(e.signal == signals.META_INIT):
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p22_r2_under_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_r2_over_hidden_region(rr, e):
  status = return_status.UNHANDLED
  if(e.signal==signals.force_region_init):
    status = rr.trans(p_p22_r2_under_hidden_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_r2_over_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_r2_over_hidden_region")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p22_r2_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_s21(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_s21")
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e1")):
    status = rr.trans(p_p22_s22)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_s21")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p22_r2_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_s22(rr, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_s22")
    status = return_status.HANDLED
  elif(rr.token_match(e.signal_name, "e2")):
    status = rr.trans(p_p22_r2_final)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_s22")
    status = return_status.HANDLED
  else:
    rr.temp.fun = p_p22_r2_over_hidden_region
    status = return_status.SUPER
  return status

@p_spy_on
def p_p22_r2_final(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_r2_final")
    r.final = True
    r.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_r2_final")
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
    pprint("enter p_r2_final")
    r.final = True
    r.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_r2_final")
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
    pprint("enter outer_state")
    status = return_status.HANDLED
  elif(self.token_match(e.signal_name, "to_p")):
    if self.live_spy and self.instrumented:
      self.live_spy_callback("{}:outer_state".format(e.signal_name))
    status = self.trans(p)
  elif(self.token_match(e.signal_name, "E0")):
    pprint("enter outer_state")
    if self.live_spy and self.instrumented:
      self.live_spy_callback("{}:outer_state".format(e.signal_name))
    _e = self.meta_init(t=p_p11_s22, sig=e.signal_name)
    self.scribble(payload_string(_e))
    #_f, _state = self.peel_meta(_e)
    #self.regions['p']._post_lifo(Event(signal=signals.force_region_init))
    #self.post_fifo(_e)
    #status = self.trans(_state)
    #self.post_fifo(_e.payload.event)
    #status = self.trans(_e.payload.state)
    self.post_fifo(_e.payload.event)
    status = self.trans(_e.payload.state)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit outer_state")
    status = return_status.HANDLED
  else:
    self.temp.fun = self.bottom
    status = return_status.SUPER
  return status

@spy_on
def some_other_state(self, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter some_other_state")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit some_other_state")
    status = return_status.HANDLED
  elif(e.signal == signals.F0):
    if self.live_spy and self.instrumented:
      self.live_spy_callback("{}:outer_state".format(e.signal_name))
    _e = self.meta_init(t=p_p22_s21, sig=e.signal_name)
    self.post_fifo(_e.payload.event)
    status = self.trans(_e.payload.state)
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
      self.live_spy_callback("[p] {}".format(e.signal_name))
    (_e, _state) = self.peel_meta(e) # search for META_INIT
    if _state:
      self.regions['p']._post_fifo(_e)
    pprint("enter p")
    self.regions['p'].post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  # any event handled within there regions must be pushed from here
  elif(type(self.regions) == dict and (self.token_match(e.signal_name, "e1") or
      self.token_match(e.signal_name, "e2") or
      self.token_match(e.signal_name, "e3") or
      self.token_match(e.signal_name, "e4") or
      self.token_match(e.signal_name, "e5") or
      self.token_match(e.signal_name, "C0") or
      self.token_match(e.signal_name, "E2") or
      self.token_match(e.signal_name, "G3") or
      self.token_match(e.signal_name, "F1") or
      self.token_match(e.signal_name, "F2") or
      self.token_match(e.signal_name, "G0") or
      self.token_match(e.signal_name, self.regions['p_p11'].final_signal_name) or
      self.token_match(e.signal_name, self.regions['p_p12'].final_signal_name) or
      self.token_match(e.signal_name, self.regions['p_p22'].final_signal_name)
      )):
    if self.live_spy and self.instrumented:
      self.live_spy_callback("[p] {}".format(e.signal_name))
    self.regions['p'].post_fifo(e)
    status = return_status.HANDLED
  elif(e.signal == signals.META_INIT):
    self.regions['p']._post_lifo(Event(signal=signals.force_region_init))
    self.regions['p'].post_fifo(e.payload.event.payload.event)
    status = return_status.HANDLED
  elif(self.token_match(e.signal_name, "E1")):
    if self.live_spy and self.instrumented:
      self.live_spy_callback("[p] {}".format(e.signal_name))
    _e = self.meta_init(t=p_p11_s12, s=p, sig=e.signal_name)
    self.scribble(payload_string(_e))
    self.regions['p']._post_lifo(Event(signal=signals.force_region_init))
    self.regions['p'].post_fifo(_e)
    status = return_status.HANDLED
  # final token match
  elif(type(self.regions) == dict and self.token_match(e.signal_name,
    self.regions['p'].final_signal_name)):
    if self.live_spy and self.instrumented:
      self.live_spy_callback("[p] {}".format('region_exit'))
    self.regions['p'].post_fifo(Event(signal=signals.region_exit))
    status = self.trans(some_other_state)
  elif(self.token_match(e.signal_name, "to_o")):
    status = self.trans(outer_state)
  elif(self.token_match(e.signal_name, "to_s")):
    status = self.trans(some_other_state)
  elif(self.token_match(e.signal_name, "A1")):
    status = self.trans(p)
  elif(e.signal == signals.META_EXIT):
    self.regions['p']._post_lifo(Event(signal=signals.force_region_exit))
    self.regions['p']._complete_circuit()
    status = return_status.HANDLED

  #elif e.signal == signals.META_EXIT:
  #  region1 = r.get_region()
  #  region2 = r.get_region(e.payload.state)
  #  if region1 == region2:
  #    if e.payload.event == None:
  #      status = r.trans(e.payload.state)
  #    else:
  #      r.outer.post_fifo(e.payload.event)
  #  else:
  #    r.outer._post_lifo(e)
  #    r.outmost.complete_circuit()
  #  status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    if self.live_spy and self.instrumented:
      self.live_spy_callback("[p] {}".format('region_exit'))
    self.regions['p'].post_lifo(Event(signal=signals.region_exit))
    pprint("exit p")
    status = return_status.HANDLED
  elif(e.signal == signals.region_exit):
    if self.live_spy and self.instrumented:
      self.live_spy_callback("[p] {}".format('region_exit'))
    status = return_status.HANDLED
  else:
    self.temp.fun = outer_state
    status = return_status.SUPER
  return status


if __name__ == '__main__':
  regression = True
  active_states = None
  example = XmlChart(
    name='x',
    log_file="/mnt/c/github/miros-xml/experiment/parallel_example_4.log",
    live_trace=False,
    live_spy=True,
  )
  #example.instrumented = False
  example.instrumented = True
  example.start()
  time.sleep(0.20)
  example.report("\nstarting regression\n")

  if regression:
    def build_test(s, expected_result, old_result, duration=0.2):
      example.post_fifo(Event(signal=s))
      time.sleep(duration)
      active_states = example.active_states()[:]
      string1 = "{:>79}{:>5} <-> {:<80}".format(str(old_result), s, str(active_states))
      string2 = "\n{} <- {} == {}\n".format(str(old_result), s, str(active_states))
      print(string1)
      example.report(string2)
      if active_states != expected_result:
        previous_frame = inspect.currentframe().f_back
        fdata = FrameData(*inspect.getframeinfo(previous_frame))
        function_name = '__main__'
        line_number   = fdata.line_number
        print("Assert error from {}:{}".format(function_name, line_number))
        print("Expecting: {}".format(active_states))
        print("Observed:  {}".format(expected_result))
        exit(0)
      assert active_states == expected_result
      return active_states

    active_states = example.active_states()
    print("{:>10} -> {}".format("start", active_states))

    old_results = example.active_states()[:]

    old_results = build_test(
      s='to_p',
      expected_result=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      duration=0.2
    )

    old_results = build_test(
      s='to_p',
      expected_result=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      s='e4',
      expected_result=[['p_p11_s12', 'p_p11_s21'], 'p_s21'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      s='e1',
      expected_result=[['p_p11_r1_final', 'p_p11_s22'], 'p_s21'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      s='e2',
      expected_result=[[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_s21'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      s='C0',
      expected_result=[[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], ['p_p22_s11', 'p_p22_s21']],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      s='e4',
      expected_result=[['p_p12_s12', 'p_p12_s21'], ['p_p22_s12', 'p_p22_s21']],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      s='e1',
      expected_result=[['p_p12_r1_final', 'p_p12_s22'], ['p_p22_r1_final', 'p_p22_s22']],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      s='e2',
      expected_result=['some_other_state'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      s='to_p',
      expected_result=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      s='e4',
      expected_result=[['p_p11_s12', 'p_p11_s21'], 'p_s21'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      s='e1',
      expected_result=[['p_p11_r1_final', 'p_p11_s22'], 'p_s21'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      s='to_o',
      expected_result=['outer_state'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      s='E0',
      expected_result=[['p_p11_s11', 'p_p11_s22'], 'p_s21'],
      old_result=old_results,
      duration=0.2
    )


    onion = example.meta_init(t=p_p11_s21, sig='E0')
    assert onion.payload.state == p
    assert onion.payload.event.payload.state == p_r1_region
    assert onion.payload.event.payload.event.payload.state == p_p11
    assert onion.payload.event.payload.event.payload.event.payload.state == p_p11_r2_region
    assert onion.payload.event.payload.event.payload.event.payload.event.payload.state == p_p11_s21
    assert onion.payload.event.payload.event.payload.event.payload.event.payload.event == None

    old_results = build_test(
      s='E1',
      expected_result=[['p_p11_s12', 'p_p11_s21'], 'p_s21'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      s='E2',
      expected_result=[['p_p11_s12', 'p_p11_s21'], 'p_s21'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      s='C0',
      expected_result=[[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], ['p_p22_s11', 'p_p22_s21']],
      old_result=old_results,
      duration=0.2
    )

    # Write comments in doc describing how E2 will cause p_p12_s21 to exit and
    # then re-initialize back into p_p12_s21.   (The E2 hits the p_p12 regional
    # wall, so it effects all parallel regions within it)
    old_results = build_test(
      s='E2',
      expected_result=[[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s21'], ['p_p22_s11', 'p_p22_s21']],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      s='E0',
      expected_result=[['p_p11_s11', 'p_p11_s22'], 'p_s21'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      s='F1',
      expected_result=[['p_p11_s11', 'p_p11_s21'], ['p_p22_s11', 'p_p22_s21']],
      old_result=old_results,
      duration=0.2
    )

    #old_results = build_test(
    #  s='E0',
    #  expected_result=[['p_p11_s11', 'p_p11_s22'], 'p_s21' ],
    #  old_result=old_results,
    #  duration=0.2
    #)

    ##old_results = build_test(
    ##  s='F2',
    ##  expected_result=[[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_s21'],
    ##  old_result=old_results,
    ##  duration=0.2
    ##)

    #old_results = build_test(
    #  s='to_s',
    #  expected_result=['some_other_state'],
    #  old_result=old_results,
    #  duration=0.2
    #)

    #old_results = build_test(
    #  s='F0',
    #  expected_result=[['p_p11_s11', 'p_p11_s21'], ['p_p22_s11', 'p_p22_s21']],
    #  old_result=old_results,
    #  duration=0.2
    #)

    #old_results = build_test(
    #  s='A1',
    #  expected_result=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
    #  old_result=old_results,
    #  duration=0.2
    #)

    #old_results = build_test(
    #  s='G0',
    #  expected_result=[[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_r2_under_hidden_region'],
    #  old_result=old_results,
    #  duration=0.2
    #)

  else:
    # remove the following from regression
    active_states = example.active_states()[:]
    old_results = active_states

    example.post_fifo(Event(signal=signals.to_p))
    time.sleep(0.2)
    active_states = example.active_states()[:]
    string1 = "{:>79}{:>5} <-> {:<80}".format(str(old_results), 'to_p', str(active_states))
    string2 = "\n{} <- {} == {}\n".format(str(old_results), 'to_p', str(active_states))
    example.report(string2)
    old_results = active_states

    example.post_fifo(Event(signal='C0'))
    time.sleep(0.4)
    active_states = example.active_states()[:]
    string1 = "{:>79}{:>5} <-> {:<80}".format(str(old_results), 'C0', str(active_states))
    string2 = "\n{} <- {} == {}\n".format(str(old_results), 'C0', str(active_states))
    print(string1)
    example.report(string2)
    old_results = active_states

    example.post_fifo(Event(signal='G1'))
    time.sleep(0.4)
    active_states = example.active_states()[:]
    string1 = "{:>79}{:>5} <-> {:<80}".format(str(old_results), 'G1', str(active_states))
    string2 = "\n{} <- {} == {}\n".format(str(old_results), 'C0', str(active_states))
    print(string1)
    example.report(string2)
    # assert active_states == [[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], ['p_p23_s11', 'p_p22_s21']]
    # old_results = active_states

    # example.post_fifo(Event(signal='G1'))
    # time.sleep(0.4)
    # active_states = example.active_states()[:]
    # string1 = "{:>79}{:>5} <-> {:<80}".format(str(old_results), 'G1', str(active_states))
    # string2 = "\n{} <- {} == {}\n".format(str(old_results), 'G1', str(active_states))
    # print(string1)
    # example.report(string2)
    # expected_result=['p_r1_under_hidden_region', ['p_p23_11', 'p_p22_s21']],
    # assert active_states == expected_result
    # old_results = active_states


  time.sleep(1000)

