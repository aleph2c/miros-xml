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

import pprint as xprint
def pp(item):
  xprint.pprint(item)

META_SIGNAL_PAYLOAD = namedtuple("META_SIGNAL_PAYLOAD",
  ['event', 'state',  'previous_state', 'previous_signal'])

FrameData = namedtuple('FrameData', [
  'filename',
  'line_number',
  'function_name',
  'lines',
  'index'])

@lru_cache(maxsize=128)
def f_to_s(fn):
  '''function to str'''
  return re.match(r'.+function ([a-zA-Z1-9_]+)?[ ]', str(fn)).group(1)

@lru_cache(maxsize=128)
def s_to_s(event_or_signal_number):
  '''signal to str'''
  if type(event_or_signal_number) == int:
    signal_name = signals.name_for_signal(event_or_signal_number)
  elif type(event_or_signal_number) == str:
    signal_name = event_or_signal_number
  else:
    signal_name = event_or_signal_number.signal_name
  return signal_name

def payload_string(e):
  tabs = ""
  result = ""

  if e.payload is None:
    result = "{}".format(e.signal_name)
  else:
    while(True):
      previous_signal_name = s_to_s(e.payload.previous_signal)
      result += "{}[n]::{}:{} [n-1]::{}:{} ->\n".format(
        tabs,
        e.signal_name,
        f_to_s(e.payload.state),
        previous_signal_name,
        f_to_s(e.payload.previous_state),
      )
      if e.payload.event is None:
        break
      else:
        e = e.payload.event
        tabs += "  "
  return result

def pprint(value):
  pass
  # print(value)

def state(fn):
  '''Statechart state function wrapper, provides instrumentation and
  dynamically assigns the inner attribute'''
  @wraps(fn)
  def _state(chart, *args):
    fn_as_s = f_to_s(fn)
    if fn_as_s not in chart.regions:
      chart.inner = None
    else:
      chart.inner = chart.regions[fn_as_s]
    chart.current_function_name = fn_as_s

    status = spy_on(fn)(chart, *args)
    return status
  return _state


def othogonal_state(fn):
  '''Othogonal component state function wrapper, provides instrumentation and
  dynamically assigns the inner attribute.'''
  @wraps(fn)
  def _pspy_on(region, *args):

    # dynamically assign the inner attribute
    fn_as_s = f_to_s(fn)
    if fn_as_s not in region.inners:
      region.inners[fn_as_s] = None
      if fn_as_s in region.outmost.regions:
        region.inners[fn_as_s] = region.outmost.regions[fn_as_s]
    region.inner = region.inners[fn_as_s]
    region.current_function_name = fn_as_s

    # instrument the region
    if region.instrumented:
      status = spy_on(fn)(region, *args)  # call to state function here
      for line in list(region.rtc.spy):
        m = re.search(r'SEARCH_FOR_SUPER_SIGNAL', str(line))
        if not m:
          if hasattr(region, "outmost"):
            region.outmost.live_spy_callback(
              "[{}] {}".format(region.name, line))
          else:
            region.live_spy_callback(
              "[{}] {}".format(region.name, line))
      region.rtc.spy.clear()
    else:
      e = args[0] if len(args) == 1 else args[-1]
      status = fn(region, e)  # call to state function here
    return status

  return _pspy_on

Reflections = []

class Region(HsmWithQueues):
  def __init__(self,
    name, starting_state, outmost, outer, same,
    final_event, under_hidden_state_function,
    region_state_function, over_hidden_state_function,
    instrumented=True):
    '''Create an orthogonal component HSM


    **Args**:
       | ``name`` (str): The name of the region, naming follows a convention
       | ``starting_state`` (fn): The starting state of the region
       | ``outmost`` (InstrumentedActiveObject): The statechart that is
       |                          connected to a driving thread
       | ``outer`` (Regions): A Regions object equivalent to an
       |                      outer parallel region
       | ``same`` (Regions): A Regions object equivalent to this object's
       |                          parallel region.
       | ``final_event`` (Event): The event that will be fired with all states
       |                          in this parallel part of the statechart are in
       |                          their final states.
       | ``under_hidden_state_function`` (fn): The inert state for this Region
       | ``region_state_function`` (fn): The state which contains the
       |                          programmable init feature
       | ``over_hidden_state_function`` (fn): The state that can force a
       |                          transition to the region_state_function
       | ``instrumented=True`` (bool): Do we want instrumentation?


    **Returns**:
       (Region): This HSM

    **Example(s)**:

    .. code-block:: python

      # from within the Regions class
      region =\
        Region(
          name='bob',
          starting_state=under_hidden_state_function,
          outmost=self.outmost,
          outer=outer,
          same=self,
          final_event = Event(signal='bob_final'),
          under_hidden_state_function = under_hidden_state_function,
          region_state_function = region_state_function,
          over_hidden_state_function = over_hidden_state_function,
        )

    '''

    super().__init__()
    self.name = name
    self.starting_state = starting_state
    self.final_event = final_event
    self.fns = {}
    self.fns['under_hidden_state_function'] = under_hidden_state_function
    self.fns['region_state_function'] = region_state_function
    self.fns['over_hidden_state_function'] = over_hidden_state_function
    self.instrumented = instrumented
    self.bottom = self.top

    self.outmost = outmost
    self.outer = outer
    self.same = same
    # The inners dict will be indexed by state function names as strings.
    # It will be populated as the state machine is run, by the orthgonal_state
    # decorator.  This collection will be used to provide the 'inner' attribute
    # with its regions object if the function using this attribute is an
    # injector
    self.inners = {}
    self.current_function_name = None  # dynamically assigned

    assert callable(self.fns['under_hidden_state_function'])
    assert callable(self.fns['region_state_function'])
    assert callable(self.fns['over_hidden_state_function'])

    self.final = False

    # this will be populated by the 'link' command have each
    # region has been added to the regions object
    self.regions = []

  def scribble(self, string):
    '''Add some state context to the spy instrumention'''
    # the current_function_name is set by the orthongal_state decoractor
    if self.outmost.live_spy and self.outmost.instrumented:
      self.outmost.live_spy_callback("[{}] {}".format(
        self.current_function_name, string))

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

  def meta_peel(self, e):
    result = (None, None)
    if len(self.queue) >= 1 and \
      (self.queue[0].signal == signals.INIT_META_SIGNAL or
       self.queue[0].signal == signals.EXIT_META_SIGNAL):
      _e = self.queue.popleft()
      result = (_e.payload.event, _e.payload.state)
    return result

  @lru_cache(maxsize=32)
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

  @lru_cache(maxsize=32)
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

  def function_name(self):
    previous_frame = inspect.currentframe().f_back
    fdata = FrameData(*inspect.getframeinfo(previous_frame))
    function_name = fdata.function_name
    return function_name

  def pop_event(self):
    result = None
    if len(self.queue) >= 1:
      result = self.queue.popleft()
    return result


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
    for the EXIT_META_SIGNAL signal.

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
        outer=outer,
        same=self,
        final_event = Event(signal=self.final_signal_name),
        under_hidden_state_function = under_hidden_state_function,
        region_state_function = region_state_function,
        over_hidden_state_function = over_hidden_state_function,
      )
    self._regions.append(region)
    self.lookup[region_state_function] = region
    return self

  def get_obj_for_fn(self, fn):
    result = self._regions[fn] if fn in self._regions else None
    return result

  def link(self):
    '''Create the 'same' and 'regions' attributes for each region object in this
    regions object.

    The region objects will be placed into a list and any region will be able to
    access the other region objects at its level by accessing that list.  This
    list will be called regions, and it is an attribute of the region object.
    Linking a region to it's other region object's is required to provide the
    final_event feature and that kind of thing.

    The link method will also create the "same" attribute.  This is a reference
    to this regions object, or the thing that contains the post_fifo, post_life
    ... methods which place and drive events into region objects at the same
    level of the orthgonal component hierarchy.

    A call to 'link' should be made once all of the region objects have been
    added to this regions object.

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
      region.same = self
    return self

  def post_fifo(self, e):
    self._post_fifo(e)
    [region.complete_circuit() for region in self._regions]

  def _post_fifo(self, e):
    regions = self._regions
    [region.post_fifo(e) for region in regions]

  def post_lifo(self, e):
    self._post_lifo(e)
    [region.complete_circuit() for region in self._regions]

  def meta_bounce_across(self, e):
    trans_init = False
    if e.payload.event.signal != e.payload.previous_signal:
      trans_init = True
    if trans_init:
      for region in self._regions:
        if region.has_state(e.payload.previous_state):
          region.pop_event()
          region.post_lifo(Event(signal=signals.region_exit))
        else:
          region.post_lifo(Event(signal=signals.force_region_init))
      self._complete_circuit()
    else:
      self.post_lifo(Event=signals.force_region_init)

  def meta_bounce_same(self, e):
    trans_init = False
    if e.payload.event.signal != e.payload.previous_signal:
      trans_init = True
    if not trans_init:
      for region in self._regions:
        if region.has_state(e.payload.previous_state):
          region.post_lifo(Event(signal=signals.force_region_init))
        else:
          region.pop_event()
          region.post_lifo(Event(signal=signals.region_exit))
      self._complete_circuit()
    else:
      self.post_lifo(Event=signals.force_region_init)

  def meta_bounce(self, e):
    self.post_lifo(Event=signals.force_region_init)

  def _post_lifo(self, e):
    [region.post_lifo(e) for region in self._regions]

  def _complete_circuit(self):
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

    self.current_function_name = None  # dynamically assigned

  def ref(self):
    '''debug function, so it can be slow'''
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

  def meta_peel(self, e):
    result = (None, None)
    if len(self.queue.deque) >= 1 and \
      (self.queue.deque[0].signal == signals.INIT_META_SIGNAL or
       self.queue.deque[0].signal == signals.EXIT_META_SIGNAL):

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
  def meta_init(self, s, t, sig):
    '''Build target and meta event for the state.  The meta event will be a
    recursive INIT_META_SIGNAL event for a given WTF signal and return a target for
    it's first pass off.

    **Note**:
       see `e0-wtf-event
       <https://aleph2c.github.io/miros-xml/recipes.html#e0-wtf-event>`_ for
       details about and why a INIT_META_SIGNAL is constructed and why it is needed.

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

    @lru_cache(maxsize=32)
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

    while region and hasattr(region, 'outer'):
      target_state, region_holding_state, region = \
        find_fns(region_holding_state)
      if s is not None and region_holding_state == s:
        break
      if target_state:
        onion_states += [target_state, region_holding_state]

    # Wrap up the onion meta event from the inside out.
    # History items at the last layer of the outer part of the
    # INIT_META_SIGNAL need to reference an even more outer part
    # of the onion, the source of the meta_init call.
    event = None
    init_onion = onion_states[:]
    for index, entry_target in enumerate(init_onion):
      previous_signal = signals.INIT_META_SIGNAL
      if index == len(init_onion) - 1:
        previous_signal = sig
        previous_state = s
      else:
        previous_state =  init_onion[index + 1]

      event = Event(
        signal=signals.INIT_META_SIGNAL,
        payload=META_SIGNAL_PAYLOAD(
          event=event,
          state=entry_target,
          previous_state=previous_state,
          previous_signal=previous_signal,
        )
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
    '''Create an event onion which can be passed over zero or one orthogonal
    components.

    The orthogonal component pattern allows HSM objects within other HSM
    objects.  The event processor used by the miros library does not support
    transition over HSM boundaries.  This method creates recursive events which
    allow for transitions out of an into orthogonal components, assuming their
    states have been written to support these events.

    **Note**:
      The trans event will contain a payload of the META_SIGNAL_PAYLOAD type.
      See the top of the files for details.

    **Args**:
       | ``s`` (function): source state function (where we are coming from)
       | ``t`` (function): target state function (where we want to go)
       | ``sig`` (type1): signal name of event that initiated the need for this
       |                  event


    **Returns**:
       (Event): An event which can contain other events within it.

    **Example(s)**:

    .. code-block:: python

      @othogonal_state
      def p_p11_s12(r, e):
        elif(e.signal == signals.G1):
          status = return_status.HANDLED

          # Reference the parallel_region_to_orthogonal_component_mapping_6.pdf in
          # this repository
          status = return_status.HANDLED
          _e = r.outmost.meta_trans(
            s=p_p11_s12,
            t=p_s22,
            sig=e.signal_name
          )
          r.outmost.regions['p_p11'].post_fifo(_e)

    '''
    source = s
    target = t
    lca = self.lca(source, target)

    exit_onion = self.build_onion(
        t=source,
        s=lca,
        sig=sig)[1:]
    exit_onion.reverse()

    entry_onion = self.build_onion(
        s=lca,
        t=target,
        sig=sig)[:-1]

    # Wrap up the onion meta event from the inside out.  History items at the
    # last layer of the outer part of the INIT_META_SIGNAL need to reference an
    # even more outer part of the onion, the meta exit details.
    event = None
    for index, entry_target in enumerate(entry_onion):
      previous_signal = signals.INIT_META_SIGNAL
      if index == len(entry_onion) - 1:
        previous_state = exit_onion[0]
        previous_signal = signals.EXIT_META_SIGNAL
      else:
        previous_state = entry_onion[index + 1]

      event = Event(
        signal=signals.INIT_META_SIGNAL,
        payload=META_SIGNAL_PAYLOAD(
          event=event,
          state=entry_target,
          previous_state=previous_state,
          previous_signal=previous_signal,
        )
      )

    # Wrapping the EXIT_META_SIGNAL details around the META INIT part
    # on the onion meta event.  When we are at the outer layer
    # we need to write in the event that caused this meta event
    # and from what state it was created.
    for index, exit_target in enumerate(exit_onion):
      previous_signal = signals.EXIT_META_SIGNAL
      if index == len(entry_onion) - 1:
        previous_state = source
        previous_signal = sig
      else:
        previous_state = entry_onion[index + 1]

      event = Event(
        signal=signals.EXIT_META_SIGNAL,
        payload=META_SIGNAL_PAYLOAD(
          event=event,
          state=exit_target,
          previous_state=previous_state,
          previous_signal=previous_signal,
        )
      )

    return (t, event)

  @lru_cache(maxsize=32)
  def lca(self, _s, _t):
    #region = None
    # get reversed onion
    s_onion = self.build_onion(_s, sig=None)[::-1]
    t_onion = self.build_onion(_t, sig=None)[::-1]
    _lca = list(set(s_onion).intersection(set(t_onion)))[-1]
    _lca = self.bottom if _lca is None else _lca
    return _lca

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
@othogonal_state
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

@othogonal_state
def p_r1_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_r1_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = r.meta_peel(e)  # search for INIT_META_SIGNAL
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
  elif(e.signal == signals.INIT_META_SIGNAL):
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

@othogonal_state
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
@othogonal_state
def p_p11(r, e):
  '''
  r is either p_r1, p_r2 region
  r.outer = p
  '''
  status = return_status.UNHANDLED

  # enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11")
    r.scribble(e.signal_name)
    (_e, _state) = r.meta_peel(e)  # search for INIT_META_SIGNAL
    if _state:
      r.inner._post_fifo(_e)
    r.inner.post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED

  # any event handled within there regions must be pushed from here
  elif(r.token_match(e.signal_name, "e1") or
       r.token_match(e.signal_name, "e2") or
       r.token_match(e.signal_name, "G1")
       ):
    r.scribble(e.signal_name)
    r.inner.post_fifo(e)
    status = return_status.HANDLED

  elif e.signal == signals.EXIT_META_SIGNAL:
    (_e, _state) = e.payload.event, e.payload.state
    print("b")
    print(_state)
    if r.has_a_child(p_p11, _state):
      r.outer.post_lifo(_e)
    status = return_status.HANDLED
  elif e.signal == signals.region_exit:
    r.scribble(Event(e.signal_name))
    #post_lifo(Event(signal=signals.region_exit))
    status = r.trans(p_r1_under_hidden_region)
  elif e.signal == signals.EXIT_SIGNAL:
    r.inner.post_lifo(Event(signal=signals.region_exit))
    pprint("exit p_p11")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r1_over_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_r1_under_hidden_region(r, e):
  status = return_status.UNHANDLED

  if(r.token_match(e.signal_name, "enter_region")):
    status = r.trans(p_p11_r1_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_r1_under_hidden_region")
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_r1_under_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.REGION_NAME_SIGNAL):
    r.temp.region = "p_p11"
    status = return_status.HANDLED
  else:
    r.temp.fun = r.bottom
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_r1_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_r1_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = r.meta_peel(e)  # search for INIT_META_SIGNAL
    # if _state is a child of this state then transition to it
    if _state is None or not r.has_a_child(p_p11_r1_region, _state):
      status = r.trans(p_p11_s11)
    else:
      status = r.trans(_state)
      if _e is not None:
        r.post_fifo(_e)
  elif(e.signal == signals.EXIT_META_SIGNAL):
    # if the state is in this region continue meta exit
    (_e, _state) = e.payload.event, e.payload.state
    print("a")
    print(_state)
    if r.has_a_child(p_p11_r1_region, _state):
      r.outer.post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.region_exit):
    status = r.trans(p_p11_r1_under_hidden_region)
  elif(e.signal == signals.INIT_META_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_r1_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p11_r1_under_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_r1_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.force_region_init):
    status = r.trans(p_p11_r1_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_r1_over_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_r1_over_hidden_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p11_r1_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_s11(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_s11")
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e1")):
    status = r.trans(p_p11_s12)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_s11")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p11_r1_over_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_s12(r, e):
  status = return_status.UNHANDLED

  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_s12")
    status = return_status.HANDLED

  elif(e.signal == signals.G1):
    status = return_status.HANDLED
    t, _e = r.outmost.meta_trans(
      s=p_p11_s12,
      t=p_s22,
      sig=e.signal_name
    )
    print(payload_string(_e))
    r.same.post_fifo(_e)

  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_s12")
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e1")):
    status = r.trans(p_p11_r1_final)
  else:
    r.temp.fun = p_p11_r1_over_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_r1_final(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_r1_final")
    r.final = True
    r.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_r1_final")
    r.final = False
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p11_r1_over_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_r2_under_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(r.token_match(e.signal_name, "enter_region")):
    status = r.trans(p_p11_r2_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_r2_under_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_r2_under_hidden_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = r.bottom
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_r2_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_r2_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    (_e, _state) = r.meta_peel(e)  # search for INIT_META_SIGNAL
    # if _state is a child of this state then transition to it
    if _state is None or not r.has_a_child(p_p11_r2_region, _state):
      status = r.trans(p_p11_s21)
    else:
      status = r.trans(_state)
      if _e is not None:
        r.post_fifo(_e)
  elif(e.signal == signals.region_exit):
    status = r.trans(p_p11_r2_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_r2_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p11_r2_under_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_r2_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.force_region_init):
    status = r.trans(p_p11_r2_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_r2_over_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_r2_over_hidden_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p11_r2_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_s21(r, e):
  status = return_status.UNHANDLED

  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_s21")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_s21")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p11_r2_over_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
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

@othogonal_state
def p_r2_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_r2_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    status = return_status.HANDLED
    (_e, _state) = r.meta_peel(e)  # search for INIT_META_SIGNAL
    print("d")
    print(_state)
    if _state is None or not r.has_a_child(p_r2_region, _state):
      status = r.trans(p_s21)
    else:
      status = r.trans(_state)
      print("p_r2_region init {}".format(_state))
      if _e is not None:
        r.post_fifo(_e)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_r2_region")
    status = return_status.HANDLED
  elif(e.signal == signals.region_exit):
    status = r.trans(p_r2_under_hidden_region)
  elif(e.signal == signals.INIT_META_SIGNAL):
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r2_under_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_r2_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.force_region_init):
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

@othogonal_state
def p_s21(r, e):
  status = return_status.UNHANDLED

  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_s21")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_s21")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r2_over_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_s22(r, e):
  status = return_status.UNHANDLED

  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_s22")
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e2")):
    status = r.trans(p_r2_final)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_s22")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r2_over_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
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

@state
def p(self, e):
  status = return_status.UNHANDLED

  # enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    if self.live_spy and self.instrumented:
      self.live_spy_callback("[p] {}".format(e.signal_name))
    (_e, _state) = self.meta_peel(e)  # search for INIT_META_SIGNAL
    print('b')
    if _state:
      self.inner._post_fifo(_e)
    pprint("enter p")
    self.inner.post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  # any event handled within there regions must be pushed from here
  elif(type(self.regions) == dict and (self.token_match(e.signal_name, "e1") or
      self.token_match(e.signal_name, "e2") or
      self.token_match(e.signal_name, "G1") or
      self.token_match(e.signal_name, self.regions['p_p11'].final_signal_name)
       )):
    if self.live_spy and self.instrumented:
      self.live_spy_callback("[p] {}".format(e.signal_name))
    self.inner.post_fifo(e)
    status = return_status.HANDLED
  # final token match
  elif(type(self.regions) == dict and self.token_match(e.signal_name,
    self.inner.final_signal_name)):
    if self.live_spy and self.instrumented:
      self.live_spy_callback("[p] {}".format('region_exit'))
    self.inner.post_fifo(Event(signal=signals.region_exit))
    #status = self.trans(some_other_state)
  elif e.signal == signals.INIT_META_SIGNAL:
    if self.live_spy and self.instrumented:
      self.live_spy_callback("[p] {}".format(e.signal_name))
    _e, _state = e.payload.event, e.payload.state
    print("c")
    print(_state)
    if _state:
      self.inner._post_fifo(_e)
    self.inner.meta_bounce_across(e)
    #self.regions['p'].post_lifo(Event(signal=signals.force_region_init))
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    if self.live_spy and self.instrumented:
      self.live_spy_callback("[p] {}".format('region_exit'))
    self.inner.post_lifo(Event(signal=signals.region_exit))
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

@state
def outer_state(self, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter outer_state")
    status = return_status.HANDLED
  elif(self.token_match(e.signal_name, "to_p")):
    if self.live_spy and self.instrumented:
      self.live_spy_callback("{}:outer_state".format(e.signal_name))
    status = self.trans(p)
  elif(self.token_match(e.signal_name, 'E0')):
    _e = self.meta_init(s=outer_state, t=p_p11_s12, sig=e.signal)
    print(payload_string(_e))
    _e, _state = _e.payload.event, _e.payload.state
    self.post_fifo(_e)
    status = self.trans(_state)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit outer_state")
    status = return_status.HANDLED
  else:
    self.temp.fun = self.bottom
    status = return_status.SUPER
  return status

if __name__ == '__main__':
  regression = True
  active_states = None
  example = XmlChart(
    name='x',
    log_file="/mnt/c/github/miros-xml/experiment/g1.log",
    live_trace=False,
    live_spy=True,
  )
  #example.instrumented = False
  example.instrumented = True
  example.start()
  time.sleep(0.20)
  example.report("\nstarting regression\n")

  if regression:
    def build_test(sig, expected_result, old_result, duration=0.2):
      '''test function, so it can be slow'''
      example.post_fifo(Event(signal=sig))
      if sig == 'G1':
        time.sleep(0.2)
        active_states = example.active_states()[:]
        string1 = "{:>39}{:>5} <-> {:<80}".format(str(old_result), sig, str(active_states))
        print(string1)
      time.sleep(duration)
      active_states = example.active_states()[:]
      string1 = "{:>39}{:>5} <-> {:<80}".format(str(old_result), sig, str(active_states))
      string2 = "\n{} <- {} == {}\n".format(str(old_result), sig, str(active_states))
      print(string1)
      example.report(string2)
      if active_states != expected_result:
        previous_frame = inspect.currentframe().f_back
        fdata = FrameData(*inspect.getframeinfo(previous_frame))
        function_name = '__main__'
        line_number   = fdata.line_number
        print("Assert error from {}:{}".format(function_name, line_number))
        print("From: {}->{}".format(sig, old_result))
        print("Expecting: {}".format(expected_result))
        print("Observed:  {}".format(active_states))
        #exit(0)
      #assert active_states == expected_result
      return active_states

    active_states = example.active_states()
    print("{:>39}{:>5} <-> {:<80}".format("start", "", str(active_states)))

    old_results = example.active_states()[:]

    old_results = build_test(
      sig='to_p',
      expected_result=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='E0',
      expected_result=[['p_p11_s12', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      duration=1000.2
    )


    #old_results = build_test(
    #  s='e1',
    #  expected_result=[['p_p11_s12', 'p_p11_s21'], 'p_s21'],
    #  old_result= old_results,
    #  duration=0.2
    #)

    #old_results = build_test(
    #  s='G1',
    #  expected_result=['p_r1_under_hidden_region', 'p_s22'],
    #  old_result= old_results,
    #  duration=1000.2
    #)


