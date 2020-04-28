
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
  return fn.__name__

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
  '''Reflect upon an event

    **Note**:
       If the event is a meta event, the resulting string will put each inner
       part of the onion on a new line, indented to distiguish it from the
       previous line.

       If the event provided is a normal event, its signal_name will be returned

    **Args**:
       | ``e`` (Event): The event to reflect upon


    **Returns**:
       (str): A string describing the event

    **Example(s)**:

    .. code-block:: python

       # ps has been aliased to payload_string
       print(ps(Event(signal=signals.hello_world))  # hello_world

  '''
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

# This is a debug function, so we want the name short
ps = payload_string

def pprint(value):
  pass
  # print(value)

def state(fn):
  '''Statechart state function wrapper, provides instrumentation and
  dynamically assigns the inner attribute'''
  @wraps(fn)
  def _state(chart, *args):
    fn_as_s = fn.__name__
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

    # these can be trampled as a side effect of a search (meta_init, meta_trans)
    # so make sure you salt their results away when you use these functions
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
    state_name = self.state_name
    state_fn = self.state_fn

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

    self.state_fn = state_fn
    self.state_name = state_name
    self.temp.fun = old_temp
    self.state.fun = old_fun
    return result

  @lru_cache(maxsize=32)
  def has_state(self, state):
    result = False

    old_temp = self.temp.fun
    old_fun = self.state.fun
    state_name = self.state_name
    state_fn = self.state_fn

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

    self.state_fn = state_fn
    self.state_name = state_name
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

  def _post_fifo(self, e):
    super().post_fifo(e)

  def _post_lifo(self, e):
    super().post_lifo(e)

  def _complete_circuit(self):
    super().complete_circuit()
  #def post_fifo(self, e):
  #  self._post_fifo(e)
  #  self.complete_circuit()

  #def post_lifo(self, e):
  #  self._post_lifo(e)
  #  self.complete_circuit()


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

    self.current_function_name = None  # dynamically assigned

  def regions_queues_string(self):
    '''Reflect upon all queues for all region objects in statechart

    **Returns**:
       (str): A reflection upon the queue contents for all regions

    **Example(s)**:

    .. code-block:: python

        # rqs is aliased to regions_queues_string
        print(self.rqs())

    '''
    previous_frame = inspect.currentframe().f_back
    fdata = FrameData(*inspect.getframeinfo(previous_frame))
    function_name = fdata.function_name
    line_number   = fdata.line_number

    width = 78
    result = ""

    loc_and_number_report = ">>>> {} {} <<<<".format(function_name, line_number)
    additional_space =  width - len(loc_and_number_report)
    result += "{}{}\n".format(loc_and_number_report, "<" * additional_space)
    result += "-" * int(width / 2) + "\n"

    for name, regions in self.regions.items():
      for region_index, region in enumerate(regions._regions):
        region_summary = ""
        _len = len(region.queue)
        region_summary = "{}, ql={}:".format(region.name, _len)
        region_summary = region_summary + " {}" if _len == 0 else region_summary
        result += "{}\n".format(region_summary)
        for index, e in enumerate(region.queue):
          _ps = ps(e)
          _ps = re.sub(r'([ ]+)(\[n\].+)', r'   \1\2', _ps)
          queue_summary = str(index) + ": " + _ps
          result += queue_summary + "\n"
        result += "-" * int(width / 2) + "\n"
    result += "<" * width + "\n"
    result += "\n"
    return result

  # This is a debug method, so we want the name short
  rqs = regions_queues_string

  def start(self):
    _instrumented = self.instrumented
    if self.live_spy:
      for key in self.regions.keys():
        self.regions[key].instrumented = self.instrumented
    else:
      for key in self.regions.keys():
        self.regions[key].instrumented = False

    for key in self.regions.keys():
      self.regions[key].start()

    self.start_at(outer_state)
    self.instrumented = _instrumented

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
  def meta_init(self, r, s, t, sig):
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
    inner = r.inner
    current_function_name = r.current_function_name

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

    r.inner = inner
    r.current_function_name = current_function_name

    return event

  def build_onion(self, t, sig, s=None):
    '''Find an list of state functions which describe a single gradient in the
    HHSM, from the source 's', to the target, 't'.

    **Note**:
       If it is not possible to draw line matching a single gradient between the
       two input functions.  The 's' is replaced with the lowest common ancestor
       of 't' and the provided 's' and the build_onion returns that list instead.

       The resulting list is in reverse.  I can't remember why I did it this
       way, and I'm not going to touch this "feature" right now.

    **Args**:
       | ``t`` (fn): the target state
       | ``sig`` (str): Event signal_name
       | ``s=None`` (fn): the source state


    **Returns**:
       (list): a reverse list describing a single gradient of state functions,
       from t to s (this is why it is reversed, it would be expected to be s to
       to to)

    **Example(s)**:

      To see the graph this example is referencing go to
      `this link <https://github.com/aleph2c/miros-xml/blob/master/doc/_static/xml_chart_4.pdf>`_

    .. code-block:: python


      result1 = example.build_onion(s=p, t=p_p12_p11_s12, sig='TEST')
      assert(result1 == [
        p_p12_p11_s12,
        p_p12_p11_r1_region,
        p_p12_p11,
        p_p12_r1_region,
        p_p12,
        p_r1_region,
        p,
      ])

      result2 = example.build_onion(t=p, s=p_p12_p11_s12, sig='TEST')
      assert(result2 == [
        p,
        p_r1_region,
        p_p12,
        p_p12_r1_region,
        p_p12_p11,
        p_p12_p11_r1_region,
        p_p12_p11_s12,
      ])


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
         | (tuple): (regions_state_function,
         |           injector,
         |           region_name)


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
        onion_states += [target_state, s]
        break
      if target_state:
        onion_states += [target_state, region_holding_state]
    if None in onion_states:
      new_target = s
      new_source = t
      onion_states = self.build_onion(s=new_source, t=new_target, sig=sig)
      onion_states.reverse()
    return onion_states

  @lru_cache(maxsize=64)
  def meta_trans(self, r, s, t, sig):
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
    inner = r.inner
    current_function_name = r.current_function_name

    # TODO: clean this up:

    source = s
    target = t
    #if sig == 'F1':
    #  import pdb; pdb.set_trace()
    lca = self.lca(source, target)
    #print("lca: ", lca)

    exit_onion1 = self.build_onion(
        t=source,
        s=lca,
        sig=sig)
    exit_onion = exit_onion1[1:]


    strippped_onion = []
    for fn in exit_onion:
      strippped_onion.append(fn)
      if fn == lca:
        break

    strippped_onion.reverse()
    exit_onion = strippped_onion[:]
    # exit_onion.reverse()

    entry_onion1 = self.build_onion(
        s=lca,
        t=target,
        sig=sig)[0:-1]

    entry_onion = entry_onion1[:]

    if entry_onion[-1] == exit_onion[-1]:
      bounce_type = signals.BOUNCE_SAME_META_SIGNAL
    else:
      bounce_type = signals.BOUNCE_ACROSS_META_SIGNAL

    # Wrap up the onion meta event from the inside out.  History items at the
    # last layer of the outer part of the INIT_META_SIGNAL need to reference an
    # even more outer part of the onion, the meta exit details.
    event = None
    for index, entry_target in enumerate(entry_onion):

      #signal_name = signals.INIT_META_SIGNAL if entry_target != lca else bounce_type
      signal_name = signals.INIT_META_SIGNAL

      if index == len(entry_onion) - 1:
        if (len(exit_onion) == 1 and
          exit_onion[0] == entry_onion[-1]):
            previous_state = source
            previous_signal = sig
        else:
          previous_state = exit_onion[0]
          previous_signal = signals.EXIT_META_SIGNAL
        event = Event(
          signal=signal_name,
          payload=META_SIGNAL_PAYLOAD(
            event=event,
            state=entry_target,
            previous_state=previous_state,
            previous_signal=previous_signal,
          )
        )
      else:
        previous_state = entry_onion[index + 1]
        previous_signal = signals.INIT_META_SIGNAL
        event = Event(
          signal=signal_name,
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

    if len(exit_onion) > 1:
      for index, exit_target in enumerate(exit_onion):
        signal_name = signals.EXIT_META_SIGNAL if exit_target != lca else bounce_type

        previous_signal = signals.EXIT_META_SIGNAL
        if index == len(exit_onion) - 1:
          previous_state = source
          previous_signal = sig
        else:
          previous_state = exit_onion[index + 1]

        event = Event(
          signal=signal_name,
          payload=META_SIGNAL_PAYLOAD(
            event=event,
            state=exit_target,
            previous_state=previous_state,
            previous_signal=previous_signal,
          )
        )
    else:
      previous_state = source
      previous_signal = sig
      event = Event(
        signal=bounce_type,
        payload=META_SIGNAL_PAYLOAD(
          event=event,
          state=exit_onion[0],
          previous_state=previous_state,
          previous_signal=previous_signal,
        )
      )

    r.inner = inner
    r.current_function_name = current_function_name
    if sig == 'F1':
      print(ps(event))
    return (t, event)

  @lru_cache(maxsize=32)
  def lca(self, _s, _t):
    s_onion = self.build_onion(_s, sig=None)[::-1]
    t_onion = self.build_onion(_t, sig=None)[::-1]
    _lca = list(set(s_onion).intersection(set(t_onion)))[-1]
    _lca = self.bottom if _lca is None else _lca
    return _lca

  @lru_cache(maxsize=32)
  def lci(self, s, t):
    '''return the least common injector'''
    s_onion = self.build_onion(s=s, t=t, sig=None)
    t_onion = self.build_onion(s=t, t=s, sig=None)
    _lci = list(set(s_onion).intersection(set(t_onion)))[0]
    _lci = self.bottom if _lci is None else _lci
    return _lci

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

################################################################################
#                                   p region                                   #
################################################################################

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
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)

    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.
    if _state == p_r1_region:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is a child of this state then transition to it
    if _state is None or not r.has_a_child(p_r1_region, _state):
      status = r.trans(p_p11)
    else:
      status = r.trans(_state)
      if _e is not None:
        r.post_fifo(_e)
  elif(e.signal == signals.EXIT_META_SIGNAL):
    (_e, _state) = e.payload.event, e.payload.state
    if r.has_a_child(p_r1_region, _state):
      r.outer.post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
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
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)
    if _state:
      r.inner._post_fifo(_e)
    r.inner.post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED

  # any event handled within there regions must be pushed from here
  elif(r.token_match(e.signal_name, "e1") or
       r.token_match(e.signal_name, "e2") or
       r.token_match(e.signal_name, "e4") or
       r.token_match(e.signal_name, "A") or
       r.token_match(e.signal_name, "F1") or
       r.token_match(e.signal_name, "G1") or
       r.token_match(e.signal_name, "G3")
       ):
    r.scribble(e.signal_name)
    r.inner.post_fifo(e)
    status = return_status.HANDLED
  elif r.token_match(e.signal_name, r.outmost.regions['p_p11'].final_signal_name):
    r.scribble(e.signal_name)
    status = r.trans(p_p12)
  elif r.token_match(e.signal_name, "C0"):
    status = r.trans(p_p12)
  elif e.signal == signals.EXIT_META_SIGNAL:
    (_e, _state) = e.payload.event, e.payload.state
    if r.has_a_child(p_p11, _state):
      # The next state is going to be our region handler skip it and post this
      # region handler would have posted to the outer HSM
      (_e, _state) = _e.payload.event, _e.payload.state
      r.outer.post_lifo(_e)
    status = return_status.HANDLED
  elif r.token_match(e.signal_name, "G0"):
    status = return_status.HANDLED
  elif e.signal == signals.exit_region:
    r.scribble(Event(e.signal_name))
    status = r.trans(p_r1_under_hidden_region)
  elif e.signal == signals.EXIT_SIGNAL:
    #scribble(Event(e.signal_name))
    r.inner.post_lifo(Event(signal=signals.exit_region))
    pprint("exit p_p11")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r1_over_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
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

@othogonal_state
def p_p11_r1_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_r1_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)

    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.
    if _state == p_p11_r1_region:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is a child of this state then transition to it
    if _state is None or not r.has_a_child(p_p11_r1_region, _state):
      status = r.trans(p_p11_s11)
    else:
      status = r.trans(_state)
      if _e is not None:
        r.post_fifo(_e)
  elif e.signal == signals.EXIT_META_SIGNAL:
    (_e, _state) = e.payload.event, e.payload.state
    if r.has_a_child(p_p11_r1_region, _state):
      # The next state is going to be our region handler skip it and post this
      # region handler would have posted to the outer HSM
      (_e, _state) = _e.payload.event, _e.payload.state
      r.outer.post_lifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
    status = r.trans(p_p11_r1_under_hidden_region)
  elif(e.signal == signals.INIT_META_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_r1_region")
    #r.inner.post_lifo(Event(signal=signals.exit_region))
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
  elif(r.token_match(e.signal_name, "e4")):
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
  # p_p11_s21
  elif(e.signal == signals.INIT_SIGNAL):
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)

    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.
    if _state == p_p11_r2_region:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is a child of this state then transition to it
    if _state is None or not r.has_a_child(p_p11_r2_region, _state):
      status = r.trans(p_p11_s21)
    else:
      status = r.trans(_state)
      if _e is not None:
        r.post_fifo(_e)
  elif(e.signal == signals.EXIT_META_SIGNAL):
    (_e, _state) = e.payload.event, e.payload.state
    if r.has_a_child(p_p11_r2_region, _state):
      r.outer.post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_META_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
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
  elif(r.token_match(e.signal_name, "e1")):
    status = r.trans(p_p11_s22)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_s21")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p11_r2_over_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_s22(r, e):
  '''
  '''
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_s22")
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "G3")):
    _e = r.outmost.meta_trans(r, t=p_s21, s=p_p11_s22, sig=e.signal_name)
    r.outer.post_lifo(_e)
    r.outmost.complete_circuit()
    #status = rr.trans(rr.fns['under_hidden_state_function'])
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "F2")):
    _e = r.meta_trans(r, t=p_p12_p11_s12, s=p_p11_s22, sig=e.signal_name)
    r.outer.post_lifo(_e)
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e2")):
    status = r.trans(p_p11_r2_final)
  elif(r.token_match(e.signal_name, "e1")):
    status = r.trans(p_p11_s21)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_s22")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p11_r2_over_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_r2_final(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p11_r2_final")
    r.final = True
    r.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_r2_final")
    r.final = False
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p11_r2_over_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
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

@othogonal_state
def p_p12(r, e):
  status = return_status.UNHANDLED

  # enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12")
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)
    if _state:
      r.inner._post_fifo(_e)
    r.inner.post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  # any event handled within there regions must be pushed from here
  elif(r.token_match(e.signal_name, "e1") or
       r.token_match(e.signal_name, "e2") or
       r.token_match(e.signal_name, "e4") or
       r.token_match(e.signal_name, "G1") or
       r.token_match(e.signal_name, "G3")
       ):
    r.scribble(e.signal_name)
    r.inner.post_fifo(e)
    status = return_status.HANDLED

  # All internal injectors will have to have this structure
  elif e.signal == signals.EXIT_META_SIGNAL:
    (_e, _state) = e.payload.event, e.payload.state
    if r.has_a_child(p_p12, _state):
      # The next state is going to be our region handler skip it and post this
      # region handler would have posted to the outer HSM
      (_e, _state) = _e.payload.event, _e.payload.state
      r.outer.post_lifo(_e)
    status = return_status.HANDLED
  # final token match
  elif(r.token_match(e.signal_name, r.outmost.regions['p_p12'].final_signal_name)):
    r.scribble(e.signal_name)
    status = r.trans(p_r1_final)
  elif(r.token_match(e.signal_name, "e5")):
    status = r.trans(p_r1_final)
  elif(e.signal == signals.E2):
    r.scribble(e.signal_name)
    _e = r.outmost.meta_init(r, t=p_p12_p11_s12, s=p_p12, sig=e.signal_name)
    # this force_region_init might be a problem
    r.inner._post_lifo(Event(signal=signals.force_region_init))
    r.inner.post_fifo(_e)
    #r.scribble(r.outmost.rqs())
    status = return_status.HANDLED
  # exit signals
  elif(e.signal == signals.exit_region):
    r.scribble(e.signal_name)
    #post_lifo(Event(signal=signals.exit_region))
    status = r.trans(p_r1_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    r.inner.post_lifo(Event(signal=signals.exit_region))
    pprint("exit p_p12")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r1_over_hidden_region
    status = return_status.SUPER
  return status

# inner parallel
@othogonal_state
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
@othogonal_state
def p_p12_r1_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_r1_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)

    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.
    if _state == p_p12_r1_region:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is a child of this state then transition to it
    if _state is None or not r.has_a_child(p_p12_r1_region, _state):
      status = r.trans(p_p12_p11)
    else:
      status = r.trans(_state)
      if _e is not None:
        r.post_fifo(_e)
  elif(e.signal == signals.EXIT_META_SIGNAL):
    (_e, _state) = e.payload.event, e.payload.state
    if r.has_a_child(p_p12_r1_region, _state):
      r.outer.post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
    status = r.trans(p_p12_r1_under_hidden_region)
  elif(e.signal == signals.INIT_META_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_r1_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p12_r1_under_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_r1_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.force_region_init):
    status = r.trans(p_p12_r1_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_r1_over_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_r1_over_hidden_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p12_r1_region
    status = return_status.SUPER
  return status


# inner parallel
@othogonal_state
def p_p12_p11(r, e):
  status = return_status.UNHANDLED

  # enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_p11")
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)
    if _state:
      r.inner._post_fifo(_e)
    r.inner.post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "G1")):
    r.scribble(e.signal_name)
    r.inner.post_fifo(e)
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, r.outmost.regions['p_p12_p11'].final_signal_name)):
    r.scribble(e.signal_name)
    status = r.trans(p_p12_s12)
  elif e.signal == signals.EXIT_META_SIGNAL:
    (_e, _state) = e.payload.event, e.payload.state
    if r.has_a_child(p_p12_p11, _state):
      # The next state is going to be our region handler skip it and post this
      # region handler would have posted to the outer HSM
      (_e, _state) = _e.payload.event, _e.payload.state
      r.outer.post_lifo(_e)
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e4")):
    status = r.trans(p_p12_s12)
  elif(e.signal == signals.exit_region):
    r.scribble(e.signal_name)
    status = r.trans(p_p12_r1_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_p11")
    r.inner.post_lifo(Event(signal=signals.exit_region))
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p12_r1_over_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
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

@othogonal_state
def p_p12_p11_r1_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_p11_r1_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)

    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.
    if _state == p_p12_p11_r1_region:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is a child of this state then transition to it
    if _state is None or not r.has_a_child(p_p12_p11_r1_region, _state):
      status = r.trans(p_p12_p11_s11)
    else:
      status = r.trans(_state)
      if _e is not None:
        r.post_fifo(_e)
  elif(e.signal == signals.INIT_META_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_META_SIGNAL):
    (_e, _state) = e.payload.event, e.payload.state
    if r.has_a_child(p_p12_p11_r1_region, _state):
      r.outer.post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_p11_r1_region")
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
    status = r.trans(p_p12_p11_r1_under_hidden_region)
  elif(e.signal == signals.INIT_META_SIGNAL):
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p12_p11_r1_under_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_p11_r1_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.force_region_init):
    status = r.trans(p_p12_p11_r1_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_p11_r1_over_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_p11_r1_over_hidden_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p12_p11_r1_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_p11_s11(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_p11_s11")
    status = return_status.HANDLED
  elif(e.signal == signals.e1):
    status = r.trans(p_p12_p11_s12)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_p11_s11")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p12_p11_r1_over_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_p11_s12(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_p11_s12")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_p11_s12")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p12_p11_r1_over_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_p11_r2_under_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(r.token_match(e.signal_name, "enter_region")):
    status = r.trans(p_p12_p11_r2_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_p11_r2_under_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_p11_r2_under_hidden_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = r.bottom
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_p11_r2_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_p11_r2_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)

    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.
    if _state == p_p12_p11_r2_region:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is a child of this state then transition to it
    if _state is None or not r.has_a_child(p_p11_r1_region, _state):
      status = r.trans(p_p12_p11_s21)
    else:
      status = r.trans(_state)
      if _e is not None:
        r.post_fifo(_e)
  elif(e.signal == signals.EXIT_META_SIGNAL):
    (_e, _state) = e.payload.event, e.payload.state
    if r.has_a_child(p_p12_p11_r2_region, _state):
      r.outer.post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
    status = r.trans(p_p12_p11_r2_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_p11_r2_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_META_SIGNAL):
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p12_p11_r2_under_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_p11_r2_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.force_region_init):
    status = r.trans(p_p12_p11_r2_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_p11_r2_over_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_p11_r2_over_hidden_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p12_p11_r2_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_p11_s21(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_p11_s21")
    status = return_status.HANDLED
  elif(e.signal == signals.G1):
    _state, _e = r.outmost.meta_trans(
      r,
      s=p_p12_p11_s21,
      t=p_p22_s11,
      sig=e.signal_name
    )
    r.same.post_fifo(_e)
    status = return_status.UNHANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_p11_s21")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p12_p11_r2_over_hidden_region
    status = return_status.SUPER
  return status

# inner parallel
@othogonal_state
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
@othogonal_state
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

@othogonal_state
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
@othogonal_state
def p_p12_r2_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_r2_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)

    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.
    if _state == p_p12_r2_region:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is a child of this state then transition to it
    if _state is None or not r.has_a_child(p_p11_r1_region, _state):
      status = r.trans(p_p12_s21)
    else:
      status = r.trans(_state)
      if _e is not None:
        r.post_fifo(_e)
  elif(e.signal == signals.EXIT_META_SIGNAL):
    (_e, _state) = e.payload.event, e.payload.state
    if r.has_a_child(p_p12_r2_region, _state):
      r.outer.post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
    status = r.trans(p_p12_r2_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_r2_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_META_SIGNAL):
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p12_r2_under_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_r2_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.force_region_init):
    status = r.trans(p_p12_r2_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_r2_over_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p11_r2_over_hidden_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p12_r2_region
    status = return_status.SUPER
  return status

# inner parallel
@othogonal_state
def p_p12_s21(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_s21")
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e1")):
    status = r.trans(p_p12_s22)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_s21")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p12_r2_over_hidden_region
    status = return_status.SUPER
  return status

# inner parallel
@othogonal_state
def p_p12_s22(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_s22")
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e2")):
    status = r.trans(p_p12_r2_final)
  elif(r.token_match(e.signal_name, "e1")):
    status = r.trans(p_p12_s21)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_s22")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p12_r2_over_hidden_region
    status = return_status.SUPER
  return status

# inner parallel
@othogonal_state
def p_p12_r2_final(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p12_r2_final")
    r.final = True
    r.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p12_r2_final")
    r.final = False
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p12_r2_over_hidden_region
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
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)

    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.
    if _state == p_r2_region:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is a child of this state then transition to it
    if _state is None or not r.has_a_child(p_r2_region, _state):
      status = r.trans(p_s21)
    else:
      status = r.trans(_state)
      if _e is not None:
        r.post_fifo(_e)
  elif e.signal == signals.BOUNCE_SAME_META_SIGNAL:
    r.scribble(e.signal_name)
    _e, _state = e.payload.event, e.payload.state
    for region in r.outmost.regions['p']._regions:
      if region.has_state(e.payload.previous_state):
        region._post_fifo(_e)
        region._post_lifo(Event(signal=signals.enter_region))
      else:
        region._post_fifo(Event(signal=signals.Ignore))
    r.outmost.regions['p']._complete_circuit()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_META_SIGNAL):
    (_e, _state) = e.payload.event, e.payload.state
    if r.has_a_child(p_r2_region, _state):
      r.outer.post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_r2_region")
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
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
  elif(r.token_match(e.signal_name, "C0")):
    status = r.trans(p_p22)
  elif(r.token_match(e.signal_name, "F1")):
    _state, _e = r.outmost.meta_trans(r, t=p_p22_s11, s=p_s21, sig=e.signal_name)
    for region in r.same._regions:
      print(region.name)
      #region._post_fifo(Event(signal=signals.force_region_init))
      region._post_lifo(_e)
      region._complete_circuit()
    status = r.trans(p_p22)
  elif(r.token_match(e.signal_name, "G0")):
    _state, _e = r.outmost.meta_trans(
      r,
      s=p_s21,
      t=p_p12_p11_s21,
      sig=e.signal_name
    )
    r.same.post_lifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_s21")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_r2_over_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22(r, e):
  status = return_status.UNHANDLED

  # enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)
    if _state:
      print(ps(_e))
      r.inner._post_fifo(_e)
    r.inner.post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  # any event handled within there regions must be pushed from here
  elif(r.token_match(e.signal_name, "e1") or
       r.token_match(e.signal_name, "e2") or
       r.token_match(e.signal_name, "e4") or
       r.token_match(e.signal_name, "E2")
       ):
    r.scribble(e.signal_name)
    r.inner.post_fifo(e)
    status = return_status.HANDLED
  # final token match
  elif(r.token_match(e.signal_name, r.outmost.regions['p_p22'].final_signal_name)):
    r.scribble(e.signal_name)
    status = r.trans(p_r2_final)
  elif(e.signal == signals.EXIT_SIGNAL):
    r.inner.post_lifo(Event(signal=signals.exit_region))
    pprint("exit p_p22")
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
    r.scribble(e.signal_name)
    status = r.trans(p_r2_under_hidden_region)
  else:
    r.temp.fun = p_r2_over_hidden_region
    status = return_status.SUPER
  return status

# inner parallel
@othogonal_state
def p_p22_r1_under_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(r.token_match(e.signal_name, "enter_region")):
    status = r.trans(p_p22_r1_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_r1_under_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_r1_under_hidden_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = r.bottom
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22_r1_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_r1_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)

    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.
    if _state == p_p22_r1_region:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is a child of this state then transition to it
    if _state is None or not r.has_a_child(p_p22_r1_region, _state):
      status = r.trans(p_p22_s11)
    else:
      print(_state)
      status = r.trans(_state)
      if _e is not None:
        r.post_fifo(_e)
  # can we get rid of exit_region?
  elif(e.signal == signals.exit_region):
    status = r.trans(p_p22_r1_under_hidden_region)
  elif(e.signal == signals.INIT_META_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_r1_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p22_r1_under_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22_r1_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.force_region_init):
    status = r.trans(p_p22_r1_under_hidden_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_r1_over_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_r1_over_hidden_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p22_r1_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22_s11(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_s11")
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e4")):
    status = r.trans(p_p22_s12)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_s11")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p22_r1_over_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22_s12(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_s12")
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e1")):
    status = r.trans(p_p22_r1_final)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_s12")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p22_r1_over_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
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
@othogonal_state
def p_p22_r2_under_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(r.token_match(e.signal_name, "enter_region")):
    status = r.trans(p_p22_r2_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_r2_under_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_r2_under_hidden_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = r.bottom
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22_r2_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_r2_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)

    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.
    if _state == p_p22_r2_region:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is a child of this state then transition to it
    if _state is None or not r.has_a_child(p_p22_r2_region, _state):
      status = r.trans(p_p22_s21)
    else:
      status = r.trans(_state)
      if _e is not None:
        r.post_fifo(_e)
  elif(e.signal == signals.EXIT_META_SIGNAL):
    (_e, _state) = e.payload.event, e.payload.state
    if r.has_a_child(p_p22_r2_region, _state):
      r.outer.post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
    status = r.trans(p_p22_r2_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_r2_region")
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_META_SIGNAL):
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p22_r2_under_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22_r2_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.force_region_init):
    status = r.trans(p_p22_r2_under_hidden_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_r2_over_hidden_region")
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_r2_over_hidden_region")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p22_r2_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22_s21(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_s21")
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e1")):
    status = r.trans(p_p22_s22)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_s21")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p22_r2_over_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22_s22(r, e):
  status = return_status.UNHANDLED
  if(e.signal == signals.ENTRY_SIGNAL):
    pprint("enter p_p22_s22")
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e2")):
    status = r.trans(p_p22_r2_final)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p_p22_s22")
    status = return_status.HANDLED
  else:
    r.temp.fun = p_p22_r2_over_hidden_region
    status = return_status.SUPER
  return status

@othogonal_state
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
    _e = self.meta_init(r=self, t=p_p11_s22, s=outer_state, sig=e.signal_name)
    self.post_fifo(_e.payload.event)
    #self.live_spy_callback(ps(_e))
    #self.complete_circuit()
    status = self.trans(_e.payload.state)
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit outer_state")
    status = return_status.HANDLED
  else:
    self.temp.fun = self.bottom
    status = return_status.SUPER
  return status

@state
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
    _e = self.meta_init(r=self, t=p_p22_s21, sig=e.signal_name)
    self.post_fifo(_e.payload.event)
    status = self.trans(_e.payload.state)
  else:
    self.temp.fun = outer_state
    status = return_status.SUPER
  return status

@state
def p(self, e):
  status = return_status.UNHANDLED

  # enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    self.scribble("[p] {}".format(e.signal_name))
    (_e, _state) = self.meta_peel(e)  # search for INIT_META_SIGNAL
    if _state:
      self.inner._post_fifo(_e)
    pprint("enter p")
    self.inner.post_lifo(Event(signal=signals.enter_region))
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
      self.token_match(e.signal_name, "G1") or
      self.token_match(e.signal_name, self.regions['p_p11'].final_signal_name) or
      self.token_match(e.signal_name, self.regions['p_p12'].final_signal_name) or
      self.token_match(e.signal_name, self.regions['p_p22'].final_signal_name)
  )):
    self.scribble("[p] {}".format(e.signal_name))
    self.inner.post_fifo(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_META_SIGNAL):
    # hasn't been updated
    self.regions['p']._post_lifo(Event(signal=signals.force_region_init))
    self.regions['p'].post_fifo(e.payload.event.payload.event)
    status = return_status.HANDLED
  elif e.signal == signals.BOUNCE_ACROSS_META_SIGNAL:
    self.scribble("[p] {}".format(e.signal_name))
    _e, _state = e.payload.event, e.payload.state
    self.inner._post_fifo(_e)
    for region in self.inner._regions:
      if region.has_state(e.payload.previous_state):
        region.pop_event()
        region._post_lifo(Event(signal=signals.exit_region))
      else:
        region.post_lifo(Event(signal=signals.enter_region))
    [region.complete_circuit() for region in self.inner._regions]
    status = return_status.HANDLED
  elif(self.token_match(e.signal_name, "E1")):
    self.scribble("[p] {}".format(e.signal_name))
    _e = self.meta_init(r=self, s=p, t=p_p11_s12, sig=e.signal)
    self.scribble(payload_string(_e))
    self.inner._post_lifo(Event(signal=signals.force_region_init))
    self.inner.post_fifo(_e)
    status = return_status.HANDLED
  #elif e.signal == signals.BOUNCE_SAME_META_SIGNAL:
    #self.scribble("[p] {}".format(e.signal_name))
    #_e, _state = e.payload.event, e.payload.state
    #self.inner._post_fifo(_e)
    #for region in self.inner._regions:
    #  if region.has_state(e.payload.previous_state):
    #    region.post_lifo(Event(signal=signals.enter_region))
    #  else:
    #    region.pop_event()
    #    region._post_lifo(Event(signal=signals.exit_region))
    #print(self.rqs())
    #[region.complete_circuit() for region in self.inner._regions]
  # status = return_status.HANDLED
  elif(self.token_match(e.signal_name, "E1")):
    self.scribble("[p] {}".format(e.signal_name))
    _e = self.meta_init(r=self, s=p, t=p_p11_s12, sig=e.signal)
    self.scribble(payload_string(_e))
    self.inner._post_lifo(Event(signal=signals.force_region_init))
    self.inner.post_fifo(_e)
    status = return_status.HANDLED
  # final token match
  elif(type(self.regions) == dict and self.token_match(e.signal_name,
    self.regions['p'].final_signal_name)):
    self.scribble("[p] {}".format('exit_region'))
    self.regions['p'].post_fifo(Event(signal=signals.exit_region))
    status = self.trans(some_other_state)
  elif(self.token_match(e.signal_name, "to_o")):
    status = self.trans(outer_state)
  elif(self.token_match(e.signal_name, "to_s")):
    status = self.trans(some_other_state)
  elif(self.token_match(e.signal_name, "A1")):
    status = self.trans(p)
  elif(e.signal == signals.EXIT_META_SIGNAL):
    self.regions['p']._post_lifo(Event(signal=signals.force_region_exit))
    self.regions['p']._complete_circuit()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    pprint("exit p")
    self.scribble("[p] {}".format('exit_region'))
    self.inner.post_lifo(Event(signal=signals.exit_region))
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
    self.scribble("[p] {}".format('exit_region'))
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
    log_file="/mnt/c/github/miros-xml/experiment/4th_example.log",
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
      #if sig == 'G1':
      #  active_states = example.active_states()[:]
      #  string1 = "{:>39}{:>5} <-> {:<80}".format(str(old_result), sig, str(active_states))
      #  print(string1)
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
        import pdb; pdb.set_trace()
        example.active_states()
        time.sleep(1000)
        #exit(0)
      #assert active_states == expected_result
      return active_states

    def build_reflect(sig, expected_result, old_result, duration=0.2):
      '''test function, so it can be slow'''
      example.post_fifo(Event(signal=sig))
      #if sig == 'G1':
      #  active_states = example.active_states()[:]
      #  string1 = "{:>39}{:>5} <-> {:<80}".format(str(old_result), sig, str(active_states))
      #  print(string1)
      time.sleep(duration)
      active_states = example.active_states()[:]
      string1 = "{:>39}{:>5} <-> {:<80}".format(str(old_result), sig, str(active_states))
      string2 = "\n{} <- {} == {}\n".format(str(old_result), sig, str(active_states))
      print(string1)
      example.report(string2)
      #assert active_states == expected_result
      return active_states

    result1 = example.build_onion(s=p, t=p_p11, sig='TEST')
    assert(result1 == [p_p11, p_r1_region, p])
    result2 = example.build_onion(s=p_p11, t=p, sig='TEST')
    assert(result2 == [p, p_r1_region, p_p11])

    result1 = example.build_onion(s=p, t=p_p11_s11, sig='TEST')
    assert(result1 == [p_p11_s11, p_p11_r1_region, p_p11, p_r1_region, p])
    result2 = example.build_onion(s=p_p11_s11, t=p, sig='TEST')
    assert(result2 == [p, p_r1_region, p_p11, p_p11_r1_region, p_p11_s11])

    result1 = example.build_onion(s=p, t=p_p12_p11_s12, sig='TEST')
    assert(result1 == [
      p_p12_p11_s12,
      p_p12_p11_r1_region,
      p_p12_p11,
      p_p12_r1_region,
      p_p12,
      p_r1_region,
      p,
    ])

    result2 = example.build_onion(t=p, s=p_p12_p11_s12, sig='TEST')
    assert(result2 == [
      p,
      p_r1_region,
      p_p12,
      p_p12_r1_region,
      p_p12_p11,
      p_p12_p11_r1_region,
      p_p12_p11_s12,
    ])

    result1 = example.build_onion(s=p_p11, t=p_p12, sig='TEST')
    assert(result1 == [
      p_p12,
      p_r1_region,
      p,
    ])
    result2 = example.build_onion(t=p_p11, s=p_p12, sig='TEST')
    pp(result1)
    pp(result2)
    print(example.lci(s=p_p11, t=p_p12))

    #result2 = example.build_onion(s=p_p11_s11, t=p, sig='TEST')
    #assert(result2 == [p, p_r1_region, p_p11, p_p11_r1_region, p_p11_s11])

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
      sig='to_p',
      expected_result=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='e4',
      expected_result=[['p_p11_s12', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='e1',
      expected_result=[['p_p11_r1_final', 'p_p11_s22'], 'p_s21'],
      old_result= old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='e2',
      expected_result=[[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_s21'],
      old_result= old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='C0',
      expected_result=[[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], ['p_p22_s11', 'p_p22_s21']],
      old_result= old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='e4',
      expected_result=[['p_p12_s12', 'p_p12_s21'], ['p_p22_s12', 'p_p22_s21']],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='e1',
      expected_result=[['p_p12_r1_final', 'p_p12_s22'], ['p_p22_r1_final', 'p_p22_s22']],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='e2',
      expected_result=['some_other_state'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='to_p',
      expected_result=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='e4',
      expected_result=[['p_p11_s12', 'p_p11_s21'], 'p_s21'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='e1',
      expected_result=[['p_p11_r1_final', 'p_p11_s22'], 'p_s21'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='to_o',
      expected_result=['outer_state'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='E0',
      expected_result=[['p_p11_s11', 'p_p11_s22'], 'p_s21'],
      old_result= old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='E1',
      expected_result=[['p_p11_s12', 'p_p11_s21'], 'p_s21'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='E2',
      expected_result=[['p_p11_s12', 'p_p11_s21'], 'p_s21'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='C0',
      expected_result=[[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], ['p_p22_s11', 'p_p22_s21']],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='E2',
      expected_result=[[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s21'], ['p_p22_s11', 'p_p22_s21']],
      old_result=old_results,
      duration=0.2
    )


    old_results = build_test(
      sig='E0',
      expected_result=[['p_p11_s11', 'p_p11_s22'], 'p_s21'],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='C0',
      expected_result=[[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], ['p_p22_s11', 'p_p22_s21']],
      old_result=old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='G1',
      expected_result=['p_r1_under_hidden_region', ['p_p22_s11', 'p_p22_s21']],
      old_result= old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='A1',
      expected_result=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='G0',
      expected_result=[[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_r2_under_hidden_region'],
      old_result= old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='A1',
      expected_result=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      duration=0.2
    )

    old_results = build_test(
      sig='F1',
      expected_result=[['p_p11_s11', 'p_p11_s21'], ['p_p22_s11', 'p_p22_s21']],
      old_result=old_results,
      duration=0.2
    )

    #old_results = build_test(
    #  sig='to_p_s22',
    #  expected_result=['p_r1_under_hidden_region', 'p_s22'],
    #  old_result= old_results,
    #  duration=0.2
    #)

    #old_results = build_test(
    #  sig='G0',
    #  expected_result=[['p_p11_s12', 'p_p11_s21'], 'p_r2_under_hidden_region'],
    #  old_result= old_results,
    #  duration=0.2
    #)


