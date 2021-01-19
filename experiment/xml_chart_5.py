# For diagram see /doc/_static/xml_chart_5.pdf
#
# To continuously test:
# while inotifywait -e modify xml_chart_5.py logger_config.yaml; do python xml_chart_5.py; done
import re
import sys
import time
import yaml
import inspect
import logging
import pprint as xprint
from functools import wraps
from functools import partial
from functools import lru_cache
from collections import namedtuple

# writing it this way to avoid pyflake errors
from logging import config as logging_config

from miros import Event
from miros import spy_on
from miros import signals
from miros import ActiveObject
from miros import return_status
from miros import HsmWithQueues

# To create a deconstruction of a given event:
event_to_investigate = '_SRE3'

reduce_test_size = True

# 'module_namespace' will be used to add functions to the globals() namespace
module_namespace = sys.modules[__name__]

# used for testing
state_memory = []

def csm():
  '''clear state memory (TESTING FUNCTION)'''
  global state_memory
  state_memory = []

def rsm(state, event):
  '''record state memory (TESTING FUNCTION)'''
  state_memory.append((state.__name__, event.signal_name))

def diff_state_memory(expected, observed):
  '''diff expected and observed state memory

    **Note**:

       This diff the contents of the expected state/signal_name memory
       against the observed state/signal_name memory (TESTING FUNCTION)

    **Args**:
       | ``expected`` (list): a list of state to signal_name tuples
       | ``observed``   (list): a list of state to signal_name tuples


    **Returns**:
       (tuple): (result (True if matched, False otherwise),
                 diff,
                 observed,
                 expected)

    **Example(s)**:

    .. code-block:: python

        state_memory =\
          [('p_p11_s21', 'EXIT_SIGNAL'),
           ('p_p11_s22', 'ENTRY_SIGNAL'),
           ('p_p11_s22', 'INIT_SIGNAL')]
        expected =\
          [('p_p11_s21', 'EXIT_SIGNAL'),
           ('p_p11_s22', 'INIT_SIGNAL')]

        result, diff, observed, expected = diff_state_memory(
          expected_rtc_state_signal_pairs)

        assert(result==False)
        assert(diff==[('p_p11_s22', 'ENTRY_SIGNAL')])
        assert(observed==\
          [('p_p11_s21', 'EXIT_SIGNAL'),
           ('p_p11_s22', 'ENTRY_SIGNAL'),
           ('p_p11_s22', 'INIT_SIGNAL')])
        assert(expected_rtc_state_signal_pairs==\
          [('p_p11_s21', 'EXIT_SIGNAL'),
           ('p_p11_s22', 'INIT_SIGNAL')])

  '''
  global state_memory
  expected_rtc_set = set(expected)
  observed_rtc = observed
  #diff = [x for x in observed_rtc if x not in expected_rtc_set]

  diff = []

  largest = observed[:] if len(observed) >= len(expected) else expected[:]
  for i in range(len(largest)):
    if i < len(observed) and i < len(expected):
      observed_state, observed_signal = observed[i]
      expected_state, expected_signal = expected[i]
      if(observed_state != expected_state or
         observed_signal != observed_signal):
        diff.append((observed_state, observed_signal))

    if i >= len(observed):
      # missing items from observed
      expected_state, expected_signal = expected[i]
      diff.append(("too_little_in_observed----" + expected_state, expected_signal))
    elif i >= len(expected):
      diff.append(("too_much_in_observed++++" + observed_state, observed_signal))

  # return (True/False, observed, expected)
  # True if expected_rtc and observed_rtc lists of tuples match
  return (True, diff, observed_rtc, expected) if \
      (len(diff) == 0 and len(expected) == len(observed)) else \
      (False, diff, observed_rtc, expected)

# credit: https://stackoverflow.com/a/9580006 (Shane Holloway)
def find_decorators(target):
  '''DESIGN HELPER FUNCTION, NOT USED IN PRODUCTION CODE'''
  import ast, inspect
  res = {}
  def visit_FunctionDef(node):
    res[node.name] = [ast.dump(e) for e in node.decorator_list]

  V = ast.NodeVisitor()
  V.visit_FunctionDef = visit_FunctionDef
  V.visit(compile(inspect.getsource(target), "?", 'exec', ast.PyCF_ONLY_AST))
  return res

def pp(item):
  '''DESIGN HELPER FUNCTION, NOT USED IN PRODUCTION CODE'''
  xprint.pprint(item)

META_SIGNAL_PAYLOAD = namedtuple("META_SIGNAL_PAYLOAD",
  ['n', 'event', 'state', 'previous_state', 'previous_signal', 'springer'])

META_HOOK_PAYLOAD = namedtuple("META_SIGNAL_PAYLOAD",
  ['event'])

FrameData = namedtuple('FrameData', [
  'filename',
  'line_number',
  'function_name',
  'lines',
  'index'])

SEARCH_FOR_SUPER_SIGNAL = signals.SEARCH_FOR_SUPER_SIGNAL

################################################################################
#                              MIXINS                                          #
################################################################################
def flatten(l, a=None):
  if a is None:
    a = []
  for i in l:
    if isinstance(i, list):
      flatten(i, a)
    else:
      a.append(i)
  return a

def meta_trans(hsm, e, s, t):
  fn = hsm.outmost.meta_hooked(
    s=s,
    e=e
  )
  if fn is not None:
    status = fn(hsm, e)
  elif(in_same_hsm(s, t)):
    status = hsm.trans(t)
  else:
    _state, _e = hsm.outmost._meta_trans(
      hsm,
      t=t,
      s=s,
      sig=e.signal_name)
    if _state:
      status = hsm.trans(_state)
      hsm.same._post_fifo(_e)
      investigate(hsm, e, _e)
    else:
      status = return_status.HANDLED
      investigate(hsm, e, _e)
      hsm.same.post_fifo(_e)
  return status

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

def cache_clear():
  '''Clear the cached values of all caching methods and functions used by this package.'''
  s_to_s.cache_clear()
  Region.cache_clear()
  XmlChart.cache_clear()

def proto_investigate(r, e, _e=None, springer=None):
  '''Used for WTF investigations

    **Note**:
       Steps (takes about 10 minutes to set this up):
       1. print(ps(_e)) to see where the meta event is suppose to go
       2. place markers at each step, track in notes
       3. place an investigate call at each step

    **Args**:
       | ``r`` (Region): region
       | ``e`` (Event): initial event
       | ``springer=None`` (str): signal name of event that started the meta event
       | ``_e=None`` (Event): subsequent event

    **Example(s)**:

    .. code-block:: python

        # place this on every marker, the look at the logs
        investigate(r, e, 'I1', _e)

  '''
  if springer is not None:
    springer_in_event = None
    if (hasattr(_e, 'payload') and hasattr(_e.payload, 'springer')):
      springer_in_event = _e.payload.springer

    springer_in_event = "" if springer_in_event is None else \
      str(springer_in_event)

    matched = re.match(springer, springer_in_event)
    if matched:
      if hasattr(e, 'payload') and hasattr(e.payload, 'n'):
        n = e.payload.n
      else:
        if hasattr(_e, 'payload') and hasattr(_e.payload, 'n'):
          n = _e.payload.n
        else:
          n = 0
      r.scribble("{}: {}".format(n, r.outmost.rqs()))
      r.scribble('--'*10)
      r.scribble("{}: {}".format(n, r.outmost.active_states()))
      r.scribble('--'*10)
      r.scribble("{}: {}".format(n, ps(e)))
      if _e is not None:
        r.scribble("{}: {}".format(n, ps(_e)))
      r.scribble('--'*10)

investigate = partial(proto_investigate, springer=event_to_investigate)


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
      result += "{}[n={}]::{}:{} [n={}]::{}:{} ->\n".format(
        tabs,
        e.payload.n,
        e.signal_name,
        f_to_s(e.payload.state),
        e.payload.n - 1,
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

class MiniHsm():
  def __init__(self):
    class Empty:
      pass

    def __leaf__(self, *args, **kwargs):
      '''top/bottom-most state given to all HSMs'''
      return return_status.IGNORED

    self.temp = Empty
    self.temp.fun = None
    self.bottom = __leaf__
    self.top = __leaf__
    self.instrumented = False

@lru_cache(maxsize=128)
def within(outer, inner):
  hsm = MiniHsm()
  result = False
  super_e = Event(signal=SEARCH_FOR_SUPER_SIGNAL)
  hsm.temp.fun = inner

  # Search from the 'inner' state, outward;
  # either you will get find the 'outer' state or you will get to the bottom of
  # the HSM.
  while(True):
    if(hsm.temp.fun.__name__ == outer.__name__):
      result = True
      r = return_status.IGNORED
    else:
      r = hsm.temp.fun(hsm, super_e)
    if r == return_status.IGNORED:
      break
  return result

@lru_cache(maxsize=128)
def in_same_hsm(state1, state2):
  hsm = MiniHsm()  # only lets you search outward
  result = False
  hsm.temp.fun = state1

  super_e = Event(signal=SEARCH_FOR_SUPER_SIGNAL)
  temp, outer_state = state1, state1
  while(True):
    r = hsm.temp.fun(hsm, super_e)
    if r == return_status.IGNORED:
      break
    outer_state = temp
    temp = hsm.temp.fun
  return within(outer_state, state2)

################################################################################
#                            DECORATORS                                        #
################################################################################
def state(fn):
  '''Statechart state function wrapper, provides instrumentation and
  dynamically assigns the inner attribute'''

  @wraps(fn)
  def _state(chart, *args):
    fn_as_s = fn.__name__

    if len(args) == 1:
      e = args[0]
    else:
      e = args[-1]

    if hasattr(chart, 'regions'):
      if fn_as_s not in chart.regions:
        chart.inner = chart
      else:
        chart.inner = chart.regions[fn_as_s]
    chart.current_function_name = fn_as_s

    chart.state_name = fn_as_s
    chart.state_fn = fn

    if(e.signal == signals.REFLECTION_SIGNAL):
      # We are no longer going to return a ReturnStatus object
      # instead we write the function name as a string
      status = fn_as_s
      return status

    #status = spy_on(fn)(chart, *args)
    status = fn(chart, e)
    return status
  return _state


def othogonal_state(fn):
  '''Othogonal component state function wrapper, provides instrumentation and
  dynamically assigns the inner attribute.'''
  @wraps(fn)
  def _pspy_on(region, *args):

    if type(region) == XmlChart:
      return state(fn)(region, *args)

    # dynamically assign the inner attribute
    if hasattr(region, 'inners'):
      fn_as_s = f_to_s(fn)
      if fn_as_s not in region.inners:
        region.inners[fn_as_s] = None
        if fn_as_s in region.outmost.regions:
          region.inners[fn_as_s] = region.outmost.regions[fn_as_s]

      # these can be trampled as a side effect of a search (_meta_trans)
      # so make sure you salt their results away when you use these functions
      region.inner = region.inners[fn_as_s]
      region.current_function_name = fn_as_s

    # instrument the region
    if region.instrumented:
      status = spy_on(fn)(region, *args)  # call to state function here
      region.rtc.spy.clear()
    else:
      e = args[0] if len(args) == 1 else args[-1]
      status = fn(region, e)  # call to state function here
    return status

  return _pspy_on

################################################################################
#                             CLASSES                                          #
################################################################################
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

  @property
  def final_signal_name(self):
    return self.final_event.signal_name

  def scribble(self, string):
    '''Add some state context to the spy instrumention'''
    self.outmost.scribble("[{}] {}".format(
      self.current_function_name, string)
    )

  def _scribble(self, string):
    self.outmost._scribble("[{}] {}".format(
      self.current_function_name, string)
    )

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
    if other is None:
      result = False
    else:
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

  def p_spy(self, e):
    self.outmost.live_spy_callback(
      '{}:{}'.format(e.signal_name, self.state_name)
    )

  def _p_spy(self, e):
    string = "[x] {}:{}".format(e.signal_name, self.state_name)
    self.outmost._scribble(string)

  @lru_cache(maxsize=32)
  def has_state(self, state):
    '''Determine if this region has a state.

    **Note**:
       Since the state functions can be decorated, this method compares the
       names of the functions and note their addresses.

    **Args**:
       | ``query`` (fn): a state function


    **Returns**:
       (bool): True | False

    '''
    result = False

    old_temp = self.temp.fun
    old_fun = self.state.fun
    state_name = self.state_name
    state_fn = self.state_fn

    self.temp.fun = state
    super_e = Event(signal=SEARCH_FOR_SUPER_SIGNAL)
    while(True):
      if(self.temp.fun.__name__ == self.fns['region_state_function'].__name__):
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
    super_e = Event(signal=SEARCH_FOR_SUPER_SIGNAL)
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

  def meta_trans(self, e, s, t):
    return meta_trans(
      hsm=self,
      e=e,
      s=s,
      t=t,
    )

  @staticmethod
  def cache_clear():
    Region.tockenize.cache_clear()
    Region.token_match.cache_clear()
    within.cache_clear()
    Region.get_region.cache_clear()



class InstrumentedActiveObject(ActiveObject):
  def __init__(self, name, log_config):
    '''Add the logging object and tweak the spy and trace features of the
       statechart to use this logging object.  This class is intended to be
       subclassed.

    **Note**:
       "logger_config.yaml" must exist in the same directory as this file.

       The log_config argument will contain a key contained in the
       "logger_config.yaml" file, in this same directory.  If you would like to
       change how the logger is behaving, like it's handler, its formatter, log
       level, edit this "logger_config.yaml" file instead of changing the code
       in this class.

    **Args**:
       | ``name`` (str): The name of this chart
       | ``log_config`` (str): The key of the logger_config.yaml which describes
       |                       the kind of handler used by this logger

    **Returns**:
      NA: Intended to be subclassed only

    **Example(s)**:

    .. code-block:: python

      # Since this Class will be subclassed by XmlChart, we will show the
      # XmlChart's construction in this example.  Note how the ``log_config``
      # argument is used.  You will find a ``xml_chart_5`` key under the
      # ``loggers`` keyword in the "logger_config.yaml" file which corresponds
      # to the way we want our chart's logger to behave.
      example = XmlChart(
        name='x',
        log_config="xml_chart_5",
        live_trace=False,
        live_spy=True,
      )

    '''
    super().__init__(name)

    self.log_config = log_config
    self.old_states = None

    with open('logger_config.yaml') as f:
      _dict = yaml.safe_load(f.read())
    logging_config.dictConfig(_dict)
    self.logger = logging.getLogger(self.log_config)
    self.logger.rotation_file_name = self.log_config
    self.clear_log()

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
    self.logger.info("T: " + trace)
    self.old_states = new_states

  def spy_callback(self, spy):
    '''spy with machine name pre-pending'''
    self.logger.info("S: [%s] %s" % (self.name, spy))

  def scribble(self, string):
    self.logger.info("S: {}".format(string))

  def _scribble(self, string):
    self.logger.debug("S: {}".format(string))

  def report(self, message):
    self.logger.warning("R: --- %s" % message)

  def _report(self, message):
    self.logger.warning("R: --- %s" % message)

  def clear_log(self):
    '''Clear this chart's log files'''
    for hdlr in self.logger.handlers[:]:
      if(isinstance(hdlr, logging.FileHandler) or
         isinstance(hdlr, logging.RotatingFileHandler)):
          with open(hdlr.baseFilename, "w") as f:
            f.write("")


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
    under_s = region_name + "_under_hidden_region"
    region_s = region_name + "_region"
    over_s = region_name + "_over_hidden_region"
    under_hidden_state_function = getattr(module_namespace, under_s)
    region_state_function = getattr(module_namespace, region_s)
    over_hidden_state_function = getattr(module_namespace, over_s)

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
  def __init__(self, name, log_config, starting_state, live_spy=None, live_trace=None):

    super().__init__(name, log_config)
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
    self.outmost = self
    self.inner = self
    self.same = self
    self.final_signal_name = None
    self.last_rtc_active_states = []
    self.starting_state = starting_state

  def next_rtc(self):
    '''Overload of the HsmWithQueues next_rtc, like the super's
    next_rtc and we update the last_rtc_active_states attribute of
    this class after we complete the run to completion event from
    the whole hhsm.'''

    # perform a run to completion process for this statechart
    super().next_rtc()
    #self.queue.wait()
    # store the active states of this hhsm
    self.last_rtc_active_states = [
      getattr(module_namespace, sfn) for sfn in flatten(self.active_states())
    ]


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
    if function_name == 'proto_investigate':
      previous_frame = inspect.currentframe().f_back.f_back
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
        region_summary = "{}:{}, ql={}:".format(region.name, region.state_name, _len)
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

    self.start_at(self.starting_state)
    self.instrumented = _instrumented

  @lru_cache(maxsize=32)
  def tockenize(self, signal_name):
    return set(signal_name.split("."))

  @lru_cache(maxsize=32)
  def token_match(self, resident, other):
    if other is None:
      result = False
    else:
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

    # parallel state names
    psn = self.regions.keys()

    result = []
    for n, regions in self.regions.items():
      for _region in regions._regions:
        result.append({_region.name: _region.state_name})

    def recursive_get_states(name):
      states = []
      if name in psn:
        for region in self.regions[name]._regions:
          if region.state_name in psn:
            _states = recursive_get_states(region.state_name)
            states.append(_states)
          else:
            states.append(region.state_name)
      else:
        states.append(self.state_name)
      return states

    states = recursive_get_states(self.state_name)
    return states

  def _active_states(self):
    result = []
    for n, regions in self.regions.items():
      for _region in regions._regions:
        result.append({_region.name: _region.state_name})
    return result

  def _post_lifo(self, e, outmost=None):
    self.post_lifo(e)

  def _post_fifo(self, e, outmost=None):
    self.post_fifo(e)

  def build_onion(self, t, sig, s=None, tried=None):
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
    if s == t:
      return []
    onion_states = []
    onion_states.append(t)

    def find_fns(state):
      '''For a given state find (region_state_function,
      outer_function_that_holds_the_region, region_object)

      **Args**:
         | ``state`` (state_function): the target of the WTF event given to
         |                             build_onion


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
          has_state = r.has_state(state)
          if has_state:
            outer_function_state_holds_the_region = getattr(module_namespace, rs.name)
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
        onion_states += [target_state, s]
        break
      if target_state:
        onion_states += [target_state, region_holding_state]

    if None in onion_states:
      if tried:
        return []
      if s is None:
        onion_states = [t]
      else:
        new_target = s
        new_source = t
        onion_states = self.build_onion(s=new_source, t=new_target, sig=sig, tried=True)
        if not onion_states:
          return onion_states
        else:
          onion_states.reverse()
    return onion_states


  def meta_trans(self, e, s, t):
    return meta_trans(
      hsm=self,
      e=e,
      s=s,
      t=t,
    )

  @lru_cache(maxsize=128)
  def _meta_trans(self, r, s, t, sig):
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
       | ``sig`` (str): signal name of event that initiated the need for this
       |                  event


    **Returns**:
       (Event): An event which can contain other events within it.

    **Example(s)**:

    .. code-block:: python

      def meta_trans(hsm, e, s, t):
        fn = hsm.outmost.meta_hooked(
          s=s,
          e=e
        )
        if fn is not None:
          status = fn(hsm, e)
        elif(in_same_hsm(source=s, target=t)):
          status = hsm.trans(t)
        else:
          _state, _e = hsm.outmost._meta_trans(
            hsm,
            t=t,
            s=s,
            sig=e.signal_name)
          if _state:
            status = hsm.trans(_state)
            hsm.same._post_fifo(_e)
            investigate(hsm, e, _e)
          else:
            status = return_status.HANDLED
            investigate(hsm, e, _e)
            hsm.same.post_fifo(_e)
        return status

    '''
    source = s
    target = t
    lca = self.lca(s, t)

    profile = ""
    profile += "self: {}\n".format(self)
    profile += "r: {}\n".format(r)
    profile += "s: {}\n".format(s)
    profile += "t: {}\n".format(t)
    profile += "sig: {}\n".format(sig)
    profile += "lca: {}\n".format(lca)

    event = None
    entry_onion = []
    exit_onion = []
    outer_injector = None

    if lca.__name__ == 'top':
      outer_injector = self.build_onion(t, sig=None)[-1]
    else:
      stripped_onion = []
      for fn in self.build_onion(t=s, s=lca, sig=sig)[1:]:
        stripped_onion.append(fn)
        if fn == lca:
          break
      stripped_onion.reverse()
      exit_onion = stripped_onion[:]

    if lca != t:
      entry_onion = self.build_onion(s=lca, t=t, sig=sig)[0:-1]

    # Wrap up the onion meta event from the inside out.  History items at the
    # last layer of the outer part of the INIT_META_SIGNAL need to reference an
    # even more outer part of the onion, the meta exit details.
    profile += "entry_onion: {}\n".format(entry_onion)
    profile += "exit_onion: {}\n".format(exit_onion)

    if entry_onion != [] and exit_onion != [] and entry_onion[-1] == exit_onion[0]:
      bounce_type = signals.BOUNCE_SAME_META_SIGNAL
      if len(exit_onion) == 1:
        _state = entry_onion[-2]
      else:
        _state = None
    elif(entry_onion == []):
      bounce_type = signals.OUTER_TRANS_REQUIRED
      _state = None
    elif(exit_onion == []):
      bounce_type = signals.BOUNCE_SAME_META_SIGNAL
      if s in entry_onion:
        entry_onion = entry_onion[0:entry_onion.index(s)]
      _state = None
    else:
      bounce_type = signals.BOUNCE_ACROSS_META_SIGNAL
      _state = None

    if _state and len(exit_onion) == 1:
      number = len(entry_onion) - 1
    else:
      number = len(entry_onion) + len(exit_onion)
      if exit_onion == [] and bounce_type == signals.BOUNCE_SAME_META_SIGNAL:
        number += 1

    if bounce_type == signals.OUTER_TRANS_REQUIRED:
      number += 1
      previous_state = exit_onion[0]
      previous_signal = signals.EXIT_META_SIGNAL
      event = Event(
        signal=bounce_type,
        payload=META_SIGNAL_PAYLOAD(
          n=number,
          event=event,
          state=t,
          previous_state=previous_state,
          previous_signal=previous_signal,
          springer=sig,
        )
      )
      number -= 1
    else:
      for index, entry_target in enumerate(entry_onion):

        signal_name = signals.INIT_META_SIGNAL

        if index == len(entry_onion) - 1:
          if (len(exit_onion) == 1 and
            exit_onion[0] == entry_onion[-1]):
              previous_state = s
              previous_signal = sig
          else:
            if len(exit_onion) > 0:
              previous_state = exit_onion[0]
              previous_signal = signals.EXIT_META_SIGNAL
            elif exit_onion == [] and bounce_type == signals.BOUNCE_SAME_META_SIGNAL:
              previous_state = entry_target
              previous_signal = bounce_type
            else:
              previous_state = s
              previous_signal = sig
          event = Event(
            signal=signal_name,
            payload=META_SIGNAL_PAYLOAD(
              n=number,
              event=event,
              state=entry_target,
              previous_state=previous_state,
              previous_signal=previous_signal,
              springer=sig,
            )
          )
        else:
          previous_state = entry_onion[index + 1]
          previous_signal = signals.INIT_META_SIGNAL
          event = Event(
            signal=signal_name,
            payload=META_SIGNAL_PAYLOAD(
              n=number,
              event=event,
              state=entry_target,
              previous_state=previous_state,
              previous_signal=previous_signal,
              springer=sig,
            )
          )
        number -= 1

    # Wrapping the EXIT_META_SIGNAL details around the META INIT part
    # on the onion meta event.  When we are at the outer layer
    # we need to write in the event that caused this meta event
    # and from what state it was created.
    if len(exit_onion) > 1:
      for index, exit_target in enumerate(exit_onion):
        signal_name = signals.EXIT_META_SIGNAL if exit_target != lca else bounce_type

        previous_signal = signals.EXIT_META_SIGNAL
        if index == len(exit_onion) - 1:
          previous_state = s
          previous_signal = sig
        else:
          previous_state = exit_onion[index + 1]

        event = Event(
          signal=signal_name,
          payload=META_SIGNAL_PAYLOAD(
            n=number,
            event=event,
            state=exit_target,
            previous_state=previous_state,
            previous_signal=previous_signal,
            springer=sig,
          )
        )
        number -= 1

    elif len(exit_onion) == 1:
      previous_state = s
      previous_signal = sig
      event = Event(
        signal=bounce_type,
        payload=META_SIGNAL_PAYLOAD(
          n=number,
          event=event,
          state=exit_onion[0],
          previous_state=previous_state,
          previous_signal=previous_signal,
          springer=sig,
        )
      )
      number -= 1
    else:
      bounce_onion = [s]
      for index, bounce_target in enumerate(bounce_onion):
        previous_state = bounce_onion[index]
        previous_state = bounce_onion[0]
        previous_signal = sig
        # here is the issue for SRE2
        if outer_injector is not None:
          signal_name = signals.INIT_META_SIGNAL
        else:
          signal_name = signals.BOUNCE_SAME_META_SIGNAL

        event = Event(
          signal=signal_name,
          payload=META_SIGNAL_PAYLOAD(
            n=number,
            event=event,
            state=bounce_target,
            previous_state=previous_state,
            previous_signal=previous_signal,
            springer=sig,
          )
        )
        _state = None
        number -= 1

    if _state and len(exit_onion) == 1:
      event = event.payload.event.payload.event
    if outer_injector is not None:
      _state = outer_injector

    profile += "ps: \n{}\n".format(ps(event))
    self.logger.critical(profile)

    return (_state, event)

  @lru_cache(maxsize=32)
  def lca(self, _s, _t):
    if (within(outer, _t)):
      return outer
    s_onion = self.build_onion(_s, sig=None)[::-1]
    t_onion = self.build_onion(_t, sig=None)[::-1]
    t_onion.reverse()
    _lca = self.bottom
    for t in t_onion:
      if t in s_onion:
        _lca = t
        break
    return _lca

  def p_spy(self, e):
    self.live_spy_callback(
      '{}:{}'.format(e.signal_name, self.state_name)
    )

  @lru_cache(maxsize=128)
  def _meta_hooked(self, s, t, sig):
    onion = self.build_onion(s=s, t=t, sig=sig)[:-1]
    old_temp, old_fun = self.temp.fun, self.state.fun
    e_seach_hook = Event(
      signal=signals.SEARCH_FOR_META_HOOKS,
      payload=META_HOOK_PAYLOAD(event=Event(signal=sig))
    )
    fn = None
    for layer in onion:
      if hasattr(layer, '__wrapped__'):
        _fn = layer.__wrapped__
      else:
        _fn = layer
      r = _fn(self, e_seach_hook)
      if r == return_status.HANDLED:
        fn = _fn
        break
    self.temp.fun, self.state.fun = old_temp, old_fun
    return fn

  def meta_hooked(self, s, e):
    fn = None
    for active_state in self.last_rtc_active_states:
      fn = self._meta_hooked(
        s=s,
        t=active_state,
        sig=e.signal_name
      )
      if fn is not None:
        break
    return fn

  @staticmethod
  def cache_clear():
    XmlChart.tockenize.cache_clear()
    XmlChart.token_match.cache_clear()
    XmlChart._meta_trans.cache_clear()
    XmlChart.lca.cache_clear()
    within.cache_clear()
    in_same_hsm.cache_clear()
    XmlChart._meta_hooked.cache_clear()

################################################################################
#                          STATE MACHINE                                       #
################################################################################
@othogonal_state
def p_r1_under_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = r.bottom

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(r.token_match(e.signal_name, "enter_region")):
    r._p_spy(e)
    status = r.trans(p_r1_region)
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_r1_region(r, e):
  status = return_status.UNHANDLED
  __super__ = p_r1_under_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)

    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.

    investigate(r, e, _e)
    # We can't compare the function directly because they can be arbitrarily
    # decorated by the user, so their addresses may not be the same, but their
    # names will be the same
    if _state and _state.__name__ == r.state_name:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is None or is referencing another region then follow are default
    # init behavior
    if _state is None or not within(r.state_fn, _state):
      status = r.trans(p_p11)
    else:
      # if _state is this state or a child of this state, transition to it
      status = r.trans(_state)
      # if there is more to our meta event, post it into the chart
      if _e is not None:
        r._post_fifo(_e)
  elif(e.signal == signals.INIT_META_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif e.signal == signals.BOUNCE_SAME_META_SIGNAL:
    r._p_spy(e)
    _state, _e = e.payload.state, e.payload.event
    investigate(r, e, _e)
    for region in r.same._regions:
      if r == region and r.has_state(e.payload.previous_state):
        region._post_fifo(_e)
        region._post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_META_SIGNAL):
    r._p_spy(e)
    (_e, _state) = e.payload.event, e.payload.state
    investigate(r, e, _e)
    if within(p_r1_region, _state):
      r.outer._post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
    r._p_spy(e)
    status = r.trans(p_r1_under_hidden_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_r1_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = p_r1_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.force_region_init):
    r._p_spy(e)
    status = r.trans(p_r1_region)
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11(r, e):
  status = return_status.UNHANDLED
  __super__ = p_r1_over_hidden_region
  __hooks__ = [signals.H1]

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER
  elif(e.signal == signals.SEARCH_FOR_META_HOOKS):
    if e.payload.event.signal in __hooks__:
      return return_status.HANDLED

  # enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    rsm(p_p11, e)
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)
    investigate(r, e, _e)
    if _state:
      r.inner._post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p11, e)
    r.inner.post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  # any event handled within there regions must be pushed from here
  elif(r.token_match(e.signal_name, "e1") or
       r.token_match(e.signal_name, "e2") or
       r.token_match(e.signal_name, "e4") or
       r.token_match(e.signal_name, "SRH3") or
       r.token_match(e.signal_name, "PG2")
       ):
    r.p_spy(e)
    r.inner.post_fifo(e)
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "H1")):
    r.scribble("p_p11 hooked")
    status = return_status.HANDLED
  elif r.token_match(e.signal_name, r.outmost.regions['p_p11'].final_signal_name):
    r.p_spy(e)
    status = r.trans(p_p12)
  elif r.token_match(e.signal_name, "RC1"):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p11,
      t=p_p12,
    )
  elif r.token_match(e.signal_name, "SRH2"):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p11,
      t=middle,
    )
  elif r.token_match(e.signal_name, "RA1"):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p11,
      t=p_p11,
    )
  elif r.token_match(e.signal_name, "PF1"):
    r.p_spy(e)
    status = return_status.HANDLED
  elif r.token_match(e.signal_name, "PC1"):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p11,
      t=p_s21,
    )
  elif(r.token_match(e.signal_name, "RF1")):
    r.p_spy(e)

    status = r.meta_trans(
      e=e,
      s=p_p11,
      t=p_p12_p11_s12,
    )
  elif e.signal == signals.BOUNCE_SAME_META_SIGNAL:
    r._p_spy(e)
    _state, _e = e.payload.state, e.payload.event
    r.inner._post_fifo(_e)
    investigate(r, e, _e)
    r.inner.post_lifo(Event(signal=signals.force_region_init))
    status = return_status.HANDLED
  elif e.signal == signals.OUTER_TRANS_REQUIRED:
    status = return_status.HANDLED
    r.p_spy(e)
    (_e, _state) = e.payload.event, e.payload.state
    investigate(r, e, _e)
    if _state.__name__ == r.state_fn.__name__:
      r.inner.post_fifo(Event(signal=signals.exit_region))
      r.inner.post_fifo(Event(signal=signals.enter_region))
    else:
      if within(r.state_fn, _state):
        status = r.trans(_state)
  elif e.signal == signals.EXIT_META_SIGNAL:
    r.p_spy(e)
    (_e, _state) = e.payload.event, e.payload.state
    investigate(r, e, _e)
    # this appears backwards, but it needs to be this way.
    if within(_state, r.state_fn):
      # The next state is going to be our region handler skip it and post this
      # region handler would have posted to the outer HSM
      if(_e.payload.event.signal == signals.EXIT_META_SIGNAL or
         _e.payload.event.signal == signals.BOUNCE_ACROSS_META_SIGNAL or
         _e.payload.event.signal == signals.OUTER_TRANS_REQUIRED
         ):
        (_e, _state) = _e.payload.event, _e.payload.state
        r.outer._post_lifo(_e)
      elif(_e.signal == signals.BOUNCE_ACROSS_META_SIGNAL or
           _e.signal == signals.EXIT_META_SIGNAL):
        r.outer._post_lifo(_e)
      else:
        r.same._post_lifo(_e)
    status = return_status.HANDLED
  elif e.signal == signals.exit_region:
    r._p_spy(e)
    status = r.trans(p_r1_under_hidden_region)
  elif e.signal == signals.EXIT_SIGNAL:
    r.inner.post_lifo(Event(signal=signals.exit_region))
    rsm(p_p11, e)
    r.p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_r1_under_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = r.bottom

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(r.token_match(e.signal_name, "enter_region")):
    r._p_spy(e)
    status = r.trans(p_p11_r1_region)
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_r1_region(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p11_r1_under_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)

    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.

    investigate(r, e, _e)
    # We can't compare the function directly because they can be arbitrarily
    # decorated by the user, so their addresses may not be the same, but their
    # names will be the same
    if _state and _state.__name__ == r.state_name:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is None or is referencing another region then follow are default
    # init behavior
    if _state is None or not within(r.state_fn, _state):
      status = r.trans(p_p11_s11)
    else:
      # if _state is this state or a child of this state, transition to it
      status = r.trans(_state)
      # if there is more to our meta event, post it into the chart
      if _e is not None:
        r._post_fifo(_e)
  elif(e.signal == signals.INIT_META_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif e.signal == signals.BOUNCE_SAME_META_SIGNAL:
    r._p_spy(e)
    _state, _e = e.payload.state, e.payload.event
    investigate(r, e, _e)
    for region in r.same._regions:
      if region.has_state(e.payload.previous_state):
        region._post_fifo(_e)
        region._post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  elif e.signal == signals.EXIT_META_SIGNAL:
    r._p_spy(e)
    (_e, _state) = e.payload.event, e.payload.state
    investigate(r, e, _e)
    if within(p_p11_r1_region, _state):
      (_e, _state) = _e.payload.event, _e.payload.state
      r.outer._post_lifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
    r._p_spy(e)
    status = r.trans(p_p11_r1_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_r1_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p11_r1_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.force_region_init):
    r._p_spy(e)
    status = r.trans(p_p11_r1_region)
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_s11(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p11_r1_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    rsm(p_p11_s11, e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p11_s11, e)
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e4")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p11_s11,
      t=p_p11_s12,
    )
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p11_s11, e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_s12(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p11_r1_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    rsm(p_p11_s12, e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    rsm(p_p11_s12, e)
    r.p_spy(e)
    status = return_status.HANDLED
  elif r.token_match(e.signal_name, "SRH3"):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p11_s12,
      t=p,
    )
  elif(r.token_match(e.signal_name, "e1")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p11_s12,
      t=p_p11_r1_final,
    )
  elif(e.signal == signals.EXIT_SIGNAL):
    rsm(p_p11_s12, e)
    r.p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_r1_final(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p11_r1_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    r.final = True
    r.post_p_final_to_outmost_if_ready()
    rsm(p_p11_r1_final, e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p11_r1_final, e)
    r.final = False
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_r2_under_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = r.bottom

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(r.token_match(e.signal_name, "enter_region")):
    r._p_spy(e)
    status = r.trans(p_p11_r2_region)
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_r2_region(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p11_r2_under_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)

    investigate(r, e, _e)

    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.
    if _state == p_p11_r2_region:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is a child of this state then transition to it
    if _state is None or not within(p_p11_r2_region, _state):
      status = r.trans(p_p11_s21)
    else:
      status = r.trans(_state)
      if _e is not None:
        r.post_fifo(_e)
  elif(e.signal == signals.INIT_META_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif e.signal == signals.BOUNCE_SAME_META_SIGNAL:
    r._p_spy(e)
    investigate(r, e, _e)
    _state, _e = e.payload.state, e.payload.event
    for region in r.same._regions:
      if region.has_state(e.payload.previous_state):
        region._post_fifo(_e)
        region._post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_META_SIGNAL):
    r._p_spy(e)
    (_e, _state) = e.payload.event, e.payload.state
    investigate(r, e, _e)
    if within(p_p11_r2_region, _state):
      r.outer._post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
    r._p_spy(e)
    status = r.trans(p_p11_r2_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_r2_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p11_r2_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.force_region_init):
    r._p_spy(e)
    status = r.trans(p_p11_r2_region)
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_s21(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p11_r2_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    rsm(p_p11_s21, e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p11_s21, e)
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e1")):
    status = r.meta_trans(
      e=e,
      s=p_p11_s21,
      t=p_p11_s22,
    )
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p11_s21, e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_s22(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p11_r2_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    rsm(p_p11_s22, e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p11_s22, e)
    status = return_status.HANDLED
  elif(e.signal == signals.PG2):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p11_s22,
      t=p_s21,
    )
  elif(r.token_match(e.signal_name, "e2")):
    r.p_spy(e)
    status = r.trans(p_p11_r2_final)
  elif(r.token_match(e.signal_name, "e1")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p11_s22,
      t=p_p11_s21,
    )
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p11_s22, e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p11_r2_final(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p11_r2_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    r.final = True
    r.post_p_final_to_outmost_if_ready()
    rsm(p_p11_r2_final, e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    r.final = False
    rsm(p_p11_r2_final, e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_r1_final(r, e):
  status = return_status.UNHANDLED
  __super__ = p_r1_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    r.final = True
    rsm(p_r1_final, e)
    r.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_r1_final, e)
    r.final = False
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12(r, e):
  status = return_status.UNHANDLED
  __super__ = p_r1_over_hidden_region
  __hooks__ = [signals.H1]

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER
  elif(e.signal == signals.SEARCH_FOR_META_HOOKS):
    if e.payload.event.signal in __hooks__:
      return return_status.HANDLED

  # Enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    rsm(p_p12, e)
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)
    investigate(r, e, _e)
    if _state:
      r.inner._post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p12, e)
    r.inner.post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  # Any event handled within there regions must be pushed from here
  elif(r.token_match(e.signal_name, "e1") or
       r.token_match(e.signal_name, "e2") or
       r.token_match(e.signal_name, "e4") or
       r.token_match(e.signal_name, "RD1") or
       r.token_match(e.signal_name, "PG1") or
       r.token_match(e.signal_name, "RH1") or
       r.token_match(e.signal_name, "RG1") or
       r.token_match(e.signal_name,
         r.outmost.regions['p_p12_p11'].final_signal_name)
       ):
    r.p_spy(e)
    r.inner.post_fifo(e)
    status = return_status.HANDLED
  # All internal injectors will have to have this structure
  elif e.signal == signals.EXIT_META_SIGNAL:
    r.p_spy(e)
    (_e, _state) = e.payload.event, e.payload.state
    investigate(r, e, _e)

    # this appears backwards, but it needs to be this way.
    if within(_state, r.state_fn):
      # The next state is going to be our region handler skip it and post this
      # region handler would have posted to the outer HSM
      if(_e.payload.event.signal == signals.EXIT_META_SIGNAL or
         _e.payload.event.signal == signals.BOUNCE_ACROSS_META_SIGNAL or
         _e.payload.event.signal == signals.OUTER_TRANS_REQUIRED
         ):
        (_e, _state) = _e.payload.event, _e.payload.state
        r.outer._post_lifo(_e)
      elif(_e.signal == signals.BOUNCE_ACROSS_META_SIGNAL or
           _e.signal == signals.EXIT_META_SIGNAL):
        r.outer._post_lifo(_e)
      else:
        r.same._post_lifo(_e)
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "H1")):
    r.scribble("p_p12 hooked")
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "H2")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p12,
      t=p
    )
  elif(r.token_match(e.signal_name, "RE1")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p12,
      t=p_p12_p11_s12,
    )
  elif(r.token_match(e.signal_name, "RA2")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p12,
      t=p_p12,
    )
  elif e.signal == signals.BOUNCE_SAME_META_SIGNAL:
    r._p_spy(e)
    _state, _e = e.payload.state, e.payload.event
    r.inner._post_fifo(_e)
    investigate(r, e, _e)
    r.inner.post_lifo(Event(signal=signals.force_region_init))
    status = return_status.HANDLED

  elif(r.token_match(e.signal_name, "RB1")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p12,
      t=p_p12_p11,
    )
  elif e.signal == signals.OUTER_TRANS_REQUIRED:
    r.p_spy(e)
    status = return_status.HANDLED
    (_e, _state) = e.payload.event, e.payload.state
    investigate(r, e, _e)
    if _state.__name__ == r.state_fn.__name__:
      r.inner.post_fifo(Event(signal=signals.exit_region))
      r.inner.post_fifo(Event(signal=signals.enter_region))
    else:
      if within(r.state_fn, _state):
        status = r.trans(_state)
  # Final token match
  elif(r.token_match(e.signal_name, r.outmost.regions['p_p12'].final_signal_name)):
    r.p_spy(e)
    status = r.trans(p_r1_final)
  elif(r.token_match(e.signal_name, "e5")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p12,
      t=p_r1_final,
    )
  # Exit signals
  elif(e.signal == signals.exit_region):
    r._p_spy(e)
    status = r.trans(p_r1_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    r.inner.post_lifo(Event(signal=signals.exit_region))
    rsm(p_p12, e)
    r.p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

# inner parallel
@othogonal_state
def p_p12_r1_under_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = r.bottom

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(r.token_match(e.signal_name, "enter_region")):
    r._p_spy(e)
    status = r.trans(p_p12_r1_region)
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

# inner parallel
@othogonal_state
def p_p12_r1_region(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p12_r1_under_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)

    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.

    investigate(r, e, _e)
    # We can't compare the function directly because they can be arbitrarily
    # decorated by the user, so their addresses may not be the same, but their
    # names will be the same
    if _state and _state.__name__ == r.state_name:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is None or is referencing another region then follow are default
    # init behavior
    if _state is None or not within(r.state_fn, _state):
      status = r.trans(p_p12_p11)
    else:
      # if _state is this state or a child of this state, transition to it
      status = r.trans(_state)
      # if there is more to our meta event, post it into the chart
      if _e is not None:
        r.post_fifo(_e)
  elif e.signal == signals.BOUNCE_SAME_META_SIGNAL:
    r._p_spy(e)
    _state, _e = e.payload.state, e.payload.event
    investigate(r, e, _e)
    for region in r.same._regions:
      if region.has_state(e.payload.previous_state):
        region._post_fifo(_e)
        region._post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_META_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_META_SIGNAL):
    r._p_spy(e)
    (_e, _state) = e.payload.event, e.payload.state
    investigate(r, e, _e)
    if r.within(p_p12_r1_region, _state):
      r.outer._post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
    r._p_spy(e)
    status = r.trans(p_p12_r1_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_r1_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p12_r1_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.force_region_init):
    r._p_spy(e)
    status = r.trans(p_p12_r1_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status


# inner parallel
@othogonal_state
def p_p12_p11(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p12_r1_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  # enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_p11, e)

    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)
    investigate(r, e, _e)
    if _state:
      r.inner._post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_p11, e)
    r.inner.post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "RG1") or
       r.token_match(e.signal_name, "e1") or
       r.token_match(e.signal_name, "PG1") or
       r.token_match(e.signal_name, "RH1")
       ):
    r.p_spy(e)
    r.inner.post_fifo(e)
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, r.outmost.regions['p_p12_p11'].final_signal_name)):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p12_p11,
      t=p_p12_s12,
    )
  elif(r.token_match(e.signal_name, "RD1")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p12_p11,
      t=p_p12,
    )
  elif e.signal == signals.EXIT_META_SIGNAL:
    r.p_spy(e)
    (_e, _state) = e.payload.event, e.payload.state
    investigate(r, e, _e)
    # this appears backwards, but it needs to be this way.
    if within(_state, r.state_fn):
      # The next state is going to be our region handler skip it and post this
      # region handler would have posted to the outer HSM
      if(_e.payload.event.signal == signals.EXIT_META_SIGNAL or
         _e.payload.event.signal == signals.BOUNCE_ACROSS_META_SIGNAL or
         _e.payload.event.signal == signals.OUTER_TRANS_REQUIRED
         ):
        (_e, _state) = _e.payload.event, _e.payload.state
        r.outer._post_lifo(_e)
      elif(_e.signal == signals.BOUNCE_ACROSS_META_SIGNAL or
           _e.signal == signals.EXIT_META_SIGNAL):
        r.outer._post_lifo(_e)
      else:
        r.same._post_lifo(_e)
    status = return_status.HANDLED
  elif e.signal == signals.BOUNCE_SAME_META_SIGNAL:
    r._p_spy(e)
    _state, _e = e.payload.state, e.payload.event
    r.inner._post_fifo(_e)
    investigate(r, e, _e)
    r.inner.post_lifo(Event(signal=signals.force_region_init))
    status = return_status.HANDLED
  elif e.signal == signals.OUTER_TRANS_REQUIRED:
    r.p_spy(e)
    status = return_status.HANDLED
    (_e, _state) = e.payload.event, e.payload.state
    investigate(r, e, _e)
    if _state.__name__ == r.state_fn.__name__:
      r.inner.post_fifo(Event(signal=signals.exit_region))
      r.inner.post_fifo(Event(signal=signals.enter_region))
    else:
      if within(r.state_fn, _state):
        status = r.trans(_state)
  elif(r.token_match(e.signal_name, "e4")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p12_p11,
      t=p_p12_s12,
    )
  elif(e.signal == signals.exit_region):
    r._p_spy(e)
    status = r.trans(p_p12_r1_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    r.inner.post_lifo(Event(signal=signals.exit_region))
    rsm(p_p12_p11, e)
    r.p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_p11_r1_under_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = r.bottom

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(r.token_match(e.signal_name, "enter_region")):
    status = r.trans(p_p12_p11_r1_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_p11_r1_region(r, e):
  status = return_status.UNHANDLED

  __super__ = p_p12_p11_r1_under_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)

    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.

    investigate(r, e, _e)
    # We can't compare the function directly because they can be arbitrarily
    # decorated by the user, so their addresses may not be the same, but their
    # names will be the same
    if _state and _state.__name__ == r.state_name:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is None or is referencing another region then follow are default
    # init behavior
    if _state is None or not within(r.state_fn, _state):
      status = r.trans(p_p12_p11_s11)
    else:
      # if _state is this state or a child of this state, transition to it
      status = r.trans(_state)
      # if there is more to our meta event, post it into the chart
      if _e is not None:
        r.post_fifo(_e)
  elif(e.signal == signals.INIT_META_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif e.signal == signals.BOUNCE_SAME_META_SIGNAL:
    r._p_spy(e)
    _state, _e = e.payload.state, e.payload.event
    investigate(r, e, _e)
    for region in r.same._regions:
      if region.has_state(e.payload.previous_state):
        region._post_fifo(_e)
        region._post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_META_SIGNAL):
    r._p_spy(e)
    (_e, _state) = e.payload.event, e.payload.state
    investigate(r, e, _e)
    if within(p_p12_p11_r1_region, _state):
      r.outer._post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
    r._p_spy(e)
    status = r.trans(p_p12_p11_r1_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_p11_r1_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p12_p11_r1_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.force_region_init):
    r._p_spy(e)
    status = r.trans(p_p12_p11_r1_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_p11_s11(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p12_p11_r1_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_p11_s11, e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_p11_s11, e)
    status = return_status.HANDLED
  elif(e.signal == signals.e1):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p12_p11_s11,
      t=p_p12_p11_s12,
    )
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_p11_s11, e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_p11_s12(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p12_p11_r1_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_p11_s12, e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_p11_s12, e)
    status = return_status.HANDLED
  elif r.token_match(e.signal_name, "RH1"):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p12_p11_s12,
      t=p_p12,
    )
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_p11_s12, e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_p11_r2_under_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = r.bottom

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(r.token_match(e.signal_name, "enter_region")):
    status = r.trans(p_p12_p11_r2_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_META_SIGNAL):
    (_e, _state) = e.payload.event, e.payload.state
    if within(p_p12_p11_r2_region, _state):
      r.outer._post_fifo(_e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_p11_r2_region(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p12_p11_r2_under_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)
    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.

    investigate(r, e, _e)
    # We can't compare the function directly because they can be arbitrarily
    # decorated by the user, so their addresses may not be the same, but their
    # names will be the same
    if _state and _state.__name__ == r.state_name:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is None or is referencing another region then follow are default
    # init behavior
    if _state is None or not within(r.state_fn, _state):
      status = r.trans(p_p12_p11_s21)
    else:
      # if _state is this state or a child of this state, transition to it
      status = r.trans(_state)
      # if there is more to our meta event, post it into the chart
      if _e is not None:
        r.post_fifo(_e)
  elif(e.signal == signals.INIT_META_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif e.signal == signals.BOUNCE_SAME_META_SIGNAL:
    r._p_spy(e)
    _state, _e = e.payload.state, e.payload.event
    investigate(r, e, _e)
    for region in r.same._regions:
      if region.has_state(e.payload.previous_state):
        region._post_fifo(_e)
        region._post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_META_SIGNAL):
    r._p_spy(e)
    (_e, _state) = e.payload.event, e.payload.state
    investigate(r, e, _e)
    if within(p_p12_p11_r2_region, _state):
      r.outer._post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
    r._p_spy(e)
    status = r.trans(p_p12_p11_r2_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_p11_r2_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p12_p11_r2_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.force_region_init):
    r._p_spy(e)
    status = r.trans(p_p12_p11_r2_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_p11_s21(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p12_p11_r2_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_p11_s21, e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_p11_s21, e)
    status = return_status.HANDLED
  elif(e.signal == signals.RG1):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p12_p11_s21,
      t=p_p22_s11,
    )
  elif(e.signal == signals.PG1):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p12_p11_s21,
      t=p_p22_s11,
    )
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_p11_s21, e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

# inner parallel
@othogonal_state
def p_p12_s12(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p12_r1_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_s12, e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_s12, e)
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e1")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p12_s12,
      t=p_p12_r1_final,
    )
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_s12, e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

# inner parallel
@othogonal_state
def p_p12_r1_final(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p12_r1_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    r.final = True
    rsm(p_p12_r1_final, e)
    r.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_r1_final, e)
    r.post_p_final_to_outmost_if_ready()
    r.final = False
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_r2_under_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = r.bottom

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(r.token_match(e.signal_name, "enter_region")):
    r._p_spy(e)
    status = r.trans(p_p12_r2_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

# inner parallel
@othogonal_state
def p_p12_r2_region(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p12_r2_under_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)

    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.

    investigate(r, e, _e)
    # We can't compare the function directly because they can be arbitrarily
    # decorated by the user, so their addresses may not be the same, but their
    # names will be the same
    if _state and _state.__name__ == r.state_name:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is None or is referencing another region then follow are default
    # init behavior
    if _state is None or not within(r.state_fn, _state):
      status = r.trans(p_p12_s21)
    else:
      # if _state is this state or a child of this state, transition to it
      status = r.trans(_state)
      # if there is more to our meta event, post it into the chart
      if _e is not None:
        r.post_fifo(_e)
  elif(e.signal == signals.INIT_META_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif e.signal == signals.BOUNCE_SAME_META_SIGNAL:
    r._p_spy(e)
    _state, _e = e.payload.state, e.payload.event
    investigate(r, e, _e)
    for region in r.same._regions:
      if region.has_state(e.payload.previous_state):
        region._post_fifo(_e)
        region._post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_META_SIGNAL):
    r._p_spy(e)
    (_e, _state) = e.payload.event, e.payload.state
    investigate(r, e, _e)
    if within(p_p12_r2_region, _state):
      r.outer._post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
    r._p_spy(e)
    status = r.trans(p_p12_r2_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p12_r2_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p12_r2_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.force_region_init):
    r._p_spy(e)
    status = r.trans(p_p12_r2_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

# inner parallel
@othogonal_state
def p_p12_s21(r, e):
  __super__ = p_p12_r2_over_hidden_region
  __hooks__ = [signals.H1, signals.H2]
  status = return_status.UNHANDLED

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER
  elif(e.signal == signals.SEARCH_FOR_META_HOOKS):
    if e.payload.event.signal in __hooks__:
      return return_status.HANDLED

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_s21, e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_s21, e)
    status = return_status.HANDLED
  elif(
    r.token_match(e.signal_name, "H1") or
    r.token_match(e.signal_name, "H2")
  ):
    r.scribble("p_p11 hooked")
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e1")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p12_s21,
      t=p_p12_s22,
    )
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_s21, e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

# inner parallel
@othogonal_state
def p_p12_s22(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p12_r2_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_s22, e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_s22, e)
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e2")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p12_s22,
      t=p_p12_r2_final,
    )
  elif(r.token_match(e.signal_name, "e1")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p12_s22,
      t=p_p12_s21,
    )
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_s22, e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

# inner parallel
@othogonal_state
def p_p12_r2_final(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p12_r2_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    r.final = True
    rsm(p_p12_r2_final, e)
    r.post_p_final_to_outmost_if_ready()
    r.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p12_r2_final, e)
    r.final = False
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_r2_under_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = r.bottom

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(r.token_match(e.signal_name, "enter_region")):
    r._p_spy(e)
    status = r.trans(p_r2_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_r2_region(r, e):
  status = return_status.UNHANDLED
  __super__ = p_r2_under_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)

    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.

    investigate(r, e, _e)
    # We can't compare the function directly because they can be arbitrarily
    # decorated by the user, so their addresses may not be the same, but their
    # names will be the same
    if _state and _state.__name__ == r.state_name:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is None or is referencing another region then follow are default
    # init behavior
    if _state is None or not within(r.state_fn, _state):
      status = r.trans(p_s21)
    else:
      # if _state is this state or a child of this state, transition to it
      status = r.trans(_state)
      # if there is more to our meta event, post it into the chart
      if _e is not None:
        r.post_fifo(_e)
  elif(e.signal == signals.INIT_META_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif e.signal == signals.BOUNCE_SAME_META_SIGNAL:
    r._p_spy(e)
    _state, _e = e.payload.state, e.payload.event
    investigate(r, e, _e)
    for region in r.same._regions:
      if r == region and r.has_state(e.payload.previous_state):
        region._post_fifo(_e)
        region._post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_META_SIGNAL):
    r._p_spy(e)
    (_e, _state) = e.payload.event, e.payload.state
    investigate(r, e, _e)
    if within(p_r2_region, _state):
      r.outer._post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
    r._p_spy(e)
    status = r.trans(p_r2_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_r2_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = p_r2_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.force_region_init):
    r._p_spy(e)
    status = r.trans(p_r2_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_s21(r, e):
  status = return_status.UNHANDLED
  __super__ = p_r2_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    rsm(p_s21, e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r.p_spy(e)
    rsm(p_s21, e)
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "RC1")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_s21,
      t=p_p22,
    )
  elif(r.token_match(e.signal_name, "RF1")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_s21,
      t=p_p22_s12,
    )
  elif(r.token_match(e.signal_name, "PF1")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_s21,
      t=p_p12_p11_s21,
    )

  elif r.token_match(e.signal_name, "SRH1"):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_s21,
      t=middle,
    )
  elif r.token_match(e.signal_name, "SRD2"):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_s21,
      t=p,
    )
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_s21, e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22(r, e):
  status = return_status.UNHANDLED
  __super__ = p_r2_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  # enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    rsm(p_p22, e)
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)
    investigate(r, e, _e)
    if _state:
      r.inner._post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p22, e)
    r.inner.post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  # any event handled within there regions must be pushed from here
  elif(r.token_match(e.signal_name, "e1") or
       r.token_match(e.signal_name, "e2") or
       r.token_match(e.signal_name, "e4") or
       r.token_match(e.signal_name, "SRG1") or
       r.token_match(e.signal_name, "RF1") or
       r.token_match(e.signal_name, "RE1")
       ):
    r.p_spy(e)
    r.inner.post_fifo(e)
    status = return_status.HANDLED
  # final token match
  elif r.token_match(e.signal_name, "SRD1"):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p22,
      t=p,
    )
  elif(r.token_match(e.signal_name, "RC2")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p22,
      t=p_s21,
    )
  elif(r.token_match(e.signal_name, r.outmost.regions['p_p22'].final_signal_name)):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p22,
      t=p_r2_final,
    )
  elif e.signal == signals.BOUNCE_SAME_META_SIGNAL:
    r._p_spy(e)
    _state, _e = e.payload.state, e.payload.event
    r.inner._post_fifo(_e)
    investigate(r, e, _e)
    r.inner.post_lifo(Event(signal=signals.force_region_init))
    status = return_status.HANDLED
  elif e.signal == signals.EXIT_META_SIGNAL:
    r.p_spy(e)
    (_e, _state) = e.payload.event, e.payload.state
    investigate(r, e, _e)
    # this appears backwards, but it needs to be this way.
    if within(_state, r.state_fn):
      # The next state is going to be our region handler skip it and post this
      # region handler would have posted to the outer HSM
      if(_e.payload.event.signal == signals.EXIT_META_SIGNAL or
         _e.payload.event.signal == signals.BOUNCE_ACROSS_META_SIGNAL or
         _e.payload.event.signal == signals.OUTER_TRANS_REQUIRED
         ):
        (_e, _state) = _e.payload.event, _e.payload.state
        r.outer._post_lifo(_e)
      elif(_e.signal == signals.BOUNCE_ACROSS_META_SIGNAL or
           _e.signal == signals.EXIT_META_SIGNAL):
        r.outer._post_lifo(_e)
      else:
        r.same._post_lifo(_e)
    status = return_status.HANDLED
  elif e.signal == signals.OUTER_TRANS_REQUIRED:
    r.p_spy(e)
    status = return_status.HANDLED
    (_e, _state) = e.payload.event, e.payload.state
    investigate(r, e, _e)
    if _state.__name__ == r.state_fn.__name__:
      r.inner.post_fifo(Event(signal=signals.exit_region))
      r.inner.post_fifo(Event(signal=signals.enter_region))
    else:
      if within(r.state_fn, _state):
        status = r.trans(_state)
  elif(e.signal == signals.exit_region):
    r._p_spy(e)
    status = r.trans(p_r2_under_hidden_region)
  elif(e.signal == signals.C1):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p22,
      t=p_s21,
    )
  elif(e.signal == signals.EXIT_SIGNAL):
    r.inner.post_lifo(Event(signal=signals.exit_region))
    rsm(p_p22, e)
    r.p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

# inner parallel
@othogonal_state
def p_p22_r1_under_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = r.bottom

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(r.token_match(e.signal_name, "enter_region")):
    r._p_spy(e)
    status = r.trans(p_p22_r1_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22_r1_region(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p22_r1_under_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)

    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.
    investigate(r, e, _e)
    # We can't compare the function directly because they can be arbitrarily
    # decorated by the user, so their addresses may not be the same, but their
    # names will be the same
    if _state and _state.__name__ == r.state_name:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is None or is referencing another region then follow are default
    # init behavior
    if _state is None or not within(r.state_fn, _state):
      status = r.trans(p_p22_s11)
    else:
      # if _state is this state or a child of this state, transition to it
      status = r.trans(_state)
      # if there is more to our meta event, post it into the chart
      if _e is not None:
        r.post_fifo(_e)
  elif e.signal == signals.BOUNCE_SAME_META_SIGNAL:
    r._p_spy(e)
    _state, _e = e.payload.state, e.payload.event
    investigate(r, e, _e)
    for region in r.same._regions:
      if region.has_state(e.payload.previous_state):
        region._post_fifo(_e)
        region._post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  # can we get rid of exit_region?
  elif(e.signal == signals.exit_region):
    r._p_spy(e)
    status = r.trans(p_p22_r1_under_hidden_region)
  elif(e.signal == signals.INIT_META_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22_r1_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p22_r1_region
  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.force_region_init):
    r._p_spy(e)
    status = r.trans(p_p22_r1_under_hidden_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22_s11(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p22_r1_over_hidden_region
  __hooks__ = [signals.H2]

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER
  elif(e.signal == signals.SEARCH_FOR_META_HOOKS):
    if e.payload.event.signal in __hooks__:
      return return_status.HANDLED

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    rsm(p_p22_s11, e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    rsm(p_p22_s11, e)
    r.p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.H2):
    r.scribble('p_p22_s11 hook')
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e4")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p22_s11,
      t=p_p22_s12,
    )
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p22_s11, e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22_s12(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p22_r1_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    rsm(p_p22_s12, e)
    status = return_status.HANDLED
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p22_s12, e)
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e1")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p22_s12,
      t=p_p22_r1_final,
    )
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p22_s12, e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22_r1_final(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p22_r1_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    status = return_status.HANDLED
    r.final = True
    rsm(p_p22_r1_final, e)
    r.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p22_r1_final, e)
    r.final = False
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

# inner parallel
@othogonal_state
def p_p22_r2_under_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = r.bottom

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(r.token_match(e.signal_name, "enter_region")):
    r._p_spy(e)
    status = r.trans(p_p22_r2_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22_r2_region(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p22_r2_under_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    # search for INIT_META_SIGNAL
    (_e, _state) = r.meta_peel(e)

    # If the target state is this state, just strip this layer of
    # the meta event and use the next one (this was done to make the
    # meta events consistent and easy to read and usable by different
    # types of WTF events.

    investigate(r, e, _e)
    # We can't compare the function directly because they can be arbitrarily
    # decorated by the user, so their addresses may not be the same, but their
    # names will be the same
    if _state and _state.__name__ == r.state_name:
      (_e, _state) = _e.payload.event, _e.payload.state

    # if _state is None or is referencing another region then follow are default
    # init behavior
    if _state is None or not within(r.state_fn, _state):
      status = r.trans(p_p22_s21)
    else:
      # if _state is this state or a child of this state, transition to it
      status = r.trans(_state)
      # if there is more to our meta event, post it into the chart
      if _e is not None:
        r.post_fifo(_e)
  elif(e.signal == signals.INIT_META_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif e.signal == signals.BOUNCE_SAME_META_SIGNAL:
    r._p_spy(e)
    _state, _e = e.payload.state, e.payload.event
    investigate(r, e, _e)
    for region in r.same._regions:
      if region.has_state(e.payload.previous_state):
        region._post_fifo(_e)
        region._post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_META_SIGNAL):
    r._p_spy(e)
    (_e, _state) = e.payload.event, e.payload.state
    investigate(r, e, _e)
    if within(p_p22_r2_region, _state):
      r.outer._post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
    r._p_spy(e)
    status = r.trans(p_p22_r2_under_hidden_region)
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22_r2_over_hidden_region(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p22_r2_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.force_region_init):
    r._p_spy(e)
    status = r.trans(p_p22_r2_under_hidden_region)
  elif(e.signal == signals.ENTRY_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r._p_spy(e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22_s21(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p22_r2_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER


  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    rsm(p_p22_s21, e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p22_s21, e)
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e1")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p22_s21,
      t=p_p22_s22,
    )
  elif r.token_match(e.signal_name, "SRG1"):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p22_s21,
      t=s_s1,
    )
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p22_s21, e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22_s22(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p22_r2_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    rsm(p_p22_s22, e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p22_s22, e)
    status = return_status.HANDLED
  elif(r.token_match(e.signal_name, "e2")):
    r.p_spy(e)
    status = r.meta_trans(
      e=e,
      s=p_p22_s22,
      t=p_p22_r2_final,
    )
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p22_s22, e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_p22_r2_final(r, e):
  status = return_status.UNHANDLED
  __super__ = p_p22_r2_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    r.final = True
    rsm(p_p22_r2_final, e)
    r.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    rsm(p_p22_r2_final, e)
    r.final = False
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@othogonal_state
def p_r2_final(r, e):
  status = return_status.UNHANDLED
  __super__ = p_r2_over_hidden_region

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    r.temp.fun = __super__
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    r.p_spy(e)
    r.final = True
    rsm(p_r2_final, e)
    r.post_p_final_to_outmost_if_ready()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    r.p_spy(e)
    r.final = False
    rsm(p_r2_final, e)
    status = return_status.HANDLED
  else:
    r.temp.fun = __super__
    status = return_status.SUPER
  return status

@state
def outer(self, e):
  status = return_status.UNHANDLED

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    self.temp.fun = self.bottom
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    self.p_spy(e)
    rsm(outer, e)
    #
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    self.p_spy(e)
    rsm(outer, e)
    status = return_status.HANDLED
  elif(self.token_match(e.signal_name, "SH1")):
    self.p_spy(e)
    status = self.meta_trans(
      e=e,
      s=outer,
      t=p,
    )
  elif(self.token_match(e.signal_name, "SH2")):
    self.p_spy(e)
    status = self.meta_trans(
      e=e,
      s=outer,
      t=middle,
    )
  elif(self.token_match(e.signal_name, "SRE3")):
    self.p_spy(e)
    _state, _e = self.outmost._meta_trans(
      self,
      t=p_p22,
      s=outer,
      sig=e.signal_name
    )
    investigate(self, e, _e)
    self.same._post_fifo(_e)
    if _state:
      status = self.trans(_state)
    else:
      status = return_status.UNHANDLED

  elif(e.signal == signals.EXIT_SIGNAL):
    rsm(outer, e)
    self.p_spy(e)
    status = return_status.HANDLED
  else:
    self.temp.fun = self.bottom
    status = return_status.SUPER
  return status

@state
def middle(self, e):
  status = return_status.UNHANDLED

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    self.temp.fun = outer
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    rsm(middle, e)
    self.p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    self.p_spy(e)
    rsm(middle, e)
    status = return_status.HANDLED
  elif(self.token_match(e.signal_name, "SB1")):
    self.p_spy(e)
    status = self.meta_trans(
      e=e,
      s=middle,
      t=s,
    )
  elif(self.token_match(e.signal_name, "SRE1")):
    self.p_spy(e)
    status = self.meta_trans(
      e=e,
      s=middle,
      t=p_p11,
    )
  elif(e.signal == signals.EXIT_SIGNAL):
    rsm(middle, e)
    self.p_spy(e)
    status = return_status.HANDLED
  else:
    self.temp.fun = outer
    status = return_status.SUPER
  return status

@state
def s(self, e):
  status = return_status.UNHANDLED

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    self.temp.fun = middle
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    rsm(s, e)
    self.p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    rsm(s, e)
    self.p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    rsm(s, e)
    self.p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.SD2):
    self.p_spy(e)
    status = self.meta_trans(
      e=e,
      s=s,
      t=middle,
    )
  elif(e.signal == signals.SRF1):
    self.p_spy(e)
    status = self.meta_trans(
      e=e,
      s=s,
      t=p_p22_s21,
    )
  else:
    self.temp.fun = middle
    status = return_status.SUPER
  return status

@state
def s_s1(self, e):
  status = return_status.UNHANDLED

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    self.temp.fun = s
    return return_status.SUPER

  if(e.signal == signals.ENTRY_SIGNAL):
    rsm(s_s1, e)
    self.p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    rsm(s_s1, e)
    self.p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    rsm(s_s1, e)
    self.p_spy(e)
    status = return_status.HANDLED
  else:
    self.temp.fun = s
    status = return_status.SUPER
  return status

@state
def p(self, e):
  status = return_status.UNHANDLED

  if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
    self.temp.fun = middle
    return return_status.SUPER

  # enter all regions
  if(e.signal == signals.ENTRY_SIGNAL):
    rsm(p, e)
    self.p_spy(e)
    (_e, _state) = self.meta_peel(e)  # search for INIT_META_SIGNAL
    if _state:
      self.inner._post_fifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_SIGNAL):
    rsm(p, e)
    self.p_spy(e)
    self.inner.post_lifo(Event(signal=signals.enter_region))
    status = return_status.HANDLED
  # any event handled within there regions must be pushed from here
  elif(type(self.regions) == dict and (
      self.token_match(e.signal_name, "e1") or
      self.token_match(e.signal_name, "e2") or
      self.token_match(e.signal_name, "e3") or
      self.token_match(e.signal_name, "e4") or
      self.token_match(e.signal_name, "e5") or
      self.token_match(e.signal_name, "RC1") or
      self.token_match(e.signal_name, "RC2") or
      self.token_match(e.signal_name, "H2") or
      self.token_match(e.signal_name, "RH1") or
      self.token_match(e.signal_name, "SRH2") or
      self.token_match(e.signal_name, "SRG1") or
      self.token_match(e.signal_name, "SRH3") or
      self.token_match(e.signal_name, "RE1") or
      self.token_match(e.signal_name, "RA1") or
      self.token_match(e.signal_name, "RA2") or
      self.token_match(e.signal_name, "PC1") or
      self.token_match(e.signal_name, "PF1") or
      self.token_match(e.signal_name, "RG1") or
      self.token_match(e.signal_name, "RB1") or
      self.token_match(e.signal_name, "RD1") or
      self.token_match(e.signal_name, "PG1") or
      self.token_match(e.signal_name, "PG2") or
      self.token_match(e.signal_name, "SRH1") or
      self.token_match(e.signal_name, "SRD2") or
      self.token_match(e.signal_name, "SRD1") or
      self.token_match(e.signal_name, "RF1") or
      self.token_match(e.signal_name, self.regions['p_p11'].final_signal_name) or
      self.token_match(e.signal_name, self.regions['p_p12'].final_signal_name) or
      self.token_match(e.signal_name, self.regions['p_p22'].final_signal_name) or
      self.token_match(e.signal_name, self.regions['p_p12_p11'].final_signal_name)
  )
  ):
    self.p_spy(e)
    self.inner.post_fifo(e)
    status = return_status.HANDLED
  elif(e.signal == signals.INIT_META_SIGNAL):
    self.p_spy(e)
    # hasn't been updated
    self.regions['p']._post_lifo(Event(signal=signals.force_region_init))
    self.regions['p'].post_fifo(e.payload.event)
    status = return_status.HANDLED
  elif e.signal == signals.BOUNCE_ACROSS_META_SIGNAL:
    self.p_spy(e)
    _e, _state = e.payload.event, e.payload.state
    investigate(self, e, _e)
    self.inner._post_fifo(_e)
    for region in self.inner._regions:
      if region.has_state(e.payload.previous_state):
        region.pop_event()
        region._post_lifo(Event(signal=signals.exit_region))
      else:
        region.post_lifo(Event(signal=signals.enter_region))
    [region._complete_circuit() for region in self.inner._regions]
    status = return_status.HANDLED
  elif(self.token_match(e.signal_name, "SRE2")):
    self.p_spy(e)
    status = self.meta_trans(
      e=e,
      s=p,
      t=p_p11_s12,
    )
  # final token match
  elif(type(self.regions) == dict and self.token_match(e.signal_name,
       self.regions['p'].final_signal_name)):
    self.regions['p'].post_fifo(Event(signal=signals.exit_region))
    status = self.trans(s_s1)
  elif(self.token_match(e.signal_name, "SA1")):
    self.p_spy(e)
    status = self.meta_trans(
      e=e,
      s=p,
      t=p,
    )
  elif(self.token_match(e.signal_name, "H1")):
    self.p_spy(e)
    status = self.meta_trans(
      e=e,
      s=p,
      t=middle
    )
  elif(self.token_match(e.signal_name, "SD1")):
    self.p_spy(e)
    status = self.meta_trans(
      e=e,
      s=p,
      t=middle
    )
  elif(self.token_match(e.signal_name, "SC1")):
    self.p_spy(e)
    status = self.meta_trans(
      e=e,
      s=p,
      t=s
    )
  elif(self.token_match(e.signal_name, "SD1")):
    self.p_spy(e)
    status = self.meta_trans(
      e=e,
      s=p,
      t=middle
    )
  elif(self.token_match(e.signal_name, "SRB1")):
    self.p_spy(e)
    status = self.meta_trans(
      e=e,
      s=p,
      t=p_p22,
    )
  elif(e.signal == signals.EXIT_META_SIGNAL):
    self.p_spy(e)
    (_e, _state) = e.payload.event, e.payload.state
    investigate(self, e, _e)
    self.post_lifo(_e)
    status = return_status.HANDLED
  elif(e.signal == signals.OUTER_TRANS_REQUIRED):
    self.p_spy(e)
    status = return_status.HANDLED
    _state = e.payload.state
    investigate(self, e)
    if _state != p:
      status = self.trans(_state)
    else:
      self.inner.post_fifo(Event(signal=signals.exit_region))
      self.inner.post_fifo(Event(signal=signals.enter_region))
  elif e.signal == signals.BOUNCE_SAME_META_SIGNAL:
    self.p_spy(e)
    _state, _e = e.payload.state, e.payload.event
    self.inner._post_lifo(Event(signal=signals.force_region_init))
    self.inner._post_fifo(_e)
    investigate(self, e, _e)
    self.inner._complete_circuit()
    status = return_status.HANDLED
  elif(e.signal == signals.EXIT_SIGNAL):
    self.inner.post_lifo(Event(signal=signals.exit_region))
    rsm(p, e)
    self.p_spy(e)
    status = return_status.HANDLED
  elif(e.signal == signals.exit_region):
    self._p_spy(e)
    status = return_status.HANDLED
  else:
    self.temp.fun = middle
    status = return_status.SUPER
  return status


################################################################################
#                         REGRESSION TEST                                      #
################################################################################
if __name__ == '__main__':
  regression = True
  active_states = None
  example = XmlChart(
    name='x',
    log_config="xml_chart_5",
    starting_state=outer,
    live_trace=False,
    live_spy=True,
  )

  cache_clear()
  Region.cache_clear()
  XmlChart.cache_clear()

  #example.instrumented = False
  example.instrumented = True
  example.report("starting chart")
  example.start()
  time.sleep(0.20)
  example.report("starting regression")

  if regression:
    def build_test(sig, expected_end_states, old_result, *, wait_sec=0.2, expected_rtc_state_signal_pairs=None):
      '''build a test, wait, then run the test

        **Notes**:
           The test is in two parts, the end-states are checked, then the
           intermediate transitions are checked.  If the end-states test fail
           the user will be informed of the 

           * line number of which the test failed
           * the signal_name as it is applied to the old_result
           * the Expected result
           * the Observed result
           * a Diff of the expected and the observed


        **Args**:
           | ``sig`` (string): signal_name of the event being tested
           | ``expected_end_states`` (recursive list): expected end-states of test
           | ``old_result`` (recursive list): the starting-states before the
           |                                  test is run
           | ``wait_sec=0.2`` (float): wait wait_sec between setting up the test and
           |                           running it
           | ``expected_rtc_state_signal_pairs=None`` (list of tuples): the expected
           |                           state and signal_name pairs that should have
           |                           occured during the test


        **Returns**:
           | (list):  the end states of the test as a recursive list (this can be used
           |          as the old_result for the next test)

        **Example(s)**:

        .. code-block:: python

          old_results = [['p_p11_s11', 'p_p11_s21'], 'p_s21']
          old_results = build_test(
            sig='SD1',
            expected_end_states=['middle'],
            old_result=old_results,
            wait_sec=0.2,
            expected_rtc_state_signal_pairs=\
              [('p_p11_s11', 'EXIT_SIGNAL'),
                ('p_p11_s21', 'EXIT_SIGNAL'),
                ('p_p11', 'EXIT_SIGNAL'),
                ('p_s21', 'EXIT_SIGNAL'),
                ('p', 'EXIT_SIGNAL'),
                ('middle', 'INIT_SIGNAL')]
          )

      '''
      example.post_fifo(Event(signal=sig))
      time.sleep(wait_sec)
      active_states = example.active_states()[:]
      string2 = "{} <- {} == {}".format(str(old_result), sig, str(active_states))
      example.report(string2)

      if active_states != expected_end_states:
        previous_frame = inspect.currentframe().f_back
        fdata = FrameData(*inspect.getframeinfo(previous_frame))
        function_name = '__main__'
        line_number   = fdata.line_number
        example.logger.critical("="*80)
        example.logger.critical("{}:{}".format(function_name, line_number))
        example.logger.critical("END STATE PROBLEM: {}->{}".format(sig, old_result))
        example.logger.critical("EXPECTED: {}".format(expected_end_states))
        example.logger.critical("OBSERVED:  {}".format(active_states))
        example.logger.critical(example.active_states())
        example.logger.critical("="*80)
        time.sleep(60 * 8)
        #time.sleep(5)
        exit(1)

      if expected_rtc_state_signal_pairs:
        matched, diff, observed, expected =\
          diff_state_memory(
            expected=expected_rtc_state_signal_pairs,
            observed=state_memory[:])
        csm()
        try:
          assert(matched)
        except:
          print("="*80)
          previous_frame = inspect.currentframe().f_back
          fdata = FrameData(*inspect.getframeinfo(previous_frame))
          function_name = '__main__'
          line_number   = fdata.line_number
          example.logger.critical("="*80)
          example.logger.critical("{}:{}".format(function_name, line_number))
          example.logger.critical("ORDERING PROBLEM: {} <- {}".format(old_results, sig))
          example.logger.critical("DIFF:")
          example.logger.critical(diff)
          example.logger.critical("EXPECTED:")
          pp(expected)
          #example.logger.critical(expected)
          example.logger.critical("OBSERVED:")
          pp(observed)
          #example.logger.critical(observed)
          example.logger.critical("="*80)
          exit(0)

      #assert active_states == expected_end_states
      return active_states

    def build_reflect(sig, expected_end_states, old_result, wait_sec=0.2):
      '''test function, so it can be slow'''
      example.post_fifo(Event(signal=sig))
      time.sleep(wait_sec)
      active_states = example.active_states()[:]
      string2 = "\n{} <- {} == {}\n".format(str(old_result), sig, str(active_states))
      example.report(string2)
      return active_states

    assert(
      example._meta_hooked(
        s=p,
        t=p_p11,
        sig='H1'
      ) is p_p11.__wrapped__
    )
    assert(
      example._meta_hooked(
        s=p,
        t=p_p11,
        sig='H2'
      ) is None
    )
    assert(
      example._meta_hooked(
        s=p,
        t=p_p12_p11,
        sig='H1'
      ) is p_p12.__wrapped__
    )
    assert(
      example._meta_hooked(
        s=p_p12,
        t=p_p12_s21,
        sig="H1"
      ) is p_p12_s21.__wrapped__
    )
    assert(example.lca(_s=p_p12, _t=outer) == outer)
    assert(example.lca(_s=p_p12, _t=s) == outer)

    result1 = example.build_onion(s=p, t=p_p11, sig='TEST')
    assert(result1 == [p_p11, p_r1_region, p])
    result2 = example.build_onion(s=p_p11, t=p, sig='TEST')
    assert(result2 == [p, p_r1_region, p_p11])

    result1 = example.build_onion(s=s_s1, t=s, sig='TEST')
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
    active_states = example.active_states()
    old_results = example.active_states()[:]

    # print(example.regions['p']._regions[0].final_signal_name)
    # print(example.regions['p_p11'].final_signal_name)

    #example.clear_log()
    old_results = build_test(
      sig='SH1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('outer', 'ENTRY_SIGNAL'),
         ('outer', 'INIT_SIGNAL'),
         ('middle', 'ENTRY_SIGNAL'),
         ('p', 'ENTRY_SIGNAL'),
         ('p', 'INIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    assert(
      example.meta_hooked(
        s=p,
        e=Event(signal=signals.H1)
      ) is p_p11.__wrapped__
    )

    # SD1
    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21']
    old_results = build_test(
      sig='SD1',
      expected_end_states=['middle'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
          ('p_p11_s21', 'EXIT_SIGNAL'),
          ('p_p11', 'EXIT_SIGNAL'),
          ('p_s21', 'EXIT_SIGNAL'),
          ('p', 'EXIT_SIGNAL'),
          ('middle', 'INIT_SIGNAL')]
    )

    # confirm we cannot find any hooks
    assert(
      example.meta_hooked(
        s=middle,
        e=Event(signal=signals.H1)
      ) is None
    )
    assert(
      example.meta_hooked(
        s=middle,
        e=Event(signal=signals.H2)
      ) is None
    )

    # SH1
    #example.clear_log()
    # starting: ['middle'],
    old_results = build_test(
      sig='SH1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [
         ('middle', 'EXIT_SIGNAL'),
         ('middle', 'ENTRY_SIGNAL'),
         ('p', 'ENTRY_SIGNAL'),
         ('p', 'INIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    assert(
      example.meta_hooked(
        s=middle,
        e=Event(signal=signals.H1)
      ) is p_p11.__wrapped__
    )

    assert(
      example.meta_hooked(
        s=middle,
        e=Event(signal=signals.H2)
      ) is None
    )
    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21'],
    old_results = build_test(
      sig='SC1',
      expected_end_states=['s'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL'),
         ('p', 'EXIT_SIGNAL'),
         ('s', 'ENTRY_SIGNAL'),
         ('s', 'INIT_SIGNAL')]
    )
    assert(
      example.meta_hooked(
        s=s,
        e=Event(signal=signals.H1)
      ) is None
    )
    assert(
      example.meta_hooked(
        s=s,
        e=Event(signal=signals.H2)
      ) is None
    )

    #example.clear_log()
    # starting: ['s'],
    old_results = build_test(
      sig='SD2',
      expected_end_states=['middle'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('s', 'EXIT_SIGNAL'), ('middle', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: ['middle'],
    old_results = build_test(
      sig='SB1',
      expected_end_states=['s'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('s', 'ENTRY_SIGNAL'), ('s', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: ['s'],
    old_results = build_test(
      sig='SH1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('s', 'EXIT_SIGNAL'),
         ('middle', 'EXIT_SIGNAL'),
         ('middle', 'ENTRY_SIGNAL'),
         ('p', 'ENTRY_SIGNAL'),
         ('p', 'INIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21'],
    old_results = build_test(
      sig='SA1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL'),
         ('p', 'EXIT_SIGNAL'),
         ('p', 'ENTRY_SIGNAL'),
         ('p', 'INIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21'],
    old_results = build_test(
      sig='SRE2',
      expected_end_states=[['p_p11_s12', 'p_p11_s21'], 'p_s21'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),  # unusable by client
         ('p_p11_s12', 'ENTRY_SIGNAL'),
         ('p_p11_s12', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s12', 'p_p11_s21'], 'p_s21'],
    old_results = build_test(
      sig='SD1',
      expected_end_states=['middle'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s12', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL'),
         ('p', 'EXIT_SIGNAL'),
         ('middle', 'INIT_SIGNAL')]
    )

    # example.clear_log()
    # starting: ['middle']
    old_results = build_test(
      sig='SH1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('middle', 'EXIT_SIGNAL'),
         ('middle', 'ENTRY_SIGNAL'),
         ('p', 'ENTRY_SIGNAL'),
         ('p', 'INIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21']
    old_results = build_test(
      sig='SRE1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL'),
         ('p', 'EXIT_SIGNAL'),
         ('p', 'ENTRY_SIGNAL'),
         ('p', 'INIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21']
    old_results = build_test(
      sig='SRH2',
      expected_end_states=['middle'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL'),
         ('p', 'EXIT_SIGNAL'),
         ('middle', 'INIT_SIGNAL')]
    )


    # example.clear_log()
    # starting: ['middle']
    old_results = build_test(
      sig='SRE3',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], ['p_p22_s11', 'p_p22_s21']],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('middle', 'EXIT_SIGNAL'),
         ('middle', 'ENTRY_SIGNAL'),
         ('p', 'ENTRY_SIGNAL'),
         ('p', 'INIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_p22', 'ENTRY_SIGNAL'),
         ('p_p22', 'INIT_SIGNAL'),
         ('p_p22_s11', 'ENTRY_SIGNAL'),
         ('p_p22_s11', 'INIT_SIGNAL'),
         ('p_p22_s21', 'ENTRY_SIGNAL'),
         ('p_p22_s21', 'INIT_SIGNAL')]
    )
    # p_p11 has an H1 hook
    assert(
      example.meta_hooked(
        s=p,
        e=Event(signal=signals.H1)
      ) is p_p11.__wrapped__
    )

    # p_p22_s11 has an H2 hook
    assert(
      example.meta_hooked(
        s=p,
        e=Event(signal=signals.H2)
      ) is p_p22_s11.__wrapped__
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], ['p_p22_s11', 'p_p22_s21']]
    old_results = build_test(
      sig='e4',
      expected_end_states=[['p_p11_s12', 'p_p11_s21'], ['p_p22_s12', 'p_p22_s21']],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s12', 'ENTRY_SIGNAL'),
         ('p_p11_s12', 'INIT_SIGNAL'),
         ('p_p22_s11', 'EXIT_SIGNAL'),
         ('p_p22_s12', 'ENTRY_SIGNAL'),
         ('p_p22_s12', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s12', 'p_p11_s21'], ['p_p22_s12', 'p_p22_s21']]
    old_results = build_test(
      sig='SRH3',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s12', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_p22_s12', 'EXIT_SIGNAL'),
         ('p_p22_s21', 'EXIT_SIGNAL'),
         ('p_p22', 'EXIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21'],
    old_results = build_test(
      sig='SRD2',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21'],
    old_results = build_test(
      sig='SRH1',
      expected_end_states=['middle'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
          ('p_p11_s21', 'EXIT_SIGNAL'),
          ('p_p11', 'EXIT_SIGNAL'),
          ('p_s21', 'EXIT_SIGNAL'),
          ('p', 'EXIT_SIGNAL'),
          ('middle', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: ['middle'],
    old_results = build_test(
      sig='SH1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('middle', 'EXIT_SIGNAL'),
         ('middle', 'ENTRY_SIGNAL'),
         ('p', 'ENTRY_SIGNAL'),
         ('p', 'INIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )


    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21']
    old_results = build_test(
      sig='SRB1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], ['p_p22_s11', 'p_p22_s21']],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL'),
         ('p_p22', 'ENTRY_SIGNAL'),
         ('p_p22', 'INIT_SIGNAL'),
         ('p_p22_s11', 'ENTRY_SIGNAL'),
         ('p_p22_s11', 'INIT_SIGNAL'),
         ('p_p22_s21', 'ENTRY_SIGNAL'),
         ('p_p22_s21', 'INIT_SIGNAL')]
    )
    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], ['p_p22_s11', 'p_p22_s21']]
    old_results = build_test(
      sig='SA1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_p22_s11', 'EXIT_SIGNAL'),
         ('p_p22_s21', 'EXIT_SIGNAL'),
         ('p_p22', 'EXIT_SIGNAL'),
         ('p', 'EXIT_SIGNAL'),
         ('p', 'ENTRY_SIGNAL'),
         ('p', 'INIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21'],
    old_results = build_test(
      sig='SRB1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], ['p_p22_s11', 'p_p22_s21']],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL'),
         ('p_p22', 'ENTRY_SIGNAL'),
         ('p_p22', 'INIT_SIGNAL'),
         ('p_p22_s11', 'ENTRY_SIGNAL'),
         ('p_p22_s11', 'INIT_SIGNAL'),
         ('p_p22_s21', 'ENTRY_SIGNAL'),
         ('p_p22_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], ['p_p22_s11', 'p_p22_s21']],
    old_results = build_test(
      sig='SRD1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_p22_s11', 'EXIT_SIGNAL'),
         ('p_p22_s21', 'EXIT_SIGNAL'),
         ('p_p22', 'EXIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21'],
    old_results = build_test(
      sig='SB1',
      expected_end_states=['s'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL'),
         ('p', 'EXIT_SIGNAL'),
         ('s', 'ENTRY_SIGNAL'),
         ('s', 'INIT_SIGNAL')]
    )

    # example.clear_log()
    # starting: ['s'],
    old_results = build_test(
      sig='SRF1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], ['p_p22_s11', 'p_p22_s21']],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('s', 'EXIT_SIGNAL'),
         ('p', 'ENTRY_SIGNAL'),
         ('p', 'INIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_p22', 'ENTRY_SIGNAL'),
         ('p_p22', 'INIT_SIGNAL'),
         ('p_p22_s11', 'ENTRY_SIGNAL'),
         ('p_p22_s11', 'INIT_SIGNAL'),
         ('p_p22_s21', 'ENTRY_SIGNAL'),
         ('p_p22_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], ['p_p22_s11', 'p_p22_s21']],
    old_results = build_test(
      sig='SRG1',
      expected_end_states=['s_s1'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_p22_s11', 'EXIT_SIGNAL'),
         ('p_p22_s21', 'EXIT_SIGNAL'),
         ('p_p22', 'EXIT_SIGNAL'),
         ('p', 'EXIT_SIGNAL'),
         ('s', 'ENTRY_SIGNAL'),
         ('s_s1', 'ENTRY_SIGNAL'),
         ('s_s1', 'INIT_SIGNAL')]
    )

    # starting: ['s_s1'],
    old_results = build_test(
      sig='SRF1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], ['p_p22_s11', 'p_p22_s21']],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('s_s1', 'EXIT_SIGNAL'),
         ('s', 'EXIT_SIGNAL'),
         ('p', 'ENTRY_SIGNAL'),
         ('p', 'INIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_p22', 'ENTRY_SIGNAL'),
         ('p_p22', 'INIT_SIGNAL'),
         ('p_p22_s11', 'ENTRY_SIGNAL'),
         ('p_p22_s11', 'INIT_SIGNAL'),
         ('p_p22_s21', 'ENTRY_SIGNAL'),
         ('p_p22_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], ['p_p22_s11', 'p_p22_s21']],
    old_results = build_test(
      sig='e4',
      expected_end_states=[['p_p11_s12', 'p_p11_s21'], ['p_p22_s12', 'p_p22_s21']],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s12', 'ENTRY_SIGNAL'),
         ('p_p11_s12', 'INIT_SIGNAL'),
         ('p_p22_s11', 'EXIT_SIGNAL'),
         ('p_p22_s12', 'ENTRY_SIGNAL'),
         ('p_p22_s12', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s12', 'p_p11_s21'], ['p_p22_s12', 'p_p22_s21']],
    old_results = build_test(
      sig='e1',
      expected_end_states=[['p_p11_r1_final', 'p_p11_s22'], ['p_p22_r1_final', 'p_p22_s22']],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s12', 'EXIT_SIGNAL'),
         ('p_p11_r1_final', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11_s22', 'ENTRY_SIGNAL'),
         ('p_p11_s22', 'INIT_SIGNAL'),
         ('p_p22_s12', 'EXIT_SIGNAL'),
         ('p_p22_r1_final', 'ENTRY_SIGNAL'),
         ('p_p22_s21', 'EXIT_SIGNAL'),
         ('p_p22_s22', 'ENTRY_SIGNAL'),
         ('p_p22_s22', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_r1_final', 'p_p11_s22'], ['p_p22_r1_final', 'p_p22_s22']],
    old_results = build_test(
      sig='e2',
      expected_end_states=[[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_r2_final'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s22', 'EXIT_SIGNAL'),
         ('p_p11_r2_final', 'ENTRY_SIGNAL'),
         ('p_p22_s22', 'EXIT_SIGNAL'),
         ('p_p22_r2_final', 'ENTRY_SIGNAL'),
         ('p_p11_r1_final', 'EXIT_SIGNAL'),
         ('p_p11_r2_final', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_p12', 'ENTRY_SIGNAL'),
         ('p_p12', 'INIT_SIGNAL'),
         ('p_p12_p11', 'ENTRY_SIGNAL'),
         ('p_p12_p11', 'INIT_SIGNAL'),
         ('p_p12_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s11', 'INIT_SIGNAL'),
         ('p_p12_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s21', 'INIT_SIGNAL'),
         ('p_p12_s21', 'ENTRY_SIGNAL'),
         ('p_p12_s21', 'INIT_SIGNAL'),
         ('p_p22_r1_final', 'EXIT_SIGNAL'),
         ('p_p22_r2_final', 'EXIT_SIGNAL'),
         ('p_p22', 'EXIT_SIGNAL'),
         ('p_r2_final', 'ENTRY_SIGNAL')]
    )

    #example.clear_log()
    # starting: [[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_r2_final'],
    old_results = build_test(
      sig='e5',
      expected_end_states=['s_s1'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p12_p11_s11', 'EXIT_SIGNAL'),
         ('p_p12_p11_s21', 'EXIT_SIGNAL'),
         ('p_p12_p11', 'EXIT_SIGNAL'),
         ('p_p12_s21', 'EXIT_SIGNAL'),
         ('p_p12', 'EXIT_SIGNAL'),
         ('p_r1_final', 'ENTRY_SIGNAL'),
         ('p_r1_final', 'EXIT_SIGNAL'),
         ('p_r2_final', 'EXIT_SIGNAL'),
         ('p', 'EXIT_SIGNAL'),
         ('s', 'ENTRY_SIGNAL'),
         ('s_s1', 'ENTRY_SIGNAL'),
         ('s_s1', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: ['s_s1'],
    old_results = build_test(
      sig='SH1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('s_s1', 'EXIT_SIGNAL'),
         ('s', 'EXIT_SIGNAL'),
         ('middle', 'EXIT_SIGNAL'),
         ('middle', 'ENTRY_SIGNAL'),
         ('p', 'ENTRY_SIGNAL'),
         ('p', 'INIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )


    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21'],
    old_results = build_test(
      sig='e1',
      expected_end_states=[['p_p11_s11', 'p_p11_s22'], 'p_s21'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11_s22', 'ENTRY_SIGNAL'),
         ('p_p11_s22', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s22'], 'p_s21'],
    old_results = build_test(
      sig='SA1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s22', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL'),
         ('p', 'EXIT_SIGNAL'),
         ('p', 'ENTRY_SIGNAL'),
         ('p', 'INIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21'],
    old_results = build_test(
      sig='RF1',
      expected_end_states=[[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s21'], ['p_p22_s11', 'p_p22_s21']],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_p12', 'ENTRY_SIGNAL'),
         ('p_p12', 'INIT_SIGNAL'),
         ('p_p12_p11', 'ENTRY_SIGNAL'),
         ('p_p12_p11', 'INIT_SIGNAL'),
         ('p_p12_p11_s12', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s12', 'INIT_SIGNAL'),
         ('p_p12_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s21', 'INIT_SIGNAL'),
         ('p_p12_s21', 'ENTRY_SIGNAL'),
         ('p_p12_s21', 'INIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL'),
         ('p_p22', 'ENTRY_SIGNAL'),
         ('p_p22', 'INIT_SIGNAL'),
         ('p_p22_s11', 'ENTRY_SIGNAL'),
         ('p_p22_s11', 'INIT_SIGNAL'),
         ('p_p22_s21', 'ENTRY_SIGNAL'),
         ('p_p22_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s21'], ['p_p22_s11', 'p_p22_s21']],
    old_results = build_test(
      sig='SA1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p12_p11_s12', 'EXIT_SIGNAL'),
         ('p_p12_p11_s21', 'EXIT_SIGNAL'),
         ('p_p12_p11', 'EXIT_SIGNAL'),
         ('p_p12_s21', 'EXIT_SIGNAL'),
         ('p_p12', 'EXIT_SIGNAL'),
         ('p_p22_s11', 'EXIT_SIGNAL'),
         ('p_p22_s21', 'EXIT_SIGNAL'),
         ('p_p22', 'EXIT_SIGNAL'),
         ('p', 'EXIT_SIGNAL'),
         ('p', 'ENTRY_SIGNAL'),
         ('p', 'INIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21'],
    old_results = build_test(
      sig='e1',
      expected_end_states=[['p_p11_s11', 'p_p11_s22'], 'p_s21'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11_s22', 'ENTRY_SIGNAL'),
         ('p_p11_s22', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s22'], 'p_s21'],
    old_results = build_test(
      sig='PG2',
      expected_end_states=['p_r1_under_hidden_region', 'p_s21'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
        ('p_p11_s22', 'EXIT_SIGNAL'),
        ('p_p11', 'EXIT_SIGNAL'),
        ('p_s21', 'EXIT_SIGNAL'),
        ('p_s21', 'ENTRY_SIGNAL'),
        ('p_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: ['p_r1_under_hidden_region', 'p_s21'],
    old_results = build_test(
      sig='PF1',
      expected_end_states=[[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_r2_under_hidden_region'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p12', 'ENTRY_SIGNAL'),
         ('p_p12', 'INIT_SIGNAL'),      # usable by client
         ('p_p12_p11', 'ENTRY_SIGNAL'),
         ('p_p12_p11', 'INIT_SIGNAL'),  # usuable by client
         ('p_p12_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s11', 'INIT_SIGNAL'),
         ('p_p12_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s21', 'INIT_SIGNAL'),
         ('p_p12_s21', 'ENTRY_SIGNAL'),
         ('p_p12_s21', 'INIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL')]
    )
    #example.clear_log()
    # starting: [[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_r2_under_hidden_region'],
    old_results = build_test(
      sig='RG1',
      expected_end_states=['p_r1_under_hidden_region', ['p_p22_s11', 'p_p22_s21']],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p12_p11_s11', 'EXIT_SIGNAL'),
         ('p_p12_p11_s21', 'EXIT_SIGNAL'),
         ('p_p12_p11', 'EXIT_SIGNAL'),
         ('p_p12_s21', 'EXIT_SIGNAL'),
         ('p_p12', 'EXIT_SIGNAL'),
         ('p_p22', 'ENTRY_SIGNAL'),
         ('p_p22', 'INIT_SIGNAL'),
         ('p_p22_s11', 'ENTRY_SIGNAL'),
         ('p_p22_s11', 'INIT_SIGNAL'),
         ('p_p22_s21', 'ENTRY_SIGNAL'),
         ('p_p22_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: ['p_r1_under_hidden_region', ['p_p22_s11', 'p_p22_s21']],
    old_results = build_test(
      sig='SA1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p22_s11', 'EXIT_SIGNAL'),
         ('p_p22_s21', 'EXIT_SIGNAL'),
         ('p_p22', 'EXIT_SIGNAL'),
         ('p', 'EXIT_SIGNAL'),
         ('p', 'ENTRY_SIGNAL'),
         ('p', 'INIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21'],
    old_results = build_test(
      sig='PC1',
      expected_end_states=['p_r1_under_hidden_region', 'p_s21'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: ['p_r1_under_hidden_region', 'p_s21'],
    old_results = build_test(
      sig='SA1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_s21', 'EXIT_SIGNAL'),
         ('p', 'EXIT_SIGNAL'),
         ('p', 'ENTRY_SIGNAL'),
         ('p', 'INIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21'],
    old_results = build_test(
      sig='RC1',
      expected_end_states=[[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], ['p_p22_s11', 'p_p22_s21']],
      old_result = old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_p12', 'ENTRY_SIGNAL'),
         ('p_p12', 'INIT_SIGNAL'),
         ('p_p12_p11', 'ENTRY_SIGNAL'),
         ('p_p12_p11', 'INIT_SIGNAL'),
         ('p_p12_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s11', 'INIT_SIGNAL'),
         ('p_p12_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s21', 'INIT_SIGNAL'),
         ('p_p12_s21', 'ENTRY_SIGNAL'),
         ('p_p12_s21', 'INIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL'),
         ('p_p22', 'ENTRY_SIGNAL'),
         ('p_p22', 'INIT_SIGNAL'),
         ('p_p22_s11', 'ENTRY_SIGNAL'),
         ('p_p22_s11', 'INIT_SIGNAL'),
         ('p_p22_s21', 'ENTRY_SIGNAL'),
         ('p_p22_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], ['p_p22_s11', 'p_p22_s21']],
    old_results = build_test(
      sig='RC2',
      expected_end_states=[[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_s21'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p22_s11', 'EXIT_SIGNAL'),
         ('p_p22_s21', 'EXIT_SIGNAL'),
         ('p_p22', 'EXIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_s21'],
    old_results = build_test(
      sig='e1',
      expected_end_states=[[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s22'], 'p_s21'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p12_p11_s11', 'EXIT_SIGNAL'),
         ('p_p12_p11_s12', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s12', 'INIT_SIGNAL'),
         ('p_p12_s21', 'EXIT_SIGNAL'),
         ('p_p12_s22', 'ENTRY_SIGNAL'),
         ('p_p12_s22', 'INIT_SIGNAL')]
    )

    assert(
      example.meta_hooked(
        s=p,
        e=Event(signal=signals.H1)
      ) is p_p12.__wrapped__
    )

    assert(
      example.meta_hooked(
        s=p,
        e=Event(signal=signals.H2)
      ) is None
    )

    #example.clear_log()
    # starting: [[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s22'], 'p_s21'],
    old_results = build_test(
      sig='RB1',
      expected_end_states=[[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_s21'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p12_p11_s12', 'EXIT_SIGNAL'),
         ('p_p12_p11_s21', 'EXIT_SIGNAL'),
         ('p_p12_p11', 'EXIT_SIGNAL'),
         ('p_p12_p11', 'ENTRY_SIGNAL'),
         ('p_p12_p11', 'INIT_SIGNAL'),
         ('p_p12_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s11', 'INIT_SIGNAL'),
         ('p_p12_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s21', 'INIT_SIGNAL'),
         ('p_p12_s22', 'EXIT_SIGNAL'),
         ('p_p12_s21', 'ENTRY_SIGNAL'),
         ('p_p12_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_s21'],
    old_results = build_test(
      sig='e1',
      expected_end_states=[[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s22'], 'p_s21'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p12_p11_s11', 'EXIT_SIGNAL'),
         ('p_p12_p11_s12', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s12', 'INIT_SIGNAL'),
         ('p_p12_s21', 'EXIT_SIGNAL'),
         ('p_p12_s22', 'ENTRY_SIGNAL'),
         ('p_p12_s22', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s22'], 'p_s21'],
    old_results = build_test(
      sig='RE1',
      expected_end_states=[[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_s21'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p12_p11_s12', 'EXIT_SIGNAL'),
         ('p_p12_p11_s21', 'EXIT_SIGNAL'),
         ('p_p12_p11', 'EXIT_SIGNAL'),
         ('p_p12_p11', 'ENTRY_SIGNAL'),
         ('p_p12_p11', 'INIT_SIGNAL'),
         ('p_p12_p11_s12', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s12', 'INIT_SIGNAL'),
         ('p_p12_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s21', 'INIT_SIGNAL'),
         ('p_p12_s22', 'EXIT_SIGNAL'),
         ('p_p12_s21', 'ENTRY_SIGNAL'),
         ('p_p12_s21', 'INIT_SIGNAL')]
    )
    #example.clear_log()
    # starting: [[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_s21'],
    old_results = build_test(
      sig='RH1',
      expected_end_states=[[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_s21'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p12_p11_s12', 'EXIT_SIGNAL'),
         ('p_p12_p11_s21', 'EXIT_SIGNAL'),
         ('p_p12_p11', 'EXIT_SIGNAL'),
         ('p_p12_s21', 'EXIT_SIGNAL'),
         ('p_p12_p11', 'ENTRY_SIGNAL'),
         ('p_p12_p11', 'INIT_SIGNAL'),
         ('p_p12_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s11', 'INIT_SIGNAL'),
         ('p_p12_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s21', 'INIT_SIGNAL'),
         ('p_p12_s21', 'ENTRY_SIGNAL'),
         ('p_p12_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_s21'],
    old_results = build_test(
      sig='e1',
      expected_end_states=[[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s22'], 'p_s21'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p12_p11_s11', 'EXIT_SIGNAL'),
         ('p_p12_p11_s12', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s12', 'INIT_SIGNAL'),
         ('p_p12_s21', 'EXIT_SIGNAL'),
         ('p_p12_s22', 'ENTRY_SIGNAL'),
         ('p_p12_s22', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s22'], 'p_s21'],
    old_results = build_test(
      sig='RD1',
      expected_end_states=[[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_s21'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p12_p11_s12', 'EXIT_SIGNAL'),
         ('p_p12_p11_s21', 'EXIT_SIGNAL'),
         ('p_p12_p11', 'EXIT_SIGNAL'),
         ('p_p12_s22', 'EXIT_SIGNAL'),
         ('p_p12_p11', 'ENTRY_SIGNAL'),
         ('p_p12_p11', 'INIT_SIGNAL'),
         ('p_p12_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s11', 'INIT_SIGNAL'),
         ('p_p12_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s21', 'INIT_SIGNAL'),
         ('p_p12_s21', 'ENTRY_SIGNAL'),
         ('p_p12_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_s21'],
    old_results = build_test(
      sig='SA1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result= old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p12_p11_s11', 'EXIT_SIGNAL'),
         ('p_p12_p11_s21', 'EXIT_SIGNAL'),
         ('p_p12_p11', 'EXIT_SIGNAL'),
         ('p_p12_s21', 'EXIT_SIGNAL'),
         ('p_p12', 'EXIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL'),
         ('p', 'EXIT_SIGNAL'),
         ('p', 'ENTRY_SIGNAL'),
         ('p', 'INIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21'],
    old_results = build_test(
      sig='e1',
      expected_end_states=[['p_p11_s11', 'p_p11_s22'], 'p_s21'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11_s22', 'ENTRY_SIGNAL'),
         ('p_p11_s22', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s22'], 'p_s21'],
    old_results = build_test(
      sig='RA1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s22', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21'],
    old_results = build_test(
      sig='p_p11_final',
      expected_end_states=[[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_s21'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_p12', 'ENTRY_SIGNAL'),
         ('p_p12', 'INIT_SIGNAL'),
         ('p_p12_p11', 'ENTRY_SIGNAL'),
         ('p_p12_p11', 'INIT_SIGNAL'),
         ('p_p12_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s11', 'INIT_SIGNAL'),
         ('p_p12_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s21', 'INIT_SIGNAL'),
         ('p_p12_s21', 'ENTRY_SIGNAL'),
         ('p_p12_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], 'p_s21'],
    old_results = build_test(
      sig='PG1',
      expected_end_states=['p_r1_under_hidden_region', ['p_p22_s11', 'p_p22_s21']],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p12_p11_s11', 'EXIT_SIGNAL'),
         ('p_p12_p11_s21', 'EXIT_SIGNAL'),
         ('p_p12_p11', 'EXIT_SIGNAL'),
         ('p_p12_s21', 'EXIT_SIGNAL'),
         ('p_p12', 'EXIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL'),
         ('p_p22', 'ENTRY_SIGNAL'),
         ('p_p22', 'INIT_SIGNAL'),
         ('p_p22_s11', 'ENTRY_SIGNAL'),
         ('p_p22_s11', 'INIT_SIGNAL'),
         ('p_p22_s21', 'ENTRY_SIGNAL'),
         ('p_p22_s21', 'INIT_SIGNAL')]
    )

    #########################################################################
    #                              HOOK TESTS                               #
    #########################################################################
    #example.clear_log()
    # starting: ['p_r1_under_hidden_region', ['p_p22_s11', 'p_p22_s21']],
    old_results = build_test(
      sig='SH1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p22_s11', 'EXIT_SIGNAL'),
         ('p_p22_s21', 'EXIT_SIGNAL'),
         ('p_p22', 'EXIT_SIGNAL'),
         ('p', 'EXIT_SIGNAL'),
         ('middle', 'EXIT_SIGNAL'),
         ('middle', 'ENTRY_SIGNAL'),
         ('p', 'ENTRY_SIGNAL'),
         ('p', 'INIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21'],
    old_results = build_test(
      sig='PC1',
      expected_end_states=['p_r1_under_hidden_region', 'p_s21'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    # confirm transition works to middle from p
    #example.clear_log()
    # starting: ['p_r1_under_hidden_region', 'p_s21'],
    old_results = build_test(
      sig='H1',
      expected_end_states=['middle'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_s21', 'EXIT_SIGNAL'), ('p', 'EXIT_SIGNAL'), ('middle', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: ['middle'],
    old_results = build_test(
      sig='SH1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('middle', 'EXIT_SIGNAL'),
         ('middle', 'ENTRY_SIGNAL'),
         ('p', 'ENTRY_SIGNAL'),
         ('p', 'INIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21'],
    old_results = build_test(
      sig='H1',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        []
    )

    #example.clear_log()
    # starting: [['p_p11_s11', 'p_p11_s21'], 'p_s21'],
    old_results = build_test(
      sig='RF1',
      expected_end_states=[[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s21'], ['p_p22_s11', 'p_p22_s21']],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_p12', 'ENTRY_SIGNAL'),
         ('p_p12', 'INIT_SIGNAL'),
         ('p_p12_p11', 'ENTRY_SIGNAL'),
         ('p_p12_p11', 'INIT_SIGNAL'),
         ('p_p12_p11_s12', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s12', 'INIT_SIGNAL'),
         ('p_p12_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s21', 'INIT_SIGNAL'),
         ('p_p12_s21', 'ENTRY_SIGNAL'),
         ('p_p12_s21', 'INIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL'),
         ('p_p22', 'ENTRY_SIGNAL'),
         ('p_p22', 'INIT_SIGNAL'),
         ('p_p22_s11', 'ENTRY_SIGNAL'),
         ('p_p22_s11', 'INIT_SIGNAL'),
         ('p_p22_s21', 'ENTRY_SIGNAL'),
         ('p_p22_s21', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s21'], ['p_p22_s11', 'p_p22_s21']],
    old_results = build_test(
      sig='e1',
      expected_end_states=[[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s22'], ['p_p22_s11', 'p_p22_s22']],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p12_s21', 'EXIT_SIGNAL'),
         ('p_p12_s22', 'ENTRY_SIGNAL'),
         ('p_p12_s22', 'INIT_SIGNAL'),
         ('p_p22_s21', 'EXIT_SIGNAL'),
         ('p_p22_s22', 'ENTRY_SIGNAL'),
         ('p_p22_s22', 'INIT_SIGNAL')]
    )

    #example.clear_log()
    # starting: [[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s22'], ['p_p22_s11', 'p_p22_s22']],
    old_results = build_test(
      sig='H2',
      expected_end_states=[[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s22'], ['p_p22_s11', 'p_p22_s22']],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        []
    )

    # starting: [[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s22'], ['p_p22_s11', 'p_p22_s22']],
    old_results = build_test(
      sig='e4',
      expected_end_states=[['p_p12_s12', 'p_p12_s22'], ['p_p22_s12', 'p_p22_s22']],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p12_p11_s12', 'EXIT_SIGNAL'),
         ('p_p12_p11_s21', 'EXIT_SIGNAL'),
         ('p_p12_p11', 'EXIT_SIGNAL'),
         ('p_p12_s12', 'ENTRY_SIGNAL'),
         ('p_p12_s12', 'INIT_SIGNAL'),
         ('p_p22_s11', 'EXIT_SIGNAL'),
         ('p_p22_s12', 'ENTRY_SIGNAL'),
         ('p_p22_s12', 'INIT_SIGNAL')]
    )

    # starting: [['p_p12_s12', 'p_p12_s22'], ['p_p22_s12', 'p_p22_s22']],
    old_results = build_test(
      sig='H2',
      expected_end_states=[['p_p11_s11', 'p_p11_s21'], 'p_s21'],
      old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p12_s12', 'EXIT_SIGNAL'),
         ('p_p12_s22', 'EXIT_SIGNAL'),
         ('p_p12', 'EXIT_SIGNAL'),
         ('p_p22_s12', 'EXIT_SIGNAL'),
         ('p_p22_s22', 'EXIT_SIGNAL'),
         ('p_p22', 'EXIT_SIGNAL'),
         ('p_p11', 'ENTRY_SIGNAL'),
         ('p_p11', 'INIT_SIGNAL'),
         ('p_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p11_s11', 'INIT_SIGNAL'),
         ('p_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p11_s21', 'INIT_SIGNAL'),
         ('p_s21', 'ENTRY_SIGNAL'),
         ('p_s21', 'INIT_SIGNAL')]
    )

    old_results = build_test(
      sig='RF1',
      expected_end_states=[[['p_p12_p11_s12', 'p_p12_p11_s21'], 'p_p12_s21'], ['p_p22_s11', 'p_p22_s21']], old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p11_s11', 'EXIT_SIGNAL'),
         ('p_p11_s21', 'EXIT_SIGNAL'),
         ('p_p11', 'EXIT_SIGNAL'),
         ('p_p12', 'ENTRY_SIGNAL'),
         ('p_p12', 'INIT_SIGNAL'),
         ('p_p12_p11', 'ENTRY_SIGNAL'),
         ('p_p12_p11', 'INIT_SIGNAL'),
         ('p_p12_p11_s12', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s12', 'INIT_SIGNAL'),
         ('p_p12_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s21', 'INIT_SIGNAL'),
         ('p_p12_s21', 'ENTRY_SIGNAL'),
         ('p_p12_s21', 'INIT_SIGNAL'),
         ('p_s21', 'EXIT_SIGNAL'),
         ('p_p22', 'ENTRY_SIGNAL'),
         ('p_p22', 'INIT_SIGNAL'),
         ('p_p22_s11', 'ENTRY_SIGNAL'),
         ('p_p22_s11', 'INIT_SIGNAL'),
         ('p_p22_s21', 'ENTRY_SIGNAL'),
         ('p_p22_s21', 'INIT_SIGNAL')]
    )

    old_results = build_test(
      sig='RA2',
      expected_end_states=[[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], ['p_p22_s11', 'p_p22_s21']], old_result=old_results,
      wait_sec=0.2,
      expected_rtc_state_signal_pairs=\
        [('p_p12_p11_s12', 'EXIT_SIGNAL'),
         ('p_p12_p11_s21', 'EXIT_SIGNAL'),
         ('p_p12_p11', 'EXIT_SIGNAL'),
         ('p_p12_s21', 'EXIT_SIGNAL'),
         ('p_p12', 'EXIT_SIGNAL'),
         ('p_p12', 'ENTRY_SIGNAL'),
         ('p_p12', 'INIT_SIGNAL'),
         ('p_p12_p11', 'ENTRY_SIGNAL'),
         ('p_p12_p11', 'INIT_SIGNAL'),
         ('p_p12_p11_s11', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s11', 'INIT_SIGNAL'),
         ('p_p12_p11_s21', 'ENTRY_SIGNAL'),
         ('p_p12_p11_s21', 'INIT_SIGNAL'),
         ('p_p12_s21', 'ENTRY_SIGNAL'),
         ('p_p12_s21', 'INIT_SIGNAL')]
    )

