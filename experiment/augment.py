import sys
from miros import Event
from miros import signals
from functools import wraps
from functools import partial
from miros import HsmWithQueues
from miros import return_status
from collections import namedtuple

################################################################################
#                                   BASELINE                                   #
################################################################################
#  +-------------- a ---------------+
#  |                                |
#  |    +-------b----------+        |
#  |    | entry /          |        |
#  | *-->  print('hello')  |        |       +-----c-----+
#  |    |                  |        |       |           |
#  |    |                  |        +---e1-->           |
#  |    |                  |<-------e2------+           |
#  |    +------------------+        |       +-----------+
#  |                                |
#  +--------------------------------+
def a(hsm, e):
    status = return_status.UNHANDLED
    if e.signal == signals.ENTRY_SIGNAL:
        print("a {}".format(e.signal_name))
        status = return_status.HANDLED
    elif e.signal == signals.INIT_SIGNAL:
        print("a {}".format(e.signal_name))
        status = hsm.trans(b)
    elif e.signal == signals.e1:
        print("a {}".format(e.signal_name))
        status = hsm.trans(c)
    else:
        hsm.temp.fun = hsm.top
        status = return_status.SUPER
    return status


def b(hsm, e):
    status = return_status.UNHANDLED
    if e.signal == signals.ENTRY_SIGNAL:
        print("b {}".format(e.signal_name))
        status = return_status.HANDLED
    if e.signal == signals.INIT_SIGNAL:
        print("b {}".format(e.signal_name))
        status = return_status.HANDLED
    else:
        hsm.temp.fun = a
        status = return_status.SUPER
    return status


def c(hsm, e):
    status = return_status.UNHANDLED
    if e.signal == signals.ENTRY_SIGNAL:
        print("c {}".format(e.signal_name))
        status = return_status.HANDLED
    elif e.signal == signals.INIT_SIGNAL:
        print("c {}".format(e.signal_name))
        status = return_status.HANDLED
    elif e.signal == signals.e2:
        print("c {}".format(e.signal_name))
        status = hsm.trans(b)
    else:
        hsm.temp.fun = hsm.top
        status = return_status.SUPER
    return status


bars = 16
print("-" * bars + " simple hsm")

vanilla = HsmWithQueues()
vanilla.instrumented = True
vanilla.start_at(a)
vanilla.post_fifo(Event(signal=signals.e1))
vanilla.post_fifo(Event(signal=signals.e2))
vanilla.complete_circuit()

print("-" * bars + " constructed hsm")

################################################################################
#                     FUNCTIONS BUILT FROM SPECIFICATIONS                      #
################################################################################

module_namespace = sys.modules[__name__]

SignalAndFunction = namedtuple("SignalAndFunction", 
    ["signal", "function"]
)

StateSpecification = namedtuple(
    "StateSpecification",
    ["function", "super_state", "super_state_name", "signal_function_list"],
)


def orthogonal_state(fn):
    '''stand in for the orthogonal_state instrumentation decorator'''
    @wraps(fn)
    def _pspy_on(*args, **kwargs):
        status = fn(*args, **kwargs)
        return status

    return _pspy_on


def token_match(me, other):
    '''stand in for token_match function'''
    return me == other


def parse_spec(spec):
    '''map a state function spec onto a string'''
    function_name = spec.function.__name__
    result = f"{function_name}:{spec.function}\n"
    result += f"{spec.super_state_name}:{spec.super_state}\n"
    if spec.signal_function_list:
        for (signal, function) in spec.signal_function_list:
            function_name = spec.function.__name__
            sn = signals.name_for_signal(signal)
            result += f"  {sn}:{signal} -> {function_name}:{function}\n"
    return result


def __template_state(
    hsm=None,
    e=None,
    sfl=None,
    *,
    super_state=None,
    this_function_name=None,
    imposed_behavior=None,
    remove_behavior=None,
    suggested_behavior=None,
    specification=None,
    signal=None,
    handler=None,
    hooks=None,
    **kwargs,
):
    """A function which can create itself.

    Use this to build and run states in a state machine.

    **Note:**

      This function has three different jobs:

        1. It change itself:

            1. You can impose behavior
            2. You can suggest behavior
            3. You can remove behavior

          After the function changes itself, it rebuilds itself using a partial,
          adds it decorator then assigns this new version of itself to the
          global namespace.

        2. It can be called for its behavioral specification

        3. It can be called as a regular state function.

    **Args**:
       | ``hsm=None`` (HsmWithQueues):
       | ``e=None`` (Event):
       | ``sfl=None`` ([]):
       | ``*``:  all arguments from now on have to be named
       | ``super_state=None`` (fn):
       | ``this_function_name=None`` (str):
       | ``imposed_behavior=None`` (bool|SignalAndFunction):
       | ``remove_behavior=None`` (bool|SignalAndFunction):
       | ``suggested_behavior=None`` (bool|SignalAndFunction):
       | ``specification=None`` (bool):
       | ``signal=None`` (int):
       | ``handler=None`` (callable):
       | ``**kwargs``: arbitrary keyword arguments


    **Returns**:
       (return_status|fn|StateSpecification|str):

    **Example(s)**:

    .. code-block:: python

      from miros import Event
      from miros import signals
      from functools import partial
      from miros import HsmWithQueues

      #  +-------------- a ---------------+
      #  |                                |
      #  |    +-------b----------+        |
      #  |    | entry /          |        |
      #  | *-->  print('hello')  |        |       +-----c-----+
      #  |    |                  |        |       |           |
      #  |    |                  |        +---e1-->           |
      #  |    |                  |<-------e2------+           |
      #  |    +------------------+        |       +-----------+
      #  |                                |
      #  +--------------------------------+

      # these will be the state handlers
      def template_trans(hsm, e, state_name=None, *, trans=None, **kwargs):
          status = return_status.HANDLED
          assert trans
          print("{} {}".format(state_name, e.signal_name))
          status = hsm.trans(getattr(module_namespace, trans))
          return status

      def template_handled(hsm, e, state_name=None, **kwargs):
          status = return_status.HANDLED
          print("{} {}".format(state_name, e.signal_name))
          return status

      hsm = HsmWithQueues()
      a1 = partial(__template_state, this_function_name="a1", super_state=hsm.top)()
      b1 = partial(__template_state, this_function_name="b1", super_state=a1)()
      c1 = partial(__template_state, this_function_name="c1", super_state=hsm.top)()

      a1_entry_hh = partial(template_handled)
      a1_init_hh = partial(template_trans, trans="b1")
      a1_e1_hh = partial(template_trans, trans="c1")

      a1 = a1(hsm=hsm, signal=signals.ENTRY_SIGNAL, handler=a1_entry_hh)
      a1 = a1(hsm=hsm, signal=signals.INIT_SIGNAL, handler=a1_init_hh)
      a1 = a1(hsm=hsm, signal=signals.e1, handler=a1_e1_hh)

      b1_entry_hh = partial(template_handled)
      b1_init_hh = partial(template_handled)

      b1 = b1(hsm=hsm, signal=signals.ENTRY_SIGNAL, handler=b1_entry_hh)
      b1 = b1(hsm=hsm, signal=signals.INIT_SIGNAL, handler=b1_init_hh)

      c1_entry_hh = partial(template_handled)
      c1_init_hh = partial(template_handled)
      c1_e2_hh = partial(template_trans, trans="b1")

      c1 = c1(hsm=hsm, signal=signals.ENTRY_SIGNAL, handler=c1_entry_hh)
      c1 = c1(hsm=hsm, signal=signals.INIT_SIGNAL, handler=c1_init_hh)
      c1 = c1(hsm=hsm, signal=signals.e2, handler=c1_e2_hh)

      # Run our hsm
      hsm.start_at(a1)
      hsm.post_fifo(Event(signal=signals.e1))
      hsm.post_fifo(Event(signal=signals.e2))
      hsm.complete_circuit()


    """
    if e is None:
        rebuild_function = False

        if not hasattr(module_namespace, this_function_name):
            rebuild_function = True

        if isinstance(imposed_behavior, SignalAndFunction):
            if sfl == None:
                sfl = []
            sfl = [
                SignalAndFunction(signal=signal, function=function)
                for (signal, function) in sfl
                if signal != imposed_behavior.signal
            ]
            sfl.append(imposed_behavior)
            rebuild_function = True
        if signal and not handler:
            if remove_behavior is not None and remove_behavior == True:
                behavior = SignalAndFunction(signal=signal, function=None)
                return getattr(module_namespace, this_function_name)(
                    hsm=hsm, remove_behavior=behavior
                )
        elif signal and handler:
            function = partial(
                __template_handler,
                state_function_name=this_function_name,
                signal=signal,
                handler=handler,
            )
            behavior = SignalAndFunction(signal=signal, function=function)
            if suggested_behavior is not None and suggested_behavior == True:
                return getattr(module_namespace, this_function_name)(
                    hsm=hsm, suggested_behavior=behavior
                )
            elif imposed_behavior is None or imposed_behavior == True:
                return getattr(module_namespace, this_function_name)(
                    hsm=hsm, imposed_behavior=behavior
                )
            else:
                return getattr(module_namespace, this_function_name)

        if isinstance(remove_behavior, SignalAndFunction):
            if sfl == None:
                pass
            else:
                sfl = [
                    SignalAndFunction(signal=signal, function=function)
                    for (signal, function) in sfl
                    if signal != remove_behavior.signal
                ]
            rebuild_function = True
        if isinstance(suggested_behavior, SignalAndFunction):
            if sfl == None:
                []
            else:
                if (
                    len(
                        [
                            SignalAndFunction(signal=signal, function=function)
                            for (signal, function) in sfl
                            if signal == suggested_behavior.signal
                        ]
                    )
                    >= 1
                ):
                    # the signal is already there, we are ignoring your request
                    print("... ignoring")
                    rebuild_function = False
                else:
                    rebuild_function = True
                    sfl.append(suggested_behavior)

        if rebuild_function:
            fn = partial(
                __template_state,
                sfl=sfl,
                super_state=super_state,
                this_function_name=this_function_name,
                hooks=hooks
            )
            fn.__name__ = this_function_name
            fn = orthogonal_state(fn)
            fn.__name__ = this_function_name
            setattr(module_namespace, this_function_name, fn)
            if hsm:
                if hasattr(hsm.temp, "fun"):
                    if hsm.temp.fun.__name__ == this_function_name:
                        hsm.temp.fun = fn
                if hasattr(hsm.state, "fun"):
                    if hsm.state.fun.__name__ == this_function_name:
                        hsm.state.fun = fn
            return getattr(module_namespace, this_function_name)

        if specification:
            spec = StateSpecification(
                function=getattr(module_namespace, this_function_name),
                super_state=super_state,
                super_state_name=super_state.__name__,
                signal_function_list=sfl,
            )
            return spec
        return getattr(module_namespace, this_function_name)

    if e.signal == signals.GET_STATE_SPEC:
        spec = StateSpecification(
            function=getattr(module_namespace, this_function_name),
            super_state=super_state,
            super_state_name=super_state.__name__,
            signal_function_list=sfl,
        )
        return spec
    elif e.signal == signals.REFLECTION_SIGNAL:
        return this_function_name

    handler = None

    if isinstance(sfl, list):
        for (signal, function) in sfl:
            if token_match(signal, e.signal):
                if this_function_name == 'p_p11' and e.signal == signals.e4:
                  print("here")
                handler = function
                break
    if handler:
        status = handler(hsm, e, state_function_name=this_function_name)
    else:
        if hsm.top == super_state:
            hsm.temp.fun = super_state
        else:
            hsm.temp.fun = getattr(module_namespace, super_state.__name__)
        status = return_status.SUPER
    return status


def __template_handler(
    hsm=None,
    e=None,
    handler=None,
    *,
    state_function_name=None,
    this_function_name=None,
    signal=None,
    specification=None,
    get_state_function_name=None,
    **kwargs,
):
    """This function is intended to be used by the ``__template_state``.

    A template for building event and handler functions, once it is built, it
    will serve as a function to call when its state function receives a specific
    signal: ex. ``signals.ENTRY_SIGNAL``

    **Note:**

      This function has three different jobs:

        * it creates itself: you provide it with specifications and it builds
          itself using functools.partial with some pre-loaded arguments that you
          provide (another function).  The new function, is registered with the
          module_namespace and when you can its by name again it will behave the
          way you have specified.

        * it describes its specification, you can call it with the specification
          named argument and it will return a SignalAndFunction namedtuple 

        * if you call it with an event, it will dispatch the hsm, event and this
          function's name to the handler it when you were specifying its behavior.

    **Args**:
       | ``hsm=None`` (HsmWithQueues):
       | ``e=None`` (Event):
       | ``handler=None`` (callable):
       | ``*`` all arguments from now on have to be named:
       | ``state_function_name=None`` (str):
       | ``this_function_name=None`` (str):
       | ``signal=None`` (int):
       | ``specification=None`` (bool):
       | ``get_state_function_name=None`` (str):
       | ``**kwargs``: arbitrary keyword arguements


    **Returns**:
       (return_status|fn|StateSpecification|str):

    **Example(s)**:

    .. code-block:: python

      def template_handled(hsm, e, state_name=None, **kwargs):
          status = return_status.HANDLED
          print("{} {}".format(state_name, e.signal_name))
          return status

      a1_entry_h = partial(template_handled)

      # here we are using this function
      behavior = SignalAndFunction(
        __template_state,
        state_method_name='a1',
        signal=signals.ENTRY_SIGNAL,
        handler=a1_entry_h)

      # build a state
      a1 = partial(__template_state, this_function_name="a1", super_state=hsm.top)()

      # now we impose the behavior
      a1(imposed_behavior=behavior)

    """
    if e is None:
        rebuild_function = False

        if get_state_function_name:
            return state_function_name

        if this_function_name is None:
            this_function_name = (
                state_function_name + "_" + signals.name_for_signal(signal).lower()
            )

        if not hasattr(module_namespace, this_function_name):
            rebuild_function = True

        if rebuild_function:
            fn = partial(
                __template_handler,
                this_function_name=this_function_name,
                signal=signal,
                handler=handler,
            )
            fn.__name__ = this_function_name
            setattr(module_namespace, this_function_name, fn)
            return getattr(module_namespace, this_function_name)

        if specification:
            spec = SignalAndFunction(
                function=getattr(module_namespace, this_function_name), signal=signal
            )
            return spec
    else:
        if e.signal == signals.GET_STATE_SPEC:
            spec = SignalAndFunction(
                function=getattr(module_namespace, this_function_name), signal=signal
            )
            return spec

    if e is None:
        return getattr(module_namespace, this_function_name)
    else:
        return handler(hsm, e, state_function_name)


# these will be the state handlers
def template_trans(hsm, e, state_name=None, *, trans=None, **kwargs):
    '''print out the state name and then transition'''
    status = return_status.HANDLED
    assert trans
    print("{} {}".format(state_name, e.signal_name))
    status = hsm.trans(getattr(module_namespace, trans))
    return status


def template_handled(hsm, e, state_name=None, **kwargs):
    status = return_status.HANDLED
    print("{} {}".format(state_name, e.signal_name))
    return status


hsm = HsmWithQueues()
# Define our hsm
a1 = partial(__template_state, this_function_name="a1", super_state=hsm.top)(construct=True)
b1 = partial(__template_state, this_function_name="b1", super_state=a1)(construct=True)
c1 = partial(__template_state, this_function_name="c1", super_state=hsm.top)(construct=True)

a1_entry_hh = partial(template_handled)
a1_init_hh = partial(template_trans, trans="b1")
a1_e1_hh = partial(template_trans, trans="c1")

a1 = a1(hsm=hsm, signal=signals.ENTRY_SIGNAL, handler=a1_entry_hh)
a1 = a1(hsm=hsm, signal=signals.INIT_SIGNAL, handler=a1_init_hh)
a1 = a1(hsm=hsm, signal=signals.e1, handler=a1_e1_hh)

b1_entry_hh = partial(template_handled)
b1_init_hh = partial(template_handled)

b1 = b1(hsm=hsm, signal=signals.ENTRY_SIGNAL, handler=b1_entry_hh)
b1 = b1(hsm=hsm, signal=signals.INIT_SIGNAL, handler=b1_init_hh)

c1_entry_hh = partial(template_handled)
c1_init_hh = partial(template_handled)
c1_e2_hh = partial(template_trans, trans="b1")

c1 = c1(hsm=hsm, signal=signals.ENTRY_SIGNAL, handler=c1_entry_hh)
c1 = c1(hsm=hsm, signal=signals.INIT_SIGNAL, handler=c1_init_hh)
c1 = c1(hsm=hsm, signal=signals.e2, handler=c1_e2_hh)

# Run our hsm
hsm.start_at(a1)
hsm.post_fifo(Event(signal=signals.e1))
hsm.post_fifo(Event(signal=signals.e2))
hsm.complete_circuit()

print("-" * bars + " suggested_behavior for behavior already defined")

# Suggest a behavior
a1_entry_hh = partial(template_handled)
a1(hsm=hsm, signal=signals.ENTRY_SIGNAL, handler=a1_entry_hh, suggested_behavior=True)


print("-" * bars + " suggested_behavior that wasn't there before")
b1_e2_hh = partial(template_trans, trans="c1")
b1 = b1(hsm=hsm, signal=signals.e2, handler=b1_e2_hh, suggested_behavior=True)
# b1_e2_hh = partial(template_trans, trans='a1')
hsm.post_fifo(Event(signal=signals.e2))
hsm.complete_circuit()
print("-" * bars + " imposing_behavior")
c1_e2_hh = partial(template_trans, trans="a1")
c1 = c1(hsm=hsm, signal=signals.e2, handler=c1_e2_hh)
hsm.post_fifo(Event(signal=signals.e2))
hsm.complete_circuit()
print("-" * bars + " reading the specification")
print(parse_spec(a1(specification=True)))
print("-" * bars + " removing a behavior")
print("removing e2:{} from c1".format(signals.e2))
print("c1 specification before removal")
print(parse_spec(c1(specification=True)))
c1 = c1(hsm=hsm, signal=signals.e2, remove_behavior=True)
print("c1 specification after removal")
print(parse_spec(c1(specification=True)))

