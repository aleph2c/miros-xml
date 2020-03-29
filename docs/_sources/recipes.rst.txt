
.. _recipes:

   *I'm writing these docs to help myself understand how to build an XML-to-miros parser.*

.. _recipes-recipes:

Recipes
=======

.. contents::
   :backlinks: entry

.. _recipes-summary:

Summary
^^^^^^^

I wrote this document to help me think through my problems and to try and find a
more straightforward way to implement the parallel tag.

I want miros-XML to support parallel statecharts using the dashed line, as
described by David Harel’s notation. The miros library uses an event dispatching
algorithm developed by Miro Samek. This event dispatching algorithm does
not support the Parallel statechart (orthogonal regions). However, Miro Samek
offered an alternative and faster pattern, the `orthogonal component
<https://aleph2c.github.io/miros/patterns.html#patterns-orthogonal-component>`_; an HSM
within an HSM. The problem with this orthogonal component pattern is that it
isn’t as graphically elegant as the one envisioned by David Harel. So the
statechart community expects to use orthogonal regions. David Harel’s graphical
abstractions permit more than one active state to exist within one diagram at a
time; the packing of design complexity into a small diagram’s space is very
efficient. The trade-off is you waste a lot of computer cycles when you design
this way.
If you haven’t seen orthogonal regions before, refer to `this document on
orthogonal regions and miros.
<https://aleph2c.github.io/miros/othogonalregions.html>`_.

In this project and these documents, I will be writing about how to have one
orthogonal region communicate with another. My implementation of orthogonal
regions will be a mapping of Miros Samek’s orthogonal component pattern onto a
design that will look and behave like an orthogonal region. This work will be
done by recursively mapping orthogonal components within a chart. An outer
region will ``dispatch`` events into a deeper or inner region. The outer region
which injects an event will be called an Injector and the region which receives
the events will be called an Injectee. An HSM can both be an Injector and an
Injectee, and the outermost chart will only be an Injector.

.. image:: _static/hidden_dynamics.svg
    :target: _static/hidden_dynamics.pdf
    :align: center

The chart will be run within the outermost HSM's thread, and the inner regions
will be driven by dispatching events and driving them through the orthogonal
component with the ``complete_circuit`` method.

To manifest transitions between regions, the queue of the various components
will be treated as call stacks into which instructions can be placed.  Otherwise
the majority of the chart will behave using the dynamics of the miros algorithm.

This document is being written so that I can invent enough theory and give
myself enough instruction so as to generalize the construction of any orthogonal
region, described by an XML document, within its ``<parallel>`` tag.

.. _recipes-context-and-terminology:

Context and Terminology
^^^^^^^^^^^^^^^^^^^^^^^

   .. note::

     There is something cluttered about the terminology, come back and pull
     things apart so that each thing has one meaning.

-----

    Region
         HsmWithQueues object with additional supporting methods; in the software, it is
         represented as a class. In the design, it is an orthogonal component called the
         injectee. In statechart theory, the orthogonal region is one of the areas
         partitioned within a dashed line. It and its other regions are expected to run
         concurrently.

         .. image:: _static/region.svg
             :target: _static/region.pdf
             :align: center


    Regions
         Regions contain multiple Region objects in a collection. It augments the regions
         so that they can reference one another using iterators. It adds a _post_fifo and
         _post_lifo method, which can put items into all of its inner region’s queues.

         It adds a post_fifo and post_lifo method, which will post items onto an internal
         queue, then drive the inner statecharts using their complete_circuit method
         until their queues are empty.

         .. image:: _static/regions.svg
             :target: _static/regions.pdf
             :align: center

         All other inner state methods, connected to a particular regions HSM
         will have an ``@p_spy_on`` decorator.

    XmlChart
         Is the main statechart into which all events are posted.  It is the
         outer most injector.

         .. image:: _static/XmlChart.svg
             :target: _static/XmlChart.pdf
             :align: center

         The XmlChart will have a dict containing multiple Regions.  The
         XmlChart's thread will drive the entire collection of orthogonal
         regions.  It will inject all of the events, via the regions dictionary.

         All of the XmlChart connected state methods will have a ``@spy_on``
         decorator.  All other inner state methods will have an ``@p_spy_on``
         decorator.

    Injector:
         A statechart or orthogonal component which posts events into an inner orthogonal
         component then drives the events through the region that injector encapsulates.
         The injector places events and drives events using the _post_fifo', post_fifo …
         APIs. The ``p`` and ``pp12`` states are both injectors in the following diagram:

         .. image:: _static/hidden_dynamics.svg
             :target: _static/hidden_dynamics.pdf
             :align: center

    Injectee:
          An orthogonal component who’s events are given to it and driven through it
          by an injector. An injectee can also be an injector if it drives another
          state. An injector is hidden from view in the main statechart picture with
          the dashed lines. However, you can see it at the bottom right of this
          picture.


         .. image:: _static/hidden_dynamics.svg
             :target: _static/hidden_dynamics.pdf
             :align: center

    WTF events
         Any event which crosses between regions.  See ``E0``, ``E1`` and ``E3``
         in the following diagram

         .. image:: _static/hidden_dynamics.svg
             :target: _static/hidden_dynamics.pdf
             :align: center

         The WTF events are not supported within the miros event processing
         algorithim.  This document was written, largely to understand how to
         implement these events for the miros-xml parser.

    Under Hidden States
         The outer state of an **injectee**.  It presents the illusion that a
         region can be exited.  It's a holding state with no initialization
         handler, and it is the bottom state of the HSM mapped to that region.

    Region States
         This state is sandwiched between the under and outer hidden states.  It
         contains a ``INIT_SIGNAL`` handler which calls the ``init_stack``
         method which is used for managing ``META_INIT`` events (events which
         contain events and states to initialize to).  The region state also has
         a ``region_exit`` event handler which will cause the region to
         transition the **under hidden region**.

    Over Hidden States
         This state is above the region state and its purpose to to catch the
         ``force_region_init`` event, which will force a transition to the
         region, and thereby force a call of the ``init_state`` method.

    META_INIT
         An event which contains 0 or more META_INIT events and the state which
         are intended to handle the event.  They are injected into the queue of
         inner states so that the inner state's ``init_stack`` methods can
         programmatically initialize their region.

    META_EXIT
         An event which permits a WTF exit strategy

    META_SIGNAL_PAYLOAD
         The payload of a META event.

         .. code-block:: python

            META_SIGNAL_PAYLOAD = namedtuple(
               "META_SIGNAL_PAYLOAD", ['event', 'state', 'source_event', 'region']
            )

.. _recipes-reflection-and-logging:

Reflection and Logging
^^^^^^^^^^^^^^^^^^^^^^
.. _recipes-suped-up-spy:

Souped Up Spy
------------

It would be almost impossible to tackle this problem without the spy
instrumentation.  To get the spy instrumentation working within the orthogonal
regions I wrote this wrapper and placed it above each region or state within a
region:

.. code-block:: python
  :emphasize-lines: 22
  :linenos:

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

You can see on line 22 I have filtered out any spy line with the name
``SEARCH_FOR_SUPER``.  This was to reduce the amount of noise in the
instrumentation.

The spy itself is written to a log file and/or written to the terminal.

.. _recipes-suped-up-state-name-reflection:

Souped Up State Name Reflection
-------------------------------

If you use the vanilla ``state_name`` method provided within miros you will only
be able to see the outer most state holding the orthogonal regions; but it will
not reach into this collection of orthogonal regions and report on the active state
of each of them.

To see all of the active states at once using the ``active_states`` method of
the ``XmlChart`` class.

.. image:: _static/xml_chart_4.svg
    :target: _static/xml_chart_4.pdf
    :align: center

.. code-block:: python
  :emphasize-lines: 15
  :linenos:

  example = XmlChart(
    name='parallel',
    log_file="/mnt/c/github/miros-xml/experiment/parallel_example_4.log",
    live_trace=True,
    live_spy=True,
  )

  example.start()
  time.sleep(0.01)

  example.post_fifo(Event(signal=signals.to_p))
  time.sleep(0.01)
  active_states = example.active_states()
  print("{:>10} -> {}".format("to_p", active_states))
  assert active_states == [['p_p11_s11', 'p_p11_s21'], 'p_s21']

In the above listing we see how the chart is created, started and how you can
send a ``to_p`` event into it, then we ask it for its active states.  We see it
reports ``[['p_p11_s11', 'p_p11_s21'], 'p_s21']``, which describes all of it's
current states and some regional information by having nested lists.  The
outermost list represents the whole chart and the inner list represents that
``p_p11_s11`` and ``p_p11_s21`` are within a parallel region.

To code required to make ``active_states`` is within the ``XmlChart`` class:

.. code-block:: python

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

.. _recipes-simplifying-the-inner-injector-functions:

Simplifying the Inner Injector Functions
----------------------------------------
The inner regions will need to access XmlChart methods and attributes to work.

The spy scribble method will be contained in the XmlChart object.  It will need
to be accessed by state functions used by the inner regions.  The ``outmost``
attribute can be used to access any item of the XmlChart object from within an
inner Region object.

Here is an example of how to post to the fifo of the ``p_p11`` region from
anywhere within the state chart.

.. code-block:: python

  region.outmost.regions['p_p11'].post_fifo(Event(signal=signals.some_signal))

The region accesses the outmost part of itself, the XmlChart object, then
accesses its regions dict with the 'p_p11' key, then post to that subregion's
post_fifo queu, the drives that event through that orthogonal region before
returning control back to the program.  There is a lot going on, but it is very
noisy.

Consider how we would use a the spy scribble within an inner region:

.. code-block:: python

  if region.outmost.live_spy and region.outmost.instrumented:
    region.outmost.live_spy_callback("[{}] {}".format(region.name, string))

There are common functions that will be called over and over again within the
inner region's injectors and to tighten up the code an
``outmost_region_functions`` function writer was made.  It looks like this:

.. code-block:: python
  :linenos:

   @lru_cache(maxsize=32)
   def outmost_region_functions(region, region_name):

     outmost = region.outmost
     def scribble(string):
       if outmost.live_spy and outmost.instrumented:
         outmost.live_spy_callback("[{}] {}".format(region_name, string))

     post_fifo = partial(outmost.regions[region_name].post_fifo, outmost=outmost)
     _post_fifo = partial(outmost.regions[region_name]._post_fifo, outmost=outmost)
     post_lifo = partial(outmost.regions[region_name].post_lifo, outmost=outmost)
     _post_lifo = partial(outmost.regions[region_name]._post_lifo, outmost=outmost)
     token_match = outmost.token_match
     return post_fifo, _post_fifo, post_lifo, _post_lifo, token_match, scribble


The functools partial method is used to prefill arguments to the ``post_fifo``,
``_post_fifo``, ``post_lifo``, ``_post_lifo`` and ``token_match`` methods.  A
custom ``scribble`` function is written and returned as well.

On line 1 we see that the result is cashed to speed up calls the
``outmost_region_functions``.

At the top of any injector you will see this ``outmost_region_functions``,
function builder used like this:

.. code-block:: python

   @p_spy_on
   def p_p11(r, e):
     # ..
     (post_fifo,
      _post_fifo,
      post_lifo,
      _post_lifo,
      token_match,
      scribble) = outmost_region_functions(r, 'p_p11')

      # inner region's state function code here


.. _recipes-reading-the-log-file:

Reading the Log File
---------------
The XmlChart contains the thread which drives the parallel processes.  It can
push events through each of the inner orthogonal components with calls to the
``complete_circuit`` method of each region.  However, this makes reading the
logs a bit confusing, since an orthogonal region's actions appear to occur
before XmlChart event handling which drove those actions in the first place.
This should become a bit more clear with an example, consider the following log
snippet:

.. code-block:: log
  :emphasize-lines: 28-32

   S: [x] to_p:outer_state
   S: [x] [p] ENTRY_SIGNAL
   S: [x] [p_r1] enter_region:p_r1_under_hidden_region
   S: [x] [p_r1] ENTRY_SIGNAL:p_r1_region
   S: [x] [p_r1] INIT_SIGNAL:p_r1_region
   S: [x] [p_r1] ENTRY_SIGNAL:p_r1_over_hidden_region
   S: [x] [p_p11] ENTRY_SIGNAL
   S: [x] [p_p11_r1] enter_region:p_p11_r1_under_hidden_region
   S: [x] [p_p11_r1] ENTRY_SIGNAL:p_p11_r1_region
   S: [x] [p_p11_r1] INIT_SIGNAL:p_p11_r1_region
   S: [x] [p_p11_r1] ENTRY_SIGNAL:p_p11_r1_over_hidden_region
   S: [x] [p_p11_r1] ENTRY_SIGNAL:p_p11_s11
   S: [x] [p_p11_r1] INIT_SIGNAL:p_p11_s11
   S: [x] [p_p11_r2] enter_region:p_p11_r2_under_hidden_region
   S: [x] [p_p11_r2] ENTRY_SIGNAL:p_p11_r2_region
   S: [x] [p_p11_r2] INIT_SIGNAL:p_p11_r2_region
   S: [x] [p_p11_r2] ENTRY_SIGNAL:p_p11_r2_over_hidden_region
   S: [x] [p_p11_r2] ENTRY_SIGNAL:p_p11_s21
   S: [x] [p_p11_r2] INIT_SIGNAL:p_p11_s21
   S: [x] [p_r1] ENTRY_SIGNAL:p_p11
   S: [x] [p_r1] INIT_SIGNAL:p_p11
   S: [x] [p_r2] enter_region:p_r2_under_hidden_region
   S: [x] [p_r2] ENTRY_SIGNAL:p_r2_region
   S: [x] [p_r2] INIT_SIGNAL:p_r2_region
   S: [x] [p_r2] ENTRY_SIGNAL:p_r2_over_hidden_region
   S: [x] [p_r2] ENTRY_SIGNAL:p_s21
   S: [x] [p_r2] INIT_SIGNAL:p_s21
   S: [x] to_p:outer_state
   S: [x] SEARCH_FOR_SUPER_SIGNAL:p
   S: [x] ENTRY_SIGNAL:p
   S: [x] INIT_SIGNAL:p
   S: [x] <- Queued:(0) Deferred:(0)
   R:
   ['outer_state'] <- to_p == [['p_p11_s11', 'p_p11_s21'], 'p_s21']

The highlighted code describes event handling of the XmlChart which drove the
actions seen above that part of the listing.  The output of the R: tells us how
this happened in the first place.  The system was in a ``outer_state`` then it
received a ``to_p`` event, which caused it to enter a number of parallel states,
``[['p_p11_s11', 'p_p11_s21'], 'p_s21']``.  To see how this happened, you would
read the logs before the highlighted section.

With enough effort I would make the log file linear in time, but it might not be
worth the effort.

.. _recipes-building-a-subregion:

Building a Subregion
^^^^^^^^^^^^^^^^^^^^

We will build ``p_p11`` in the following diagram:

.. image:: _static/xml_chart_4.svg
    :target: _static/xml_chart_4.pdf
    :align: center

To build the ``p_p11`` subregion you will need to:

1. Create an injector:

.. code-block:: python

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
    elif(outmost.token_match(
      e.signal_name, outmost.regions['p_p11'].final_signal_name)):
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
    elif(e.signal == signals.EXIT_SIGNAL or
         e.signal == signals.region_exit):
      if outmost.live_spy and outmost.instrumented:
        outmost.live_spy_callback(
          "{}:p_p11".format(Event(signal=signals.region_exit)))
      outmost.regions['p_p11'].post_lifo(Event(signal=signals.region_exit))
      status = return_status.HANDLED
    else:
      r.temp.fun = p_r1_over_hidden_type
      status = return_status.SUPER
    return status

2. Create the injectee states.  These are the under_hidden, region, and over_hidden state for
that subregion of the orthogonal component which behaves like a subregion:

.. code-block:: python

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

  # ..

3. Ensure all signals which are passed into the region are injected by outer injectors:

.. code-block:: python

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

4. Add the region to the XmlChart's regions dict within the XmlChart
   ``__init__`` method:

.. code-block:: python

  outer = self.regions['p']
  self.regions['p_p11'] = Regions(
    name='p_p11',
    outmost=self)\
  .add('p_p11_r1', outer=outer)\
  .add('p_p11_r2', outer=outer).link()

.. _recipes-writing-wtf-events-across-regions:

Writing WTF Events Across Regions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section will contain the recipes needed to construct the blue ``WTF``
events, or events that span across parallel regions in this example program.
The ``xml_chart_4`` diagram shown below is based upon the `hsm comprehensive
diagram in the miros project
<https://aleph2c.github.io/miros/_static/comprehensive_no_instrumentation.pdf>`_.


.. image:: _static/xml_chart_4.svg
    :target: _static/xml_chart_4.pdf
    :align: center

.. note::

  ``WTF`` is a backronym and it stands for "Witness The Fitness" (lifted from
  my friend Jen Farroll's `personal training business <http://www.witnessthefitness.ca>`_).

.. _recipes-e-events:

E WTF Events
------------

The ``E`` events start at the edge of a parallel region, then go deeper into the
chart.  See ``E0``, ``E1`` and ``E2`` in the diagram below.

.. image:: _static/xml_chart_4.svg
    :target: _static/xml_chart_4.pdf
    :align: center

The ``E`` events in the orthogonal component mapping start at an injector, then
are dispatched to all regions managed by that injector.  The ``E`` event is
caught then turned into a ``META_INIT`` which may contain 0 or more
``META_INIT`` events as payloads within it.  This is explained in detail in the
``E0`` section.  The ``META_INIT`` is kind of like an onion event, each layer
corresponding to either an injector or injectee part of the design.

.. _recipes-e0-wtf-event:

E0 WTF Event
------------
The ``E0`` event occurs from the outer most threaded state chart and it passes over
multiple regional boundaries.

.. image:: _static/xml_chart_4.svg
    :target: _static/xml_chart_4.pdf
    :align: center

This WTF meta event is initially captured in the ``outer_state`` function:

.. code-block:: python

   @spy_on
   def outer_state(self, e):
     status = return_status.UNHANDLED
     # ...
     elif(self.token_match(e.signal_name, "E0")):
       pprint("enter outer_state")
       if self.live_spy and self.instrumented:
         self.live_spy_callback("{}:outer_state".format(e.signal_name))
       _e = self.meta_init(t=p_p11_s22, sig=e.signal_name)
       self.scribble(payload_string(_e))
       self.post_fifo(_e.payload.event)
       status = self.trans(_e.payload.state)
     # ...

To build a state chart and send it an ``E0`` event, you would type the
following:

.. code-block:: python

  example = XmlChart(
    name='x',
    log_file="/mnt/c/github/miros-xml/experiment/parallel_example_4.log",
    live_trace=False,
    live_spy=True,
  )
  example.post_fifo(Event(signal="E0"))

To see what happens we can view the log:

.. code-block:: python
  :emphasize-lines: 39-50

   S: [x] E0:outer_state
   S: [x] [p_r1] <- Queued:(0) Deferred:(0)
   S: [x] [p_r2] <- Queued:(0) Deferred:(0)
   S: [x] [p_p11_r1] <- Queued:(0) Deferred:(0)
   S: [x] [p_p11_r2] <- Queued:(0) Deferred:(0)
   S: [x] [p_p12_r1] <- Queued:(0) Deferred:(0)
   S: [x] [p_p12_r2] <- Queued:(0) Deferred:(0)
   S: [x] [p_p12_p11_r1] <- Queued:(0) Deferred:(0)
   S: [x] [p_p12_p11_r2] <- Queued:(0) Deferred:(0)
   S: [x] [p_p22_r1] <- Queued:(0) Deferred:(0)
   S: [x] [p_p22_r2] <- Queued:(0) Deferred:(0)
   S: [x] [p] ENTRY_SIGNAL
   S: [x] [p_r1] enter_region:p_r1_under_hidden_region
   S: [x] [p_r1] ENTRY_SIGNAL:p_r1_region
   S: [x] [p_r1] INIT_SIGNAL:p_r1_region
   S: [x] [p_r1] POST_FIFO:META_INIT
   S: [x] [p_r1] ENTRY_SIGNAL:p_r1_over_hidden_region
   S: [x] [p_p11] ENTRY_SIGNAL
   S: [x] [p_p11_r1] enter_region:p_p11_r1_under_hidden_region
   S: [x] [p_p11_r1] ENTRY_SIGNAL:p_p11_r1_region
   S: [x] [p_p11_r1] INIT_SIGNAL:p_p11_r1_region
   S: [x] [p_p11_r1] ENTRY_SIGNAL:p_p11_r1_over_hidden_region
   S: [x] [p_p11_r1] ENTRY_SIGNAL:p_p11_s11
   S: [x] [p_p11_r1] INIT_SIGNAL:p_p11_s11
   S: [x] [p_p11_r2] enter_region:p_p11_r2_under_hidden_region
   S: [x] [p_p11_r2] ENTRY_SIGNAL:p_p11_r2_region
   S: [x] [p_p11_r2] INIT_SIGNAL:p_p11_r2_region
   S: [x] [p_p11_r2] ENTRY_SIGNAL:p_p11_r2_over_hidden_region
   S: [x] [p_p11_r2] ENTRY_SIGNAL:p_p11_s22
   S: [x] [p_p11_r2] INIT_SIGNAL:p_p11_s22
   S: [x] [p_r1] ENTRY_SIGNAL:p_p11
   S: [x] [p_r1] INIT_SIGNAL:p_p11
   S: [x] [p_r2] enter_region:p_r2_under_hidden_region
   S: [x] [p_r2] ENTRY_SIGNAL:p_r2_region
   S: [x] [p_r2] INIT_SIGNAL:p_r2_region
   S: [x] [p_r2] ENTRY_SIGNAL:p_r2_over_hidden_region
   S: [x] [p_r2] ENTRY_SIGNAL:p_s21
   S: [x] [p_r2] INIT_SIGNAL:p_s21
   S: [x] E0:outer_state
   S: [x] 
   META_INIT <function p at 0x7f5d25d526a8> ->
      META_INIT <function p_r1_region at 0x7f5d25d496a8> ->
         META_INIT <function p_p11 at 0x7f5d25d498c8> ->
            META_INIT <function p_p11_r2_region at 0x7f5d25d4b1e0> ->
               META_INIT <function p_p11_s22 at 0x7f5d25d4b510> ->
   S: [x] POST_FIFO:META_INIT
   S: [x] SEARCH_FOR_SUPER_SIGNAL:p
   S: [x] ENTRY_SIGNAL:p
   S: [x] INIT_SIGNAL:p
   S: [x] <- Queued:(0) Deferred:(0)
   R:
   ['outer_state'] <- E0 == [['p_p11_s11', 'p_p11_s22'], 'p_s21']

----

**Analysis:**

We see at the bottom of the log (highlighted) how the ``E0`` creates a
``META_INIT`` event which contains other ``META_INIT`` events.

The key to understanding how the transitions occur is to track this
``META_INIT`` event from the ``outer_state`` to the ``p_p11_s22`` state.

.. code-block:: python
  :emphasize-lines: 9
  :linenos:

   @spy_on
   def outer_state(self, e):
     status = return_status.UNHANDLED
     # ...
     elif(self.token_match(e.signal_name, "E0")):
       pprint("enter outer_state")
       if self.live_spy and self.instrumented:
         self.live_spy_callback("{}:outer_state".format(e.signal_name))
       _e = self.meta_init(t=p_p11_s22, sig=e.signal_name)
       self.scribble(payload_string(_e))
       self.post_fifo(_e.payload.event)
       status = self.trans(_e.payload.state)
     # ...

On line 9 meta_init is used to create the ``META_INIT``.  As of line 9:

.. code-block:: python

   _e.payload.state = p
   _e.payload.event = 
      META_INIT <function p_r1_region at 0x7f5d25d496a8> ->
         META_INIT <function p_p11 at 0x7f5d25d498c8> ->
            META_INIT <function p_p11_r2_region at 0x7f5d25d4b1e0> ->
               META_INIT <function p_p11_s22 at 0x7f5d25d4b510> ->

On line 10 ``_e``'s contents are injected into the log which we can see in the
previous listing.  On line 11, we place ``_e.payload.event`` into the fifo of
our XmlChart statechart.  On line 12 we transition to ``_e.payload.state`` (``p``).

Let's look at the important part of the ``p`` state function:

.. code-block:: python
  :emphasize-lines: 9
  :linenos:

  @spy_on
  def p(self, e):
    status = return_status.UNHANDLED

    # enter all regions
    if(e.signal == signals.ENTRY_SIGNAL):
      if self.live_spy and self.instrumented:
        self.live_spy_callback("[p] {}".format(e.signal_name))
      (_e, _state) = self.init_stack(e) # search for META_INIT
      if _state:
        self.regions['p']._post_fifo(_e)
      pprint("enter p")
      self.regions['p'].post_lifo(Event(signal=signals.enter_region), outmost=self)
      status = return_status.HANDLED
   # ..

The ``p`` function is the first injector.  We see on line 2 the word ``self``,
which by convention, tells us we are in a thread connected statechart and not a
orthogonal-region's HSM.

On line 9 we see that the next event and state are stripped off of the
``META_INIT`` which is sitting in the FIFO queue of the XmlChart.  This is an
exotic way to program, very eccentric.  Normally you do not touch the queues,
you let the framework handle this information for you, we are breaking this
rule, and use the queue as a kind of programming callstack.

As of line 9:

.. code-block:: python

   _state = p_r1_region
   _e = META_INIT <function p_p11 at 0x7f5d25d498c8> ->
          META_INIT <function p_p11_r2_region at 0x7f5d25d4b1e0> ->
             META_INIT <function p_p11_s22 at 0x7f5d25d4b510> ->

The ``init_stack`` method looks like this:

.. code-block:: python
  :emphasize-lines: 5
  :linenos:

  def init_stack(self, e):
    result = (None, None)
    if len(self.queue) >= 1 and \
      self.queue[0].signal == signals.META_INIT:
      _e = self.queue.popleft()
      result = (_e.payload.event, _e.payload.state)
    return result

If there is an event on the queue and it is an ``META_INIT`` then we pop it off
the stack.  We do this before the underlying miros framework has a chance to
handle it.  We parasitize the FIFO for our own purpose and the miros framework
is none the wiser for it.

Finally we return the event and the state information on line 7.

Next consider line 10-11 of the ``p`` listing:

.. code-block:: python
  :emphasize-lines: 10-11
  :linenos:

  @spy_on
  def p(self, e):
    status = return_status.UNHANDLED

    # enter all regions
    if(e.signal == signals.ENTRY_SIGNAL):
      if self.live_spy and self.instrumented:
        self.live_spy_callback("[p] {}".format(e.signal_name))
      (_e, _state) = self.init_stack(e) # search for META_INIT
      if _state:
        self.regions['p']._post_fifo(_e)
      pprint("enter p")
      self.regions['p'].post_lifo(Event(signal=signals.enter_region), outmost=self)
      status = return_status.HANDLED
   # ..

After ``init_stack`` peals off the first onion layer of our ``META_INIT`` event,
we place its inner contents into the ``p`` subregion's FIFO using the ``_post_fifo`` method.

Any posting event with a ``_`` prepended to it, by convention does not drive the
event through its inner regions, it just posts items onto their queues:

.. code-block:: python

  def _post_fifo(self, e, outmost=None):
    [region.post_fifo(e) for region in self._regions]

The ``p`` region has two sub-regions, ``p_r1`` and ``p_r2``. The ``p_r1`` has
these state functions:

   * p_r1_under_hidden_region
   * p_r1_region
   * p_r1_over_hidden_region
   * p_p11 (injector)
   * p_p12 (injector)
   * p_r1_final

The ``p_r2`` has these state functions:

   * p_r2_under_hidden_region
   * p_r2_region
   * p_r2_over_hidden_region
   * p_r2_final
   * p_s21
   * p_p22 (injector)

Looking back to ``p``, on line 13 we see how META_INIT is driven into the internal regions:

.. code-block:: python
  :emphasize-lines: 13
  :linenos:

  @spy_on
  def p(self, e):
    status = return_status.UNHANDLED

    # enter all regions
    if(e.signal == signals.ENTRY_SIGNAL):
      if self.live_spy and self.instrumented:
        self.live_spy_callback("[p] {}".format(e.signal_name))
      (_e, _state) = self.init_stack(e) # search for META_INIT
      if _state:
        self.regions['p']._post_fifo(_e)
      pprint("enter p")
      self.regions['p'].post_lifo(Event(signal=signals.enter_region), outmost=self)
      status = return_status.HANDLED

The ``p`` region's queue has a ``META_INIT`` event in it, on line 13 we push the
``enter_region`` event ahead of it using the ``post_lifo`` event.  This causes
the ``enter_region`` event to both barge ahead of the ``META_INIT`` event in
both of the ``p_r1`` and ``p_r2`` queues.

The ``post_lifo`` event does two things, it posts using a lifo technique then
drives all events through the inner regions using the ``complete_circuit``
method:

.. code-block:: python

  def post_lifo(self, e, outmost=None):
    self._post_lifo(e)
    [region.complete_circuit() for region in self._regions]

After the ``post_lifo`` call on line 13 of the p listing, there is an
``enter_region`` event and a ``META_INIT`` event on both the ``p_r1`` and
``p_r2`` orthogonal region queues.  To see what happens we need to look at our
abstract HSM strategy:

.. image:: _static/hidden_dynamics2.svg
    :target: _static/hidden_dynamics2.pdf
    :align: center

An ``enter_region`` causes the transitions from ``p_r1_under_hidden_region``
and ``p_r2_under_hidden_region`` to ``p_r1_region`` and ``p_r2_region``
respectively.  Then the ``INIT_SIGNAL`` signal of the ``p_r1_region`` and
``p_r2_region`` state functions are fired.  To see what happens next we look at
the ``p_r1_region`` injectee function:

.. code-block:: python
  :emphasize-lines: 5-14
  :linenos:

  @p_spy_on
  def p_r1_region(r, e):
    status = return_status.UNHANDLED
    # ...
    elif(e.signal == signals.INIT_SIGNAL):
      (_e, _state) = r.init_stack(e) # search for META_INIT
      # if _state is a child of this state then transition to it
      if _state is None or not r.has_a_child(p_r1_region, _state):
        status = r.trans(p_p11)
      else:
        status = r.trans(_state)
        if not _e is None:
          r.post_fifo(_e)
   # ...

On line 6 we see another layer is pealed off the ``META_INIT`` event.  If the
``_state`` information isn't present or the target state is not a child of the
``p_r1_region`` state then we fall back to our default initialization; if line 8
returns true, the ``_e`` event is thrown in the garbage and the default behavior
of the initialization occurs.  For this region the default behavior to
transition to ``p_p11``.

But line 8 returns false in our situation because:

.. code-block:: python

   _state = p_p11
   _e = META_INIT <function p_p11_r2_region at 0x7f5d25d4b1e0> ->
          META_INIT <function p_p11_s22 at 0x7f5d25d4b510> ->

So we transition to the value of ``_state``, ``p_p11``, and we post ``_e`` into our fifo and
drive the event through to completion.  Looking back to our log trace we can see
that this ``_state`` variable would have been ``p_p11``, which is the injector
for the next internal region.

So let's look at that ``p_p11`` injector.

.. code-block:: python
  :emphasize-lines: 1
  :linenos:

  @p_spy_on
  def p_p11(r, e):
    '''
    r is either p_r1, p_r2 region
    r.outer = p
    '''
    status = return_status.UNHANDLED
    outmost = r.outmost
    (post_fifo,
     _post_fifo,
     post_lifo,
     _post_lifo,
     token_match,
     scribble) = outmost_region_functions(r, 'p_p11')

    # enter all regions
    if(e.signal == signals.ENTRY_SIGNAL):
      pprint("enter p_p11")
      scribble(e.signal_name)
      (_e, _state) = r.init_stack(e) # search for META_INIT
      if _state:
        _post_fifo(_e, outmost=outmost)
      post_lifo(Event(signal=signals.enter_region))
      status = return_status.HANDLED
  # ...

We see the same pattern we saw in the ``p`` injector.  If there is an
``META_INIT`` event waiting in the queue, it is pealed.  If there is ``_state``
information in the pealing, remaining part of the event is place into the fifo
of the ``p_p11`` region, then that region's state handlers are sent an
``enter_region`` event.

As of line 9 the following is true:

.. code-block:: python

   _state = p_p11_r2_region
   _e = META_INIT <function p_p11_s22 at 0x7f5d25d4b510> ->

The ``p_p11`` region has two sub-regions, ``p_p11_r1`` and ``p_p11_r2``.  The
``p_p11_r1`` region has these state functions.

* p_p11_r1_under_hidden_region
* p_p11_r1_region
* p_p11_r1_over_hidden_region
* p_p11_r1_final
* p_p11_s11
* p_p11_s12

The ``p_p11_r2`` region has these state functions.

* p_p11_r2_under_hidden_region
* p_p11_r2_region
* p_p11_r2_over_hidden_region
* p_p11_r2_final
* p_p11_s21
* p_p11_s22

.. image:: _static/hidden_dynamics2.svg
    :target: _static/hidden_dynamics2.pdf
    :align: center

The same injectee pattern is seen again.  The ``enter_region`` causes the
transitions from ``p_p11_r1_under_hidden_region`` and
``p_p11_r1_under_hidden_region`` to ``p_p11_r1_region`` and ``p_p11_r2_region``
respectively.  Then the ``INIT_SIGNAL`` clause of ``p_p11_r1_region`` and
``p_p11_r2_region`` functions are activated.  To see what happens next we look
at the ``p_p11_r2_region`` injectee function:

.. code-block:: python
  :emphasize-lines: 6, 8, 11
  :linenos:

  @p_spy_on
  def p_p11_r2_region(rr, e):
    status = return_status.UNHANDLED
    # ... 
    elif(e.signal == signals.INIT_SIGNAL):
      (_e, _state) = rr.init_stack(e) # search for META_INIT
      # if _state is a child of this state then transition to it
      if _state is None or not rr.has_a_child(p_p11_r2_region, _state):
        status = rr.trans(p_p11_s21)
      else:
        status = rr.trans(_state)
        if not _e is None:
          rr.post_fifo(_e)
  # ...

As of line 6 the following is true:

.. code-block:: python

   _state = p_p11_s22
   _e  = None

On line 6 the last layer of the onion is pulled.  The ``_state`` variable
contains ``p_p11_s22`` and the ``_e`` is set to None.  The logic to line 8 does
not apply to ``p_p11_s22``, so we call the ``trans`` method on line 11.

.. _recipes-e1-wtf-event:

E1 WTF Event
------------

The ``E1`` event is very much like the ``E0`` event in that it uses a ``META_INIT`` event to
pass over multiple boundaries.  It difference from the ``E0`` event in that the
``META_INIT`` needs to be sent the injector managing an inner orthogonal
component.  This injector is still part of the outer containing statechart.

.. image:: _static/xml_chart_4.svg
    :target: _static/xml_chart_4.pdf
    :align: center

If we start the chart in ``[['p_p11_s11', 'p_p11_s22'], 'p_s21']`` the post an
``E1`` event we will see the following logs:

.. code-block:: python
  :emphasize-lines: 50-56
  :linenos:

   S: [x] [p_r1] <- Queued:(0) Deferred:(0)
   S: [x] [p_r2] <- Queued:(0) Deferred:(0)
   S: [x] [p_p11_r1] <- Queued:(0) Deferred:(0)
   S: [x] [p_p11_r2] <- Queued:(0) Deferred:(0)
   S: [x] [p] E1
   S: [x] [p_r1] force_region_init:p_p11
   S: [x] [p_r1] force_region_init:p_r1_over_hidden_region
   S: [x] [p_p11_r1] region_exit:p_p11_s11
   S: [x] [p_p11_r1] region_exit:p_p11_r1_over_hidden_region
   S: [x] [p_p11_r1] region_exit:p_p11_r1_region
   S: [x] [p_p11_r1] EXIT_SIGNAL:p_p11_s11
   S: [x] [p_p11_r1] EXIT_SIGNAL:p_p11_r1_over_hidden_region
   S: [x] [p_p11_r1] EXIT_SIGNAL:p_p11_r1_region
   S: [x] [p_p11_r1] INIT_SIGNAL:p_p11_r1_under_hidden_region
   S: [x] [p_p11_r2] region_exit:p_p11_s22
   S: [x] [p_p11_r2] region_exit:p_p11_r2_over_hidden_region
   S: [x] [p_p11_r2] region_exit:p_p11_r2_region
   S: [x] [p_p11_r2] EXIT_SIGNAL:p_p11_s22
   S: [x] [p_p11_r2] EXIT_SIGNAL:p_p11_r2_over_hidden_region
   S: [x] [p_p11_r2] EXIT_SIGNAL:p_p11_r2_region
   S: [x] [p_p11_r2] INIT_SIGNAL:p_p11_r2_under_hidden_region
   S: [x] [p_r1] EXIT_SIGNAL:p_p11
   S: [x] [p_r1] EXIT_SIGNAL:p_r1_over_hidden_region
   S: [x] [p_r1] INIT_SIGNAL:p_r1_region
   S: [x] [p_r1] POST_FIFO:META_INIT
   S: [x] [p_r1] ENTRY_SIGNAL:p_r1_over_hidden_region
   S: [x] [p_p11] ENTRY_SIGNAL
   S: [x] [p_p11_r1] enter_region:p_p11_r1_under_hidden_region
   S: [x] [p_p11_r1] ENTRY_SIGNAL:p_p11_r1_region
   S: [x] [p_p11_r1] INIT_SIGNAL:p_p11_r1_region
   S: [x] [p_p11_r1] ENTRY_SIGNAL:p_p11_r1_over_hidden_region
   S: [x] [p_p11_r1] ENTRY_SIGNAL:p_p11_s12
   S: [x] [p_p11_r1] INIT_SIGNAL:p_p11_s12
   S: [x] [p_p11_r2] enter_region:p_p11_r2_under_hidden_region
   S: [x] [p_p11_r2] ENTRY_SIGNAL:p_p11_r2_region
   S: [x] [p_p11_r2] INIT_SIGNAL:p_p11_r2_region
   S: [x] [p_p11_r2] ENTRY_SIGNAL:p_p11_r2_over_hidden_region
   S: [x] [p_p11_r2] ENTRY_SIGNAL:p_p11_s21
   S: [x] [p_p11_r2] INIT_SIGNAL:p_p11_s21
   S: [x] [p_r1] ENTRY_SIGNAL:p_p11
   S: [x] [p_r1] INIT_SIGNAL:p_p11
   S: [x] [p_r2] force_region_init:p_s21
   S: [x] [p_r2] force_region_init:p_r2_over_hidden_region
   S: [x] [p_r2] EXIT_SIGNAL:p_s21
   S: [x] [p_r2] EXIT_SIGNAL:p_r2_over_hidden_region
   S: [x] [p_r2] INIT_SIGNAL:p_r2_region
   S: [x] [p_r2] ENTRY_SIGNAL:p_r2_over_hidden_region
   S: [x] [p_r2] ENTRY_SIGNAL:p_s21
   S: [x] [p_r2] INIT_SIGNAL:p_s21
   S: [x] E1:p
   S: [x]
   META_INIT <function p_p11 at 0x7fb8b27c88c8> ->
      META_INIT <function p_p11_r1_region at 0x7fb8b27c8ae8> ->
         META_INIT <function p_p11_s12 at 0x7fb8b27c8e18> ->
   S: [x] E1:p:HOOK
   S: [x] <- Queued:(0) Deferred:(0)
   R:
   [['p_p11_s11', 'p_p11_s22'], 'p_s21'] <- E1 == [['p_p11_s12', 'p_p11_s21'], 'p_s21']

The workings of the outer statechart are highlighted.  Despite, ``E1`` being
handled within the ``p`` region, the code needed to manage it is written in the ``p`` function which has access the ``XmlChart`` via the ``self`` keyword:

.. code-block:: python
  :emphasize-lines: 8-11
  :linenos:

   @spy_on
   def p(self, e):
     status = return_status.UNHANDLED
   # ..
     elif(self.token_match(e.signal_name, "E1")):
       if self.live_spy and self.instrumented:
         self.live_spy_callback("{}:p".format(e.signal_name))
       _e = self.meta_init(t=p_p11_s22, s=p, sig=e.signal_name)
       self.regions['p']._post_lifo(Event(signal=signals.force_region_init))
       self.regions['p'].post_fifo(_e)
       status = return_status.HANDLED

On line 8 the meta event is constructed, with a target equal to ``p_p11_s22``
and it's sources state set to ``p``.  The event name is passed through into the
method, though it is currently not used.

Line 9, pushes a ``force_region_init`` into the ``p`` region's orthogonal
component's queue, then on line 10, the meta event is placed and the events are
driven through the orthogonal component by the ``complete_circuit`` method
within the ``post_fifo`` call.

The ``force_region_init`` event will be on the queue before the ``META_INIT``
event.  The ``p`` region has two sub-regions, ``p_r1`` and ``p_r2``. The ``p_r1`` has
these state functions:

   * p_r1_under_hidden_region
   * p_r1_region
   * p_r1_over_hidden_region
   * p_p11 (injector)
   * p_p12 (injector)
   * p_r1_final

The ``p_r2`` has these state functions:

   * p_r2_under_hidden_region
   * p_r2_region
   * p_r2_over_hidden_region
   * p_r2_final
   * p_s21
   * p_p22 (injector)

The ``force_region_init`` will cause the ``p_r1`` orthogonal component to
transition to ``p_r1_region`` and the ``p_r2`` orthogonal component to
transition to ``p_r2_region``.  It does this so that the next event, the
``META_INIT`` waiting in the next spot of the queue will be seen by the
``INIT_SIGNAL`` clause of the ``p_r1_region`` and ``p_r2_region`` functions:

.. image:: _static/hidden_dynamics2.svg
    :target: _static/hidden_dynamics2.pdf
    :align: center

Both ``p_r1_region`` and ``p_r2_region`` will now be presented with this
``META_INIT`` event:

.. code-block:: python

  META_INIT <function p_p11 at 0x7fb8b27c88c8> ->
     META_INIT <function p_p11_r1_region at 0x7fb8b27c8ae8> ->
        META_INIT <function p_p11_s12 at 0x7fb8b27c8e18> ->

To see how to successfully trace the ``META_INIT`` event to its target read the :ref:`E0
recipe <recipes-e0-wtf-event>`.  In this case we will examine how the
``p_r2_region``, which is not a target of the ``META_INIT`` ditches the event
and behaves in accordance to its default INIT_SIGNAL behavior:

.. code-block:: python
  :linenos:

   @p_spy_on
   def p_r2_region(r, e):
     status = return_status.UNHANDLED
     # ...
     elif(e.signal == signals.INIT_SIGNAL):
       status = return_status.HANDLED
       (_e, _state) = r.init_stack(e) # search for META_INIT
       if _state is None or not r.has_a_child(p_r2_region, _state):
         status = r.trans(p_s21)
       else:
         status = r.trans(_state)
         #print("p_r2_region init {}".format(_state))
         if not _e is None:
           r.post_fifo(_e)
      # ...

On line 7:

.. code-block:: python

  _state = p_p11
  _e = META_INIT <function p_p11_r1_region at 0x7fb8b27c8ae8> ->
        META_INIT <function p_p11_s12 at 0x7fb8b27c8e18> ->

This will cause the ``not r.has_a_child(p_r2_region, _state)`` to return True,
causing a transition to the ``p_s21`` state.

E2 WTF Event
------------

The ``E2`` event is like the ``E1`` event in that it uses a ``META_INIT`` event
to pass over multiple orthogonal component boundaries.  It differs from the
``E1`` event in that its ``META_INIT`` needs to be sent from within an orthogonal
component and not from outer containing statechart.

.. image:: _static/xml_chart_4.svg
    :target: _static/xml_chart_4.pdf
    :align: center

.. code-block:: python
  :emphasize-lines: 10-12

   2020-03-28 13:29:39,690 DEBUG:S: [x] [p] E2
   2020-03-28 13:29:39,693 DEBUG:S: [x] [p_r1] E2:p_p11
   2020-03-28 13:29:39,694 DEBUG:S: [x] [p_r1] E2:p_r1_over_hidden_region
   2020-03-28 13:29:39,694 DEBUG:S: [x] [p_r1] E2:p_r1_region
   2020-03-28 13:29:39,695 DEBUG:S: [x] [p_r1] E2:p_r1_under_hidden_region
   2020-03-28 13:29:39,696 DEBUG:S: [x] [p_r2] E2:p_s21
   2020-03-28 13:29:39,697 DEBUG:S: [x] [p_r2] E2:p_r2_over_hidden_region
   2020-03-28 13:29:39,697 DEBUG:S: [x] [p_r2] E2:p_r2_region
   2020-03-28 13:29:39,698 DEBUG:S: [x] [p_r2] E2:p_r2_under_hidden_region
   2020-03-28 13:29:39,699 DEBUG:S: [x] E2:p
   2020-03-28 13:29:39,699 DEBUG:S: [x] E2:p:HOOK
   2020-03-28 13:29:39,700 DEBUG:S: [x] <- Queued:(0) Deferred:(0)
   2020-03-28 13:29:39,885 DEBUG:R:
   [['p_p11_s12', 'p_p11_s21'], 'p_s21'] <- E2 == \
   [['p_p11_s12', 'p_p11_s21'], 'p_s21']

To make this work, the ``E2`` must first be injected and driven through the
internal orthogonal components by the outer most injector (``p``):

.. code-block:: python
  :emphasize-lines: 12
  :linenos:

   @spy_on
   def p(self, e):
     # ..

     # any event handled within there regions must be pushed from here
     elif(type(self.regions) == dict and (self.token_match(e.signal_name, "e1") or
         self.token_match(e.signal_name, "e2") or
         self.token_match(e.signal_name, "e3") or
         self.token_match(e.signal_name, "e4") or
         self.token_match(e.signal_name, "e5") or
         self.token_match(e.signal_name, "C0") or
         self.token_match(e.signal_name, "E2") or
         # self.token_match(e.signal_name, "G3") or
         self.token_match(e.signal_name, self.regions['p_p11'].final_signal_name) or
         self.token_match(e.signal_name, self.regions['p_p12'].final_signal_name) or
         self.token_match(e.signal_name, self.regions['p_p22'].final_signal_name)
         )):
       if self.live_spy and self.instrumented:
         self.live_spy_callback("{}:p".format(e.signal_name))
       self.regions['p'].post_fifo(e)
       status = return_status.HANDLED

The construction of its META_INIT event occurs within the ``p_p12`` handler:

.. code-block:: python
  :emphasize-lines: 14-17
  :linenos:

   @p_spy_on
   def p_p12(r, e):
     status = return_status.UNHANDLED
     outmost = r.outmost
     (post_fifo,
      _post_fifo,
      post_lifo,
      _post_lifo,
      token_match,
      scribble) = outmost_region_functions(r, 'p_p12')
     # ..
     elif outmost.token_match(e.signal_name, "E2"):
       scribble(e.signal_name)
       _e = outmost.meta_init(t=p_p12_p11_s12, s=p_p12, sig=e.signal_name)
       # this force_region_init might be a problem
       _post_lifo(Event(signal=signals.force_region_init))
       post_fifo(_e)
       status = return_status.HANDLED
     # ..

On line 14 we create a META_INIT as a reaction to the  ``E2`` event.  To build
such an ``META_INIT`` we need to specify the target, ``t``, the source ``s`` and
the event's signals name (E2).  The resulting meta event is returned as ``_e``.

On line 10 the first location of each queue of the orthogonal regions of
``p_p12`` have the ``force_region_init`` event posted to their far left location.  On
line 11, the ``_e`` meta event is placed the right of each
``force_region_init`` event for each queue in the ``p_p12`` region, then all
events are pushed through those machines.

To make the ``E2`` event work for the entire chart, a handler needs to be added
to ``p_p22``:

.. code-block:: python
  :emphasize-lines: 10
  :linenos:

   @p_spy_on
   def p_p22(r, e):
     status = return_status.UNHANDLED
     outmost = r.outmost
     (post_fifo,
      _post_fifo,
      post_lifo,
      _post_lifo,
      token_match,
      scribble) = outmost_region_functions(r, 'p_p22')
     # ..
     # any event handled within there regions must be pushed from here
     elif(token_match(e.signal_name, "e1") or
          token_match(e.signal_name, "e2") or
          token_match(e.signal_name, "e4") or
          token_match(e.signal_name, "E2")
         ):
       if outmost.live_spy and outmost.instrumented:
         outmost.live_spy_callback("{}:p_p22".format(e.signal_name))
       outmost.regions['p_p22'].post_fifo(e)
       status = return_status.HANDLED

If ``E2`` is not permitted to be driven through the ``p_p22`` the statechart
doesn't work properly.

.. _recipes-c-events:

C WTF Events
------------
``C`` events cause a transition from one orthogonal region to another.  The do
not span boundaries:

.. image:: _static/xml_chart_4.svg
    :target: _static/xml_chart_4.pdf
    :align: center

.. _recipes-c0-wtf-event:

C0 WTF Event
------------
The ``C0`` event occurs within one or more orthogonal regions.   The basic
pattern involves the outer states letting the ``C0`` enter into its required
depth, then a region's injector captures the event and uses the miros ``trans`` call
to cause the transition.  It does not require META events to manage its
transition across boundaries and is thereby the simplest ``WTF`` event.

.. image:: _static/xml_chart_4.svg
    :target: _static/xml_chart_4.pdf
    :align: center

Suppose the chart was in a ``[['p_p11_s12', 'p_p11_s21'], 'p_s21']`` combination
of states and a ``C0`` was sent to the chart.  The log file would look like
this:

.. code-block:: python
  :emphasize-lines: 63-65

   S: [x] [p] C0
   S: [x] [p_r1] C0:p_p11
   S: [x] [p_p11_r1] region_exit:p_p11_s12
   S: [x] [p_p11_r1] region_exit:p_p11_r1_over_hidden_region
   S: [x] [p_p11_r1] region_exit:p_p11_r1_region
   S: [x] [p_p11_r1] EXIT_SIGNAL:p_p11_s12
   S: [x] [p_p11_r1] EXIT_SIGNAL:p_p11_r1_over_hidden_region
   S: [x] [p_p11_r1] EXIT_SIGNAL:p_p11_r1_region
   S: [x] [p_p11_r1] INIT_SIGNAL:p_p11_r1_under_hidden_region
   S: [x] [p_p11_r2] region_exit:p_p11_s21
   S: [x] [p_p11_r2] region_exit:p_p11_r2_over_hidden_region
   S: [x] [p_p11_r2] region_exit:p_p11_r2_region
   S: [x] [p_p11_r2] EXIT_SIGNAL:p_p11_s21
   S: [x] [p_p11_r2] EXIT_SIGNAL:p_p11_r2_over_hidden_region
   S: [x] [p_p11_r2] EXIT_SIGNAL:p_p11_r2_region
   S: [x] [p_p11_r2] INIT_SIGNAL:p_p11_r2_under_hidden_region
   S: [x] [p_r1] EXIT_SIGNAL:p_p11
   S: [x] [p_p12] ENTRY_SIGNAL
   S: [x] [p_p12_r1] enter_region:p_p12_r1_under_hidden_region
   S: [x] [p_p12_r1] ENTRY_SIGNAL:p_p12_r1_region
   S: [x] [p_p12_r1] INIT_SIGNAL:p_p12_r1_region
   S: [x] [p_p12_r1] ENTRY_SIGNAL:p_p12_r1_over_hidden_region
   S: [x] [p_p12_p11] ENTRY_SIGNAL
   S: [x] [p_p12_p11_r1] enter_region:p_p12_p11_r1_under_hidden_region
   S: [x] [p_p12_p11_r1] ENTRY_SIGNAL:p_p12_p11_r1_region
   S: [x] [p_p12_p11_r1] INIT_SIGNAL:p_p12_p11_r1_region
   S: [x] [p_p12_p11_r1] ENTRY_SIGNAL:p_p12_p11_r1_over_hidden_region
   S: [x] [p_p12_p11_r1] ENTRY_SIGNAL:p_p12_p11_s11
   S: [x] [p_p12_p11_r1] INIT_SIGNAL:p_p12_p11_s11
   S: [x] [p_p12_p11_r2] enter_region:p_p12_p11_r2_under_hidden_region
   S: [x] [p_p12_p11_r2] ENTRY_SIGNAL:p_p12_p11_r2_region
   S: [x] [p_p12_p11_r2] INIT_SIGNAL:p_p12_p11_r2_region
   S: [x] [p_p12_p11_r2] ENTRY_SIGNAL:p_p12_p11_r2_over_hidden_region
   S: [x] [p_p12_p11_r2] ENTRY_SIGNAL:p_p12_p11_s21
   S: [x] [p_p12_p11_r2] INIT_SIGNAL:p_p12_p11_s21
   S: [x] [p_p12_r1] ENTRY_SIGNAL:p_p12_p11
   S: [x] [p_p12_r1] INIT_SIGNAL:p_p12_p11
   S: [x] [p_p12_r2] enter_region:p_p12_r2_under_hidden_region
   S: [x] [p_p12_r2] ENTRY_SIGNAL:p_p12_r2_region
   S: [x] [p_p12_r2] INIT_SIGNAL:p_p12_r2_region
   S: [x] [p_p12_r2] ENTRY_SIGNAL:p_p12_r2_over_hidden_region
   S: [x] [p_p12_r2] ENTRY_SIGNAL:p_p12_s21
   S: [x] [p_p12_r2] INIT_SIGNAL:p_p12_s21
   S: [x] [p_r1] ENTRY_SIGNAL:p_p12
   S: [x] [p_r1] INIT_SIGNAL:p_p12
   S: [x] [p_r2] C0:p_s21
   S: [x] [p_r2] EXIT_SIGNAL:p_s21
   S: [x] [p_p22] ENTRY_SIGNAL
   S: [x] [p_p22_r1] enter_region:p_p22_r1_under_hidden_region
   S: [x] [p_p22_r1] ENTRY_SIGNAL:p_p22_r1_region
   S: [x] [p_p22_r1] INIT_SIGNAL:p_p22_r1_region
   S: [x] [p_p22_r1] ENTRY_SIGNAL:p_p22_r1_over_hidden_region
   S: [x] [p_p22_r1] ENTRY_SIGNAL:p_p22_s11
   S: [x] [p_p22_r1] INIT_SIGNAL:p_p22_s11
   S: [x] [p_p22_r2] enter_region:p_p22_r2_under_hidden_region
   S: [x] [p_p22_r2] ENTRY_SIGNAL:p_p22_r2_region
   S: [x] [p_p22_r2] INIT_SIGNAL:p_p22_r2_region
   S: [x] [p_p22_r2] ENTRY_SIGNAL:p_p22_r2_over_hidden_region
   S: [x] [p_p22_r2] ENTRY_SIGNAL:p_p22_s21
   S: [x] [p_p22_r2] INIT_SIGNAL:p_p22_s21
   S: [x] [p_r2] ENTRY_SIGNAL:p_p22
   S: [x] [p_r2] INIT_SIGNAL:p_p22
   S: [x] C0:p
   S: [x] C0:p:HOOK
   S: [x] <- Queued:(0) Deferred:(0)
   R: [['p_p11_s12', 'p_p11_s21'], 'p_s21'] <- C0 == \
   [[['p_p12_p11_s11', 'p_p12_p11_s21'], 'p_p12_s21'], ['p_p22_s11', 'p_p22_s21']]

The highlighted parts of the log cause the upper part of the log to take place.


.. _recipes-hidden-dynamics:

Hidden Dynamics
^^^^^^^^^^^^^^^

.. raw:: html

  <a class="reference internal" href="quickstart.html"<span class="std-ref">prev</span></a>, <a class="reference internal" href="index.html#top"><span class="std std-ref">top</span></a>, <a class="reference internal" href="introduction.html"><span class="std std-ref">next</span></a>
