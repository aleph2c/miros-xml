
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

This is me writing to think, and to try and find a simpler way to implement the
parallel tag.

I would like miros-xml to support parallel statecharts using David Harel's
notation.  The Parallel statechart, or orthogonal regions are not supported by
the algorithm developed by Miro Samek.  However, Miro Samek offered an
alternative and faster pattern, the `orthogonal component
<https://aleph2c.github.io/miros/patterns.html#patterns-orthogonal-component>`_;
an HSM within an HSM.  The problem with this orthogonal component pattern is
that it isn't as graphically elegant as the one envisioned by David Harel.
David Harel's graphical abstractions permit more than one active state to
exist within one diagram at a time; the packing of design complexity into a
small diagram's space is very efficient.

If you haven't seen orthogonal regions before reference `this document on
orthogonal regions and miros.
<https://aleph2c.github.io/miros/othogonalregions.html>`_.

In this project and in these documents I will be writing about how to have one
orthogonal region communicate with another.  My implementation of orthogonal
regions will be a mapping of Miros Samek's orthogonal component pattern (HSMs
within other HSMs) onto a design which will look and behave like an orthogonal
region.  This will be done by recursively mapping orthogonal components within a
chart.  An outer region will ``dispatch`` events into a deeper or inner region.
The outer region which injects an event will be called an **Injector** and the
region which receives the events will be called an **Injectee**.  An HSM can both
be an **Injector** and an **Injectee**, and the outermost chart will only be an
**Injector**.

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
         ``HsmWithQueues`` object with additional supporting methods, in the software
         it is represented as a class.  In the design it is an orthogonal
         component called the **injectee**.  In statechart theory, the orthogonal
         region is one of the areas partitioned within a dashed line.  It and
         it's other regions are expected to run concurrently.

         .. image:: _static/region.svg
             :target: _static/region.pdf
             :align: center


    Regions
         Contains multiple ``Region`` objects in a collection.  It augments the
         regions so that they can reference one another using iterators.  It
         adds a ``_post_fifo`` and ``_post_lifo`` method which can put items
         into all of its inner region's queues.

         It adds a ``post_fifo`` and ``post_lifo`` method which will post items
         onto an inner queue, then drive the inner statecharts using their
         ``complete_circuit`` method until their queues are empty.

         .. image:: _static/regions.svg
             :target: _static/regions.pdf
             :align: center

         All other inner state methods, connected to a particular regions HSM
         will have an ``@p_spy_on`` decorator.

    ScxmlChart
         Is the main statechart into which all events are posted.  It is the
         outer most injector.

         .. image:: _static/ScxmlChart.svg
             :target: _static/ScxmlChart.pdf
             :align: center

         The ScxmlChart will have a dict containing multiple Regions.  The
         ScxmlChart's thread will drive the entire collection of orthgonal
         regions.  It will inject all of the events, via the regions dictionary.

         All of the ScxmlChart connected state methods will have a ``@spy_on``
         decorator.  All other inner state methods will have an ``@p_spy_on``
         decorator.

    Injector:
         A statechart or orthogonal component which posts events into an inner
         orthogonal component, then drives the events through the region it is
         encapsulated.  The **injector** places events and drives events using
         the ``_post_fifo', ``post_fifo`` ... APIs.  The **p** and **pp12**
         states would be an injector in the following diagram:

         .. image:: _static/hidden_dynamics.svg
             :target: _static/hidden_dynamics.pdf
             :align: center

    Injectee:
         An orthogonal component who's events are given to it, and driven through
         it by an **injector**.  An **injectee** can also be an **injector** if
         it drives another state.  An injector is hidden from view in the main
         statechart picture with the dashed lines.  However, you can see it at
         the bottom right of this picture.


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
         region can be exited.  It's a holding state with no initializtion
         handler, and it is the outer most state of the region.

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
         An event which permits an WTF exit strategy

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
  :emphasize-lines: 23,24
  :linenos:

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

You can see on lines 23 and 24 I have filtered out any spy line with the names
``START``, ``SEARCH_FOR_SUPER``, ``region_exit``, and ``*hidden_region``.  This
was to reduce the amount of noise in the instrumentation.

The spy itself is written to a log file and/or written to the terminal.

.. _recipes-suped-up-state-name-reflection:

Souped Up State Name Reflection
-------------------------------

If you use the vanilla ``state_name`` method provided within miros you will only
be able to see the outer most state holding the orthogonal regions; but it will
not reach into this collection of orthogonal regions and report on the active state
of each of them.

To see all of the active states at once using the ``active_states`` method of
the ``ScxmlChart`` class.

.. image:: _static/xml_chart_4.svg
    :target: _static/xml_chart_4.pdf
    :align: center

.. code-block:: python
  :emphasize-lines: 15
  :linenos:

  example = ScxmlChart(
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

To code required to make ``active_states`` is within the ``ScxmlChart`` class:

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

4. Add the region to the ScxmlChart's regions dict within the ScxmlChart
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

.. _recipes-e0-wtf-event:

E0 WTF Event
------------
The E0 event occurs from the outer most threaded state chart and it passes over
multiple regional boundaries.

.. image:: _static/xml_chart_4.svg
    :target: _static/xml_chart_4.pdf
    :align: center

As it passes a boundary it turns that boundary on, by issuing an INIT_SIGNAL.
If the INIT_SIGNAL handling code finds INIT_META events waiting in that region's
deque, it will pull out its META_SIGNAL_PAYLOAD, extract the state and event
information, then transition to the state and post then next event into the
deque.  In this way the deque acts like a very-lightweight program stack.

.. note::

   The downside of this technique is that the source of the ``E0`` event needs to
   know about the structure of the chart, the algorithm can not figure it out on
   the fly like it would if it were using the miros algorithim.

   This could be addressed by having the ``E0`` events or other ``E`` event be
   structured programmatically with an autocacher wrapping the method.

The source state needs to know a lot of information about the statechart for the
``E0`` signal to work.  The ``outer_state`` in the picture above would look like
this in code:

.. code-block:: python
  :emphasize-lines: 10-34
  :linenos:

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
           payload=META_SIGNAL_PAYLOAD(
             event=eeee, state="p_p11_s21", source_event=e, region=None)
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
       else:
         self.temp.fun = self.bottom
         status = return_status.SUPER
       return status

On line 10 we see the ``E0`` state handler in the ``outer_state``.  Lines 11-12
report on its discovery to the spy scribble. Line 13-32 show how to construct a
set of recursive META_INIT signals.  The bottom most is intended from an
injector, then next and injectee and so on and so forth.  Line 33 shows how it
is inserted into the FIFO for an init hack.  The entry handler of ``p`` will pull
the META_EVENT out of the deque, then push its inner META_INIT event into the
``p`` region.  Line 34 causes a transition to ``p``, so that this
event can be dealt with once ``p`` has initialized.

Here is part of the ``p`` state handler:

.. code-block:: python
  :emphasize-lines: 5-12
  :linenos:

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
  # ..

The entry signal occurs due to the transition made by ``E0`` into the ``p``
state.  On lines 6 to 7 we see that this is being reported to the spy scribble.

Line 8 shows the use of the ``init_stack`` method, which pulls META_INIT signals
out of the statechart's deque if they are there.  Lines 9-10, show that if a
state was found (A META_INIT event was present) post the next META_INIT signal
into the injectee state managed by this injector (``p``).

Finally, on line 11 the injectee region is entered by injecting the
``enter_region`` event into this orthogonal component.  This ``enter_region``
event will cause a transition from the ``p_r1_under_hidden_region`` into the
``p_r1_region`` state and the ``p_r2_under_hidden_region`` into the ``p_r2_region``
state. Once the ``p_r1_region`` and ``p_r2_region`` have been entered, the event
processor will issue an ``INIT_SIGNAL``.

To see how the META_INIT event has its information passed into the next region
we will look ``p_r1_region``:

.. code-block:: python
  :emphasize-lines: 6-15
  :linenos:

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
  # ..

The ``INIT_SIGNAL`` handler for this state function has been adjusted to read
the deque and from it pull a state and an event, if a META_INIT event was placed
there.  This is done within the ``init_stack`` method.

The default behavior for this state is seen on line 11, and the dynamics
behavior to support the transport of a ``WTF`` event deeper into the chart is
seen on lines 13-14.  Line 13 causes a transition to a dynamic state, feed by
the original injectee of the ``E0`` event.  The next META_INIT event is posted
to the next injector on line 15.

.. _recipes-hidden-dynamics:

Hidden Dynamics
^^^^^^^^^^^^^^^


.. raw:: html

  <a class="reference internal" href="quickstart.html"<span class="std-ref">prev</span></a>, <a class="reference internal" href="index.html#top"><span class="std std-ref">top</span></a>, <a class="reference internal" href="introduction.html"><span class="std std-ref">next</span></a>
