# miros-scxml WIP

  > This library is a port of the [SCXML standard](https://www.w3.org/TR/scxml/) into [miros](https://github.com/aleph2c/miros).

**Usage (Python >= 3.5)**:

        from pathlib import Path
        from miros_scxml.xml_to_miros import XmlToMiros
    
        # path to scxml file
        data_dir = Path(".") / ".." / "data" /
        scxml_path = data_dir / "scxml_test_1.scxml"
   
        # Make an active object and start it
        xml_to_chart = XmlToMiros(scxml_path)
        ao = xml_to_chart.make()
        ao.live_spy = True
        ao.start()  # will generate spy log: data_dir / scxml_test_1.log

----

## List of Unsupported Tests

* [576](https://www.w3.org/Voice/2013/scxml-irp/576/test576.txml) (parallel region init test)
* [364](https://www.w3.org/Voice/2013/scxml-irp/364/test364.txml) (default init test with parallel region)
* [403b](https://www.w3.org/Voice/2013/scxml-irp/403/test403b.txml) (we test that 'optimally enabled set' really is a set, specifically that if a transition is optimally enabled in
  two different states, it is taken only once - parallel regions)
* [403c](https://www.w3.org/Voice/2013/scxml-irp/403/test403c.txml) (we test 'optimally enabled set', specifically that preemption works correctly - parallel regions)
* [404](https://www.w3.org/Voice/2013/scxml-irp/404/test404.txml) (test that states are exited in exit order (children before parents with reverse doc order used to break ties before the executable content in the transitions.  event1, event2, event3, event4 should be raised in that order when s01p is exited - parallel region)
* [405](https://www.w3.org/Voice/2013/scxml-irp/405/test405.txml) (test that the executable content in the transitions is executed in document order after the states are exited. event1, event2, event3, event4 should be raised in that order when the state machine is entered - parallel regions)
* [406](https://www.w3.org/Voice/2013/scxml-irp/406/test406.txml) (Test that states are entered in entry order (parents before children with document order used to break ties) after the executable content in the transition is executed. event1, event2, event3, event4 should be raised in that order when the transition in s01 is taken - parallel regions)
  
## List of exceptions to the SCXML Standard

* The initial element is supported in an atomic state if it is carrying code and not causing a state transition.  It can be useful to have code run in a state's initialization.
* One-shots can be implemented in multiple ways:

      <!-- FIFO -->
      <send>event="timeout.token1.token2 delay="1s"</send>
      <!-- or as -->
      <send>eventexpr="post_fifo(timeout.token1.token2)" delay="1s"</send>
      <!-- or as -->
      <send>eventexpr="post_fifo(timeout.token1.token2)" delayexpr="times=1, delay=1.0, deferred=True"</send>

      <!-- LIFO -->
      <send>eventexpr="post_lifo(timeout.token1.token2)" delay="1s"</send>
      <!-- or as -->
      <send>eventexpr="post_lifo(timeout.token1.token2)" delayexpr="times=1, delay=1.0, deferred=True"</send>

* Multi-shots can be implemented in multiple ways:

      <!-- FIFO -->
      <send>eventexpr="post_fifo(timeout.token1.token2)" delayexpr="times=3, delay=1.0, deferred=True"</send>

      <!-- LIFO -->
      <send>eventexpr="post_lifo(timeout.token1.token2)" delayexpr="times=3, delay=1.0, deferred=True"</send>

## List of extensions to the SCXML standard

* A new ``<debug/>`` element has been added which will create a file and add an ``import pdb; pdb.set_trace()`` expression in the location miros code corresponding to where the tag was placed in the statechart.  This was added to make it easy for users to debug their statecharts.
* A ``python`` datamodel was added, so that python code can be placed in the statechart.

## Means through which the Standard was met

* The ``SCXML_INIT_SIGNAL`` internal signal was added in this library.  It was added so that eventless transitions between states can occur (see ``./data/scxml_test_1.scxml`` for an example)
* The contents of a ``<log/>`` element are transformed into ``scribble`` calls to the miros spy instrumentation stream.

## Recognition

Thank you to the following people:

* [Miro Samek](https://www.linkedin.com/in/samek) for publishing his event processing algorithm
* [Alex Zhornyak](https://github.com/alexzhornyak) for writing the [SCXML tutorial](https://github.com/alexzhornyak/SCXML-tutorial)
