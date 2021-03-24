A partial implimentation of the SCXML standard for building working concurrent statecharts from XML documents.

[In depth engineering documents for miros-xml.](https://aleph2c.github.io/miros-xml/html/)

# miros-scxml WIP

  > This library is a partial-port of the [SCXML standard](https://www.w3.org/TR/scxml/) into [miros](https://github.com/aleph2c/miros).

**Usage (Python >= 3.5)**:
```python
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
```
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

The SCXML standard serves two different objectives:  

1. To describe statecharts (Harel topological diagrams) with XML. (it should just be this)

2. To solve specific industrial problems related to VoiceXML and CCXML.

4. ``pass`` is a keyword in Python, so use another state or variable name when conducting your tests. (``_pass``)

I will implement the statechart related parts, and ignore the rest of the standard.  All generalized programming will be done with Python (using the datamodel and data tags).

If I can, I will raise the required errors so that this port can remain standards compliant.  If this turns out to be too much effort for too little value, I will break away from the standard entirely.

List of exceptions to the standard so far:

* The initial element is supported in an atomic state if it is carrying code and not causing a state transition.  It can be useful to have code run in a state's initialization.

## List of extensions to the SCXML standard

* A new ``<debug/>`` element has been added which will create a file and add an ``import pdb; pdb.set_trace()`` expression in the location miros code corresponding to where the tag was placed in the statechart.  This was added to make it easy for users to debug their statecharts.
* A ``python`` datamodel was added, so that python code can be placed in the statechart.
* One-shots can be implemented in multiple ways:

```xml
<!-- FIFO -->
<send event="timeout.token1.token2 delay="1s"/>

<!-- or as -->
<send eventexpr="post_fifo(Event(signal='timeout.token1.token2'))" delay="1s"/>

<!-- or as -->
<send eventexpr="post_fifo(Event(signal='timeout.token1.token2'))" delayexpr="times=1, period=1.0, deferred=True" />
<!-- or as -->
<send eventexpr="post_fifo(Event(signal='timeout.token1.token2'))" delayexpr="times=1, delay=1.0, deferred=True" />

<!-- LIFO -->
<send eventexpr="post_lifo(Event(signal='timeout.token1.token2'))" delay="1s"/>

<!-- or as -->
<send eventexpr="post_lifo(Event(signal='timeout.token1.token2'))" delayexpr="times=1, period=1.0, deferred=True"/>
```

Sending a payload seems very hard in SCXML
```xml
<!-- trying to write:  
  pickled_function = dill.load(open("build_payload.p", "rb"))
-->
<datamodel>
  <data expr="dill.load(open(&quote;build_payload.p&quote;, &quote;rb&quote;))", id="pickled_function"/>
</datamodel>

<!-- trying to write: 
  post_fifo(Event(signal=signals.evidence, payload=pickled_function())) 
-->
<send event="evidence">
  <param name="payload" expr="pickled_function()"/>
</send>

```


* Multi-shots can be implemented in multiple ways:

```xml
<!-- FIFO -->
<send eventexpr="post_fifo(Event(signal='timeout.token1.token2'))" delayexpr="times=3, period=1.0, deferred=True" />

<!-- LIFO -->
<send eventexpr="post_lifo(Event(signal='timeout.token1.token2'))" delayexpr="times=3, period=1.0, deferred=True" />
```

* To create a one/multi-shot then cancel it at a later time:
```xml
<!-- FIFO -->
<send  event="timeout.token1.token2" delay="1s" id="ef120"  />
<!-- later, to cancel -->
<cancel sendid="ef120"/>

<!-- or as -->
<send eventexpr="post_fifo(Event(signal='timeout.token1.token2'))" id="ef120" delay="1s" />
<!-- later, to cancel -->
<cancel sendid="ef120"  />

<!-- or as -->
<send eventexpr="post_fifo(Event(signal='timeout.token1.token2'))" id="ef120" delayexpr="times=1, period=1.0, deferred=True" />
<cancel sendid="ef120" />

<!-- or to cancel all one/multi-shots with a specific token -->
<send eventexpr="post_lifo(Event(signal='timeout.banana.apple'))" delayexpr="times=1, period=1.0, deferred=True" />
<cancel sendexpr="cancel_all(Event(signal='timeout'))""  />
<!-- or cancel with -->
<cancel sendexpr="cancel_all(Event(signal='apple'))"" />
<!-- or cancel with -->
<cancel sendexpr="cancel_all(Event(signal='banana'))"" />
```
---

  > The use of the **eventexpr** and **delayexpr** in this way may or may not be extending or breaking the standard.  I can not tell from reading the documents.

---

## Means through which the Standard was met

* The ``SCXML_INIT_SIGNAL`` internal signal was added in this library.  It was added so that eventless transitions between states can occur (see ``./data/scxml_test_1.scxml`` for an example)
* The contents of a ``<log/>`` element are transformed into ``scribble`` calls to the miros spy instrumentation stream.

## Recognition

Thank you to the following people:

* [Miro Samek](https://www.linkedin.com/in/samek) for publishing his event processing algorithm
* [Alex Zhornyak](https://github.com/alexzhornyak) for writing the [SCXML tutorial](https://github.com/alexzhornyak/SCXML-tutorial)
* [Jim Barnett](https://www.speechtechmag.com/Articles/Editorial/Feature/The-2014-Speech-Luminaries-98322.aspx) for writing the [SCXML compliance tests](https://www.w3.org/Voice/2013/scxml-irp/#tests)
