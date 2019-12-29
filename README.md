# miros-scxml WIP

  > This library is a port of the [SCXML standard](https://www.w3.org/TR/scxml/) into [miros](https://github.com/aleph2c/miros).

----

## List of Unsupported Tests

* [576](https://www.w3.org/Voice/2013/scxml-irp/576/test576.txml) (parallel region init test)
* [364](https://www.w3.org/Voice/2013/scxml-irp/364/test364.txml) (default init test with parallel region)

  
## List of exceptions to the SCXML Standard

* The initial element is supported in an atomic state if it is is carrying code and not causing a state transition.  It can be useful to have code run in a state's initialization.

## List of extensions to the SCXML standard

* A ``<debug>`` element has been added which will create a file and add an ``import pdb; pdb.set_trace()`` expression in the location where the tag was placed in the statechart.  This was added to make it easy for users to debug their statecharts.
* The ``python`` datamodel was added, so that python code can be placed in the statechart.


## Means through which the Standard was met

* The ``SCXML_INIT_SIGNAL`` internal signal is added in this library.  It was added so that eventless transitions between states can occur (see ``./data/scxml\_test\_1.scxml`` for an example)
* The contents of a log element are transformed into scribble calls to the miros spy instrumentation stream.

## Recognition

Thank you to the following people:

* [Miro Samek](https://www.linkedin.com/in/samek) for publishing his event processing algorithm
* [Alex Zhornyak](https://github.com/alexzhornyak) for writing the [SCXML tutorial](https://github.com/alexzhornyak/SCXML-tutorial)
