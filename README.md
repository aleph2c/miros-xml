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

  
## List of exceptions to the SCXML Standard

* The initial element is supported in an atomic state if it is carrying code and not causing a state transition.  It can be useful to have code run in a state's initialization.

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
