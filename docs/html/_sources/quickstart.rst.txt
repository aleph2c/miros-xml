.. _quickstart-quick-start:

Getting Your Head in the Project
================================

Here is how to pick up this project after leaving it for a long time.

.. contents::
  :depth: 2
  :local: 
  :backlinks: none

.. _quickstart-baselining:

Baselining
^^^^^^^^^^

.. code-block:: bash

 # navigate to the miros-xml repository
 # load the virtual environment
 . ./venv/bin/activate

 # load up a previously defined tmux environment
 tmuxp load tmux.yaml

Now run the tests:

.. code-block:: bash

  pytest -s

If the tests aren't passing, fix the problems.

.. _quickstart-finding-the-plan:

Finding the Plan
^^^^^^^^^^^^^^^^

The project plan is a recursive set of OODA loops in the form of a Vim wiki.
Its top level dispatch file ``plan/top.wiki``.

.. _quickstart-generating-the-docs:

Finding and Generating the Docs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tmux environment will automatically construct your ``html`` or ``pdf/svg`` pictures:

.. code-block:: bash

  cd docs
  while inotifywait -e modify ./source/*.rst ./source/_static/*.uxf; do make clean && make html; done

The pictures are drawn using `Umlet <https://www.umlet.com/>`_ and are saved as
``.uxf`` files in the ``docs/source/_static`` directory.  The text is written as
a set of ``.rst`` files in the ``docs/source`` directory.  The command ``make
clean && make html`` causes the `sphinx <https://www.sphinx-doc.org/en/master/>`_ make file to delete the old ``html``,
render new ``html`` from the ``.rst`` and create the ``.pdf`` and ``.svg``
drawings by calling the Umlet tool's command line interface.

The ``while inotifywait -e modify ./source/*.rst ./source/_static/*.uxf``
watches the two directories and runs the ``make clean && make html`` command if
the text or pictures have been changed on disk;  It automatically creates the
docs.  To view them open ``<repo-directory>/docs/html/index.html`` in your web
browser.

.. _quickstart-understanding-how-it-works:

Getting Something Running
^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

  The tmux environment should open the latest experiment in Vim, tail its logs and
  run an ``inotifywait`` service which will run the program if its source file or its logging
  configuration file has changed.

This project is in the experiment phase, the majority of the production code is
in the ``experiment`` folder.  The thing to
look for is the latest ``xml_chart_<num>.py`` file.  The file contains its own
test in the ``if __name__ == '__main__'`` code block at the bottom of the file.
To work on a given experiment, look to the top of its file, you will see
instructions on how to have it automatically test itself if you change it.  For
instance the ``xml_chart_5.1.py`` file tells us to test it with:

.. code-block:: bash

  while inotifywait -e modify xml_chart_5.1.py logger_config.yaml; do python xml_chart_5.1.py; done

This will contain the latest experiment.  To understand to architecture, read the
:ref:`how it works <how_it_works>` section.

.. _quickstart-pushing-past-latest-stage-of-the-experiment:

Pushing Past the Latest Stage of the Experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To safely re-engage with your project create a git branch with the date name in it:

.. code-block:: python

  echo "$(date '+re-engaged_%m/%d/%y')" | xargs git checkout -b
  Switching to a new branch 're-engaged_01/01/21'

Go to the latest experiment, then open the :ref:`beastiary <how_it_works-beastiary>`.

.. _quickstart-find-the-map:

Find the Map
^^^^^^^^^^^^
The diagrams can be found in ``experiment``.  To work without a diagram is
folly, if you don't have one, create one.  Currently you are working on
``xml_chart_5.pdf``

.. _quickstart-investigating-a-wtf-event:

Investigating a WTF event
^^^^^^^^^^^^^^^^^^^^^^^^^

To trace a WTF event across its orthogonal component boundaries, label it as
``event_to_investigate`` and the top of the file, run your test, then look to
the cli output or log file.  For instance, I will investigate the ``SRE3``
event, starting from the ``['middle']`` state.

.. image:: _static/xml_chart_5.svg
    :target: _static/xml_chart_5.pdf
    :class: noscale-center

From the diagram and given ``SRE3 -> ['middle']``, I would expect to see:

* middle exit
* middle entry
* p entry
* p init
* p_p11 entry
* p_p11 init
* p_p11_s11 entry
* p_p11_s11 init
* p_p11_s21 entry
* p_p11_s21 init
* p_p22 entry
* p_p22 init
* p_p_p22_s11 entry
* p_p_p22_s11 init
* p_p_p22_s21 entry
* p_p_p22_s21 init

The log files are almost unusable considering things are out of order.  The
trace looks like this:

.. code-block:: bash
  :emphasize-lines: 1
  :linenos:

    [07] R: --- [['p_p11_s11', 'p_p11_s21'], 'p_s21'] <- SRH2 == ['middle']
    [07] S: 1: ['p_r2_under_hidden_region']
    [07] S: [x] SRE3:outer
    [07] S: 1: SRE3
    [07] S: 1: [n=1]::BOUNCE_SAME_META_SIGNAL:outer [n=0]::SRE3:outer ->
      [n=2]::INIT_META_SIGNAL:p_r2_region [n=1]::BOUNCE_SAME_META_SIGNAL:p_r2_region ->
        [n=3]::INIT_META_SIGNAL:p_p22 [n=2]::INIT_META_SIGNAL:p_r2_region ->

    [07] S: 1: >>>> outer 3696 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    ---------------------------------------
    p_r1:p_r1_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_r2:p_r2_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p11_r1:p_p11_r1_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p11_r2:p_p11_r2_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p12_r1:p_p12_r1_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p12_r2:p_p12_r2_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p12_p11_r1:p_p12_p11_r1_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p12_p11_r2:p_p12_p11_r2_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p22_r1:p_p22_r1_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p22_r2:p_p22_r2_under_hidden_region, ql=0: {}
    ---------------------------------------
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    [07] S: [p_r1_region] 3: ['p_r1_region', 'p_s21']
    [07] S: [x] EXIT_SIGNAL:middle
    [07] S: [p_r1_region] 3: INIT_SIGNAL
    [07] S: [x] ENTRY_SIGNAL:middle
    [07] S: [p_r1_region] 3: [n=3]::INIT_META_SIGNAL:p_p22 [n=2]::INIT_META_SIGNAL:p_r2_region ->

    [07] S: [x] ENTRY_SIGNAL:p
    [07] S: [x] ENTRY_SIGNAL:p_p11
    [07] S: [x] ENTRY_SIGNAL:p_p11_s11
    [07] S: [x] INIT_SIGNAL:p_p11_s11
    [07] S: [x] ENTRY_SIGNAL:p_p11_s21
    [07] S: [p_r1_region] 3: >>>> p_r1_region 1992 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    ---------------------------------------
    p_r1:p_r1_region, ql=0: {}
    ---------------------------------------
    p_r2:p_s21, ql=2:
    0: force_region_init
    1: [n=2]::INIT_META_SIGNAL:p_r2_region [n=1]::BOUNCE_SAME_META_SIGNAL:p_r2_region ->
      [n=3]::INIT_META_SIGNAL:p_p22 [n=2]::INIT_META_SIGNAL:p_r2_region ->

    ---------------------------------------
    p_p11_r1:p_p11_r1_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p11_r2:p_p11_r2_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p12_r1:p_p12_r1_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p12_r2:p_p12_r2_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p12_p11_r1:p_p12_p11_r1_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p12_p11_r2:p_p12_p11_r2_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p22_r1:p_p22_r1_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p22_r2:p_p22_r2_under_hidden_region, ql=0: {}
    ---------------------------------------
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    [07] S: [x] INIT_SIGNAL:p_p11_s21
    [07] S: [p_r2_region] 3: [['p_p11_s11', 'p_p11_s21'], 'p_r2_region']
    [07] S: [x] INIT_SIGNAL:p_p11
    [07] S: [p_r2_region] 3: INIT_SIGNAL
    [07] S: [x] ENTRY_SIGNAL:p_s21
    [07] S: [p_r2_region] 3: [n=3]::INIT_META_SIGNAL:p_p22 [n=2]::INIT_META_SIGNAL:p_r2_region ->

    [07] S: [x] INIT_SIGNAL:p_s21
    [07] S: [x] INIT_SIGNAL:p
    [07] S: [x] POST_FIFO:BOUNCE_SAME_META_SIGNAL
    [07] S: [x] <- Queued:(1) Deferred:(0)
    [07] S: [x] BOUNCE_SAME_META_SIGNAL:p
    [07] S: [p_r2_region] 3: >>>> p_r2_region 523 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    ---------------------------------------
    p_r1:p_p11, ql=0: {}
    ---------------------------------------
    p_r2:p_r2_region, ql=0: {}
    ---------------------------------------
    p_p11_r1:p_p11_s11, ql=0: {}
    ---------------------------------------
    p_p11_r2:p_p11_s21, ql=0: {}
    ---------------------------------------
    p_p12_r1:p_p12_r1_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p12_r2:p_p12_r2_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p12_p11_r1:p_p12_p11_r1_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p12_p11_r2:p_p12_p11_r2_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p22_r1:p_p22_r1_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p22_r2:p_p22_r2_under_hidden_region, ql=0: {}
    ---------------------------------------
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    [07] S: [x] EXIT_SIGNAL:p_p11_s11
    [07] S: 1: [['p_p11_s11', 'p_p11_s21'], ['p_p22_s11', 'p_p22_s21']]
    [07] S: [x] EXIT_SIGNAL:p_p11_s21
    [07] S: 1: [n=1]::BOUNCE_SAME_META_SIGNAL:outer [n=0]::SRE3:outer ->
      [n=2]::INIT_META_SIGNAL:p_r2_region [n=1]::BOUNCE_SAME_META_SIGNAL:p_r2_region ->
        [n=3]::INIT_META_SIGNAL:p_p22 [n=2]::INIT_META_SIGNAL:p_r2_region ->

    [07] S: [x] EXIT_SIGNAL:p_p11
    [07] S: 1: [n=2]::INIT_META_SIGNAL:p_r2_region [n=1]::BOUNCE_SAME_META_SIGNAL:p_r2_region ->
      [n=3]::INIT_META_SIGNAL:p_p22 [n=2]::INIT_META_SIGNAL:p_r2_region ->

    [07] S: [x] ENTRY_SIGNAL:p_p11
    [07] S: [x] ENTRY_SIGNAL:p_p11_s11
    [07] S: [x] INIT_SIGNAL:p_p11_s11
    [07] S: [x] ENTRY_SIGNAL:p_p11_s21
    [07] S: [x] INIT_SIGNAL:p_p11_s21
    [07] S: 1: >>>> p 3962 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    ---------------------------------------
    p_r1:p_p11, ql=0: {}
    ---------------------------------------
    p_r2:p_p22, ql=0: {}
    ---------------------------------------
    p_p11_r1:p_p11_s11, ql=0: {}
    ---------------------------------------
    p_p11_r2:p_p11_s21, ql=0: {}
    ---------------------------------------
    p_p12_r1:p_p12_r1_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p12_r2:p_p12_r2_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p12_p11_r1:p_p12_p11_r1_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p12_p11_r2:p_p12_p11_r2_under_hidden_region, ql=0: {}
    ---------------------------------------
    p_p22_r1:p_p22_s11, ql=0: {}
    ---------------------------------------
    p_p22_r2:p_p22_s21, ql=0: {}
    ---------------------------------------
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    [07] S: [x] INIT_SIGNAL:p_p11
    [07] S: [x] EXIT_SIGNAL:p_s21
    [07] S: [x] ENTRY_SIGNAL:p_p22
    [07] S: [x] ENTRY_SIGNAL:p_p22_s11
    [07] S: [x] INIT_SIGNAL:p_p22_s11
    [07] S: [x] ENTRY_SIGNAL:p_p22_s21
    [07] S: [x] INIT_SIGNAL:p_p22_s21
    [07] S: [x] INIT_SIGNAL:p_p22
    [07] S: [x] <- Queued:(0) Deferred:(0)
    [07] R: --- ['middle'] <- SRE3 == [['p_p11_s11', 'p_p11_s21'], ['p_p22_s11', 'p_p22_s21']]

