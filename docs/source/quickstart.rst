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
instance the ``xml_chart_5.py`` file tells us to test it with:

.. code-block:: bash

  while inotifywait -e modify xml_chart_5.py logger_config.yaml; do python xml_chart_5.py; done

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


