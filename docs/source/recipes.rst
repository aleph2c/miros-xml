Recipes
=======

.. contents::
  :depth: 2
  :local: 
  :backlinks: none

This section will focus on Python techniques that aren't specific to the design.

.. _recipes-finding-the-decorators-for-a-function/method:

Finding a Function/Method's Decorators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This package is automatically building functions, which will be based on hand
written functions which have been instrumented by function wrappers
(decorators).  To reflect upon how a function has been decorated, so you can
programmatically build your own version of it, you can use the solution provided
by Shane Holloway on `stackoverflow <https://stackoverflow.com/a/9580006>`_.

.. code-block:: python

  def find_decorators(target):
      '''find a method/function's decorators

      **Note**:
         This function will not work for function's built from templates

         Solution provided by Shane Holloway on `stackoverflow
         <https://stackoverflow.com/a/9580006>`_

      **Args**:
         | ``target`` (callable): the function/method


      **Returns**:
         (dict): key/value pair, key is the function/method the value is array
        |        of decorators

      **Example(s)**:

      .. code-block:: python

          find_decorators(p_p22_s22)  \
               # => {'p_p22_s22': ["Name(id='orthogonal_state', ctx=Load())"]

          find_decorators(p)  # => {'p': ["Name(id='state', ctx=Load())"]

      '''
      import ast, inspect
      res = {}
      def visit_FunctionDef(node):
          res[node.name] = [ast.dump(e) for e in node.decorator_list]

      V = ast.NodeVisitor()
      V.visit_FunctionDef = visit_FunctionDef
      V.visit(compile(inspect.getsource(target), "?", 'exec', ast.PyCF_ONLY_AST))
      return res

.. _recipes-creating-functions-from-templates:

Creating Functions from Templates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The miros-xml package :ref:`adds four hidden functions
<<how_it_works-hidden-states-and-what-they-are-for>` (grey in the diagram) for
each orthogonal state:

.. image:: _static/hidden_states_1.svg
    :target: _static/hidden_states_1.pdf
    :class: noscale-center

**Goal**: We want the ``add`` method of the ``Regions`` object to create the grayed
out functions for us, using a predetermined naming convention.

We will use the module_namespace to attach and get functions based on a string,
using the Python ``setattr/getattr`` builtins.

.. code-block:: python

   import sys
   module_namespace = sys.modules[__name__]

   # these are the same
   module_namespace.p_r1_under_hidden_region()
   getattr(module_namespace, 'p_r1_under_hidden_region')()

Before we make something that writes functions, we will type one out by hand,
pass a full regression test, then identify the parts of the function we want to
change:

.. code-block:: python
  :emphasize-lines: 2, 12
  :linenos:

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

Things that need to change in the above function:

* it needs a unique name (line 1)
* it will reference the ``p_r1_region`` function, line 12.

Now we write a template, knowing that we will rename it later.  We use the
``*``, character to separate the non-mandatorily-named arguments from the ones
that require a name; ``region_state_name``.

.. code-block:: python
  :emphasize-lines: 7, 8, 16
  :linenos:

  def template_under_hidden_region(r, e, *, region_state_name=None, **kwargs):
    '''some docstring'''

    status = return_status.UNHANDLED
    __super__ = r.bottom

    region_state_function = \
      getattr(module_namespace, region_state_name)

    if(e.signal == SEARCH_FOR_SUPER_SIGNAL):
      r.temp.fun = __super__
      return return_status.SUPER

    if(r.token_match(e.signal_name, "enter_region")):
      r._p_spy(e)
      status = r.trans(region_state_function)
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

Notice on lines 7-8, we use ``getattr(module_namespace, region_state_name)`` to
call the function.  This function is used on line 16.

If ``region_state_name`` was set to ``p_r1_region``, our function would behave
the same as the one we wrote out by hand.  Notice, that the template does not
have an ``orthogonal`` decorator, but our handwritten function did. This
decorator will be added later.

Now we want to create a new function based on some naming conventions:

.. code-block:: python
  :emphasize-lines: 38-44, 47-48, 52, 58, 62-66
  :linenos:

  from functools import partial
  from functools import lru_cache
  from functools import update_wrapper

  # functions at the top of the file
  @lru_cache(maxsize=128)
  def under_hidden_region_function_name(region_name):
    return region_name + "_under_hidden_region"

  @lru_cache(maxsize=128)
  def region_function_name(region_name):
    return region_name + "_region"

  @lru_cache(maxsize=128)
  def over_hidden_region_function_name(region_name):
    return region_name + "_over_hidden_region"

  @lru_cache(maxsize=128)
  def final_region_function_name(region_name):
    return region_name + "_final"

  # ..

  # in the Regions class:
  def add(self, region_name, outer):
    # ...
    region_name = "p_r1"  # force the region_name for our example
    under_s = under_hidden_region_function_name(region_name)
    region_s = region_function_name(region_name)
    over_s = over_hidden_region_function_name(region_name)
    final_s = final_region_function_name(region_name)

    # only the template_under_hidden_region and under_state_function=under_s
    # variables used by the template, the rest caught by kwargs.  We do this so
    # all templates are made the same way, the user doesn't have to care what is
    # needed or what isn't, you just include all of the information and the
    # template uses what it needs.
    under_hidden_state_function = partial(
      template_under_hidden_region,
      under_state_function=under_s,
      region_state_name=region_s,
      over_state_function=over_s,
      final_state_function=final_s
    )
    # add the __doc__ and __name__ from the template_under_hidden_region
    # to the under_hidden_state_function
    under_hidden_state_function = update_wrapper(under_hidden_state_function,
        template_under_hidden_region)

    # change the name of the under_hidden_state_function to
    # 'p_r1_under_hidden_region'
    under_hidden_state_function.__name__ = under_s

    # add our orthogonal_state instrumentation decorator.  This decorator uses
    # the '@wraps(fn)' decorator, so that the name is conserved after it has
    # been wrapped.  So under_hidden_state_function is still called
    # 'p_r1_under_hidden_region' after it has been decorated
    under_hidden_state_function = othogonal_state(under_hidden_state_function)

    # add the newly created 'p_r1_under_hidden_region' function to our module's
    # namespace
    setattr(
      module_namespace,
      under_s,
      under_hidden_state_function,
    )
    assert(under_hidden_state_function.__name__ == under_s)
    assert callable(under_hidden_state_function)
    assert callable(getattr(module_namespace, under_s))

    # ... build other hidden functions
    # ... construct the regions

The function is created on lines 38-44.  We use the ``partial`` function to set
some default parameters for this function, then assign it to the
``under_hidden_state_function`` variable.  Then we give it the meta data from
the ``template_under_hidden_region`` on lines 47 and 48.  Then we explicitely
name it on line 52.

Once our function is named, we wrap its orthogonal_state decorator, line 58.  On lines 62
to 66 we add this named function to the module's namespace.  At this point we
can call ``p_r1_under_hidden_region`` as if we wrote it by hand.

The ``p_r1_under_hidden_region`` function has been programmatically created, so
we delete the one we made by hand, the rerun our regression to ensure the system
still works.

To complete the work we follow the same process for the ``final``, ``region`` and ``over_hidden_region`` functions.

.. warning::

  If you have done work like this, never use the eval(function_name_as_string)
  to get access to the function.  This will destroy the function's ``__name__`` and
  ``__doc__`` values and cause your tests to fail.



