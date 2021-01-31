.. # i_create_functions_from_templates.rst
.. # to include:
.. .. include:: i_creating_functions_from_templates.rst

The miros-xml package :ref:`adds four hidden functions
<how_it_works-hidden-states-and-what-they-are-for>` (grey in the diagram) for
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
that require a name; ``region_state_name``.  We also add the ``**kwargs`` as a
place holder for named arguments this function will not use.

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
get the region function.  This function is used on line 16.

If ``region_state_name`` was set to ``p_r1_region``, our function would behave
the same as the one we wrote out by hand.  Notice, that the template does not
have an ``orthogonal`` decorator, but our handwritten function does. This
decorator will be added later.

Now we want to create a new function based on some naming conventions:

.. code-block:: python
  :emphasize-lines: 35-40, 47-55, 57-66, 70-72
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
  def add(self, region_name, initial_state, outer):
    # ...
    region_name = "p_r1"  # force the region_name for our example
    under_s = under_hidden_region_function_name(region_name)
    region_s = region_function_name(region_name)
    over_s = over_hidden_region_function_name(region_name)
    final_s = final_region_function_name(region_name)

    # The 'template_under_hidden_region' is defined in documentation above,
    # the other templates would be defined in similar manner
    for function_name, template in [
      (under_s, template_under_hidden_region),
      (region_s, template_region),
      (over_s, template_over_hidden_region),
      (final_s, template_final)
    ]:

      # The 'template' and 'region_state_name' are used in the
      # 'template_under_hidden_region', but the other named arguments are
      # placed into its kwargs argument and ignored by the function.  These
      # extra named arguments are needed by one or more of the other template
      # functions.
      fn = partial(
        template,
        this_function_name=function_name,
        under_region_state_name=under_s,
        region_state_name=region_s,
        over_region_state_name=over_s,
        final_state_name=final_s,
        initial_state_name=initial_state
      )

      # give the fn the meta data defined by its template
      fn = update_wrapper(fn, template)
      # over-write the function name
      fn.__name__ = function_name
      # wrap the function with an instrumentation decorator
      fn = orthogonal_state(fn)
      # over-write the instrumented function with its name
      fn.__name__ = function_name
      # place the new function in this module's namespace
      setattr(module_namespace, function_name, fn)

    # Here we get the functions from the name space using our function names as
    # strings.
    region_state_function = getattr(module_namespace, region_s)
    over_hidden_state_function = getattr(module_namespace, over_s)
    under_hidden_state_function = getattr(module_namespace, under_s)

    assert(callable(p_r1_under_hidden_region))    # can call what was created
    assert(callable(under_hidden_state_function)) # can call indirectly too
    assert(callable(region_function_name))
    assert(callable(over_hidden_state_function))

    # ... construct the regions

The above code listing shows how all of our hidden functions would be made from
a set of templates, an initial_state target and some function names built by a
convention using the region name.

Lines 35-40 show how we specify a template-naming partnership.

Lines 47-55 shows how a new function is built from the template.

Lines 57-66 name our new function, wrap it with an instrumentation decorator,
name this wrapped function, then attach it to the modules namespace.

Lines 70-72 show how to get a function using its name.

.. warning::

  If you have done work like this, never use the eval(function_name_as_string)
  to get access to the function.  This will destroy the function's ``__name__`` and
  ``__doc__`` values and cause your tests to fail.
