import sys
from functools import wraps
from functools import partial
from functools import update_wrapper

module_namespace = sys.modules[__name__]

def orthogonal_state(fn):
  '''a wrapper to provide instrumentation'''
  @wraps(fn)
  def _pspy_on(region, *args, **kwargs):
    e = args[0] if len(args) == 1 else args[-1]
    fn(region, e)
  return _pspy_on

def sub_template(r, e, *, this_function_s=None, subfunction=None, region=None):
  '''hidden function which belongs to the {region} region'''
  print(this_function_s)

def comp_template(r, e, *, this_function_s=None, subfunction=None, region=None):
  '''hidden comp function which belongs to the {region} region'''

  if e == 'init':
    getattr(module_namespace, subfunction)(r, e)
  else:
    print(this_function_s)


if __name__ == '__main__':
  sub_template('r','e')

  sub_fn_name, region = 'new_sub_function', 'mary'

  sub_fn = partial(sub_template, this_function_s=sub_fn_name, region=region)
  sub_fn = update_wrapper(sub_fn, sub_template)
  sub_fn.__name__ = sub_fn_name
  sub_fn.__doc__ = sub_fn.__doc__.format(region=region)
  sub_fn = orthogonal_state(sub_fn)
  setattr(module_namespace, sub_fn_name, sub_fn)
  sub_fn('r', 'e')

  comp_fn_name, region = 'new_comp_function', region

  comp_fn = partial(comp_template,
      this_function_s=comp_fn_name,
      subfunction=sub_fn_name,
      region=region)
  comp_fn = update_wrapper(comp_fn, comp_template)
  comp_fn.__name__ = comp_fn_name
  comp_fn.__doc__ = comp_fn.__doc__.format(region=region)
  comp_fn = orthogonal_state(comp_fn)
  setattr(module_namespace, comp_fn_name, comp_fn)

  fn = getattr(module_namespace, comp_fn_name)
  fn('r', 'e')
  print(fn.__doc__)

  fn('r', 'init')

