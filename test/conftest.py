import os
import re
from pathlib import Path
from contextlib import contextmanager

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = Path(dir_path) / '..' / 'data'

@contextmanager
def stripped(log_items):
  def item_without_timestamp(item):
    m = re.match(r"[0-9-:., ]+ DEBUG:S: (.+)$", item)
    if(m is not None):
      without_time_stamp = m.group(1)
    else:
      without_time_stamp = item
    return without_time_stamp

  targets = log_items
  if len(targets) > 1:
    stripped_target = []
    for target_item in targets:
      target_item = target_item.strip()
      if len(target_item) != 0:
        stripped_target_item = item_without_timestamp(target_item)
        stripped_target.append(stripped_target_item)
    yield(stripped_target)

  else:
    target = log_items
    yield(item_without_timestamp(target))

def get_log_as_stripped_string(path):
  result = "\n"
  with open(str((path).resolve())) as fp:
    with stripped(fp.readlines()) as spy_lines:
      for s in spy_lines:
        result += s + '\n'
  return result
