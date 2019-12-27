# Read the pytest.ini file if you want to port these tests to other projects:

# The pytest.ini is full of warnings about pytest.  As a software project it has
# been fetishized with a lot of on-by-default "features".  The "features" add
# warning noise and highjack python's logging.

import os
import re
import time
import pytest
from pathlib import Path
from contextlib import contextmanager
from miros_scxml.xml_to_miros import XmlToMiros    

from miros import pp
from miros import Event
from miros import spy_on
from miros import signals
from miros import ActiveObject
from miros import return_status

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
    target = log
    yield(item_without_timestamp(target))

def get_log_as_stripped_string(path):
  result = "\n"
  with open(str((path).resolve())) as fp:
    with stripped(fp.readlines()) as spy_lines:
      for s in spy_lines:
        result += s + '\n'
  return result

@pytest.mark.skip()
@pytest.mark.scxml
def test_scxml_get_name():
  path = data_path / 'scxml_test_1.scxml'
  main = XmlToMiros(path)
  assert main.get_name() == "Scxml"  

# state_dict = {}
# state_dict['Start'] = {}
# state_dict['Start']['p'] = None
# state_dict['Start']['cl'] = []
# state_dict['Start']['cl'].append(
#   {'ENTRY_SIGNAL':"self.scribble('Hello from \'start\')"}
# )
# state_dict['Start']['cl'].append(
#   {'INIT_SIGNAL':'status = return_state.HANDLED'}
# )
# state_dict['Start']['cl'].append(
#   {'SCXML_INIT_SIGNAL': "self.trans(Work)"}
# )

# state_dict['Work'] = {}
# state_dict['Work']['p'] = None
# state_dict['Work']['cl'] = []
# state_dict['Work']['cl'].append(
#   {'ENTRY_SIGNAL': "self.scribble('Hello from \'work\')"}
# )
# state_dict['Work']['cl'].append(
#   {'INIT_SIGNAL':'status = return_state.HANDLED'}
# )
# state_dict['Work']['cl'].append(
#   {'SCXML_INIT_SIGNAL':"status = return_state.HANDLED"}
# )
# state_dict['start_at'] = 'Start'
@pytest.mark.skip()
@pytest.mark.scxml
def test_scxml_xml_dict_structured_well():
  '''
  If there is no "initial" in the attrib of the scxml tag, then start in the
  first state/parellel region of the chart.  The XML used in this test is
  missing the "initial" attrib, so we are confirming that it will start in the
  'Start' state.
  '''

  path = data_path / 'scxml_test_1.scxml'
  xml_chart = XmlToMiros(path)
  assert 'Start' in xml_chart._state_dict
  assert 'Work' in xml_chart._state_dict

  def is_signal_in_state_dict(signal_name, state_name):
    result = False
    for _dict in xml_chart._state_dict[state_name]['cl']:
      if signal_name in _dict:
        result = True
        break;
    return result

  assert is_signal_in_state_dict('ENTRY_SIGNAL', 'Start')
  assert is_signal_in_state_dict('ENTRY_SIGNAL', 'Work')
  assert is_signal_in_state_dict('INIT_SIGNAL', 'Start')
  assert is_signal_in_state_dict('INIT_SIGNAL', 'Work')
  assert is_signal_in_state_dict('SCXML_INIT_SIGNAL', 'Start')
  assert is_signal_in_state_dict('SCXML_INIT_SIGNAL', 'Work')

  assert None == xml_chart._state_dict['Start']['p']
  assert None == xml_chart._state_dict['Work']['p']

@pytest.mark.skip()
@pytest.mark.scxml
def test_scxml_xml_dict_contents():

  path = data_path / 'scxml_test_1.scxml'
  xml_chart = XmlToMiros(path)

  def string_for_signal(signal_name, state_name):
    code_string = None
    for _dict in xml_chart._state_dict[state_name]['cl']:
      if signal_name in _dict:
        code_string = _dict[signal_name]
        break;
    return code_string

  entry_state_code = \
    string_for_signal(
      state_name='Start',
      signal_name='ENTRY_SIGNAL'
    )

  assert('self.scribble(\'Hello from \\"start\\"\')\nstatus = return_status.HANDLED' == entry_state_code)

  scxml_init_signal_code = \
    string_for_signal(
      state_name='Start',
      signal_name='SCXML_INIT_SIGNAL'
    )

  assert('status = self.trans(Work)' == scxml_init_signal_code)

  entry_state_code = \
    string_for_signal(
      state_name='Work',
      signal_name='ENTRY_SIGNAL'
    )

  assert("self.scribble('Hello from \\'work\\'')\nstatus = return_status.HANDLED" == entry_state_code)

@pytest.mark.skip()
@pytest.mark.scxml
def test_start_at_with_single_initial():
  path = data_path / 'scxml_test_1.scxml'
  xml_chart = XmlToMiros(path)
  assert xml_chart._state_dict['start_at'] == 'Start'

  path = data_path / 'scxml_test_2.scxml'
  xml_chart = XmlToMiros(path)
  assert xml_chart._state_dict['start_at'] == 'Work'

@pytest.mark.scxml
def test_build_a_small_chart():
  path = data_path / 'scxml_test_1.scxml'
  xml_chart = XmlToMiros(path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(0.1)

  result = get_log_as_stripped_string(data_path / 'scxml_test_1.log')
  print(result)

  target = """
[Scxml] START
[Scxml] SEARCH_FOR_SUPER_SIGNAL:Start
[Scxml] ENTRY_SIGNAL:Start
[Scxml] POST_FIFO:SCXML_INIT_SIGNAL
[Scxml] Hello from "start"
[Scxml] INIT_SIGNAL:Start
[Scxml] <- Queued:(1) Deferred:(0)
[Scxml] SCXML_INIT_SIGNAL:Start
[Scxml] SEARCH_FOR_SUPER_SIGNAL:Work
[Scxml] SEARCH_FOR_SUPER_SIGNAL:Start
[Scxml] EXIT_SIGNAL:Start
[Scxml] ENTRY_SIGNAL:Work
[Scxml] Hello from 'work'
[Scxml] INIT_SIGNAL:Work
[Scxml] <- Queued:(0) Deferred:(0)
"""
  assert(target == result)


