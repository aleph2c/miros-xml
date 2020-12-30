# WARNING: Read the pytest.ini file if you want to port these tests to other
# projects (otherwise you will make the same mistakes again and lose hours
# trying to debug your projects):

# The pytest.ini (polluting the top level directory of this project) is full of
# "warnings" about pytest.  As a software project it has been fetishized with a
# lot of on-by-default "features".  The "features" add warning noise and
# highjack python's logging system.

import os
import sys
import dill
import time
import pytest
from pathlib import Path
from collections import namedtuple
from miros_scxml.xml_to_miros import XmlToMiros

# RULE OF THUMB with pytest:  Don't rely on its auto-configuration magic.
# Example: Pytest can't find its conftest.py file in this directory, so I have
# to force its import into this testing package to get access to the common
# testing functions.  As a rule, avoid any of these "batteries included" options
# of pytest because they will just waste your time.

# Just force it to work and focus on the thing you care about instead of pytest.

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from conftest import get_log_as_stripped_string
data_path = Path(dir_path) / '..' / 'data'

try:
  os.remove(str(Path(data_path / 'build_payload.p')))
except:
  pass
def _build_payload():
  Payload = namedtuple('Payload', ['proof'])
  payload = Payload(proof=True)
  return payload
dill.dump(_build_payload, open(str(Path(data_path / "build_payload.p")), "wb"))

@pytest.mark.pinx
def test_anything_pinx():
  '''
  We are testing if a SCXML file which imports a dill pickle (serialized
  function) will work.
  '''
  path = data_path / 'pinx_test_1.scxml'
  miros_code_path = data_path / 'pinx_test_1.py'
  xml_chart = XmlToMiros(path, miros_code_path=miros_code_path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  assert xml_chart.get_name() == "pinx"  

@pytest.mark.pinx
def test_pinx_passing_condition():
  '''
  Here we test the dynamics of a serialized function which injects a payload
  into our event.  The chart posts and event with a payload, given to it by a
  dilled function.  The receiving code has a guard condition which will cause
  a transition to "Pass" if "e.payload.proof == True".  In the XML the code to
  make the python "e.payload.proof == True" is "event.payload.proof == True".
  The XML code is written this way to make it easy to remember, since the
  "event" attribute name is used in the "send" tag.

  This test confirms a passing guard can cause a transition to another state.

  '''
  path = data_path / 'pinx_test_1.scxml'
  miros_code_path = data_path / 'pinx_test_1.py'
  xml_chart = XmlToMiros(path, miros_code_path=miros_code_path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(1.00)
  result = ao.spy()
  target = \
    ['START',
        'SEARCH_FOR_SUPER_SIGNAL:Start',
        'ENTRY_SIGNAL:Start',
        'POST_FIFO:evidence',
        'INIT_SIGNAL:Start',
        '<- Queued:(1) Deferred:(0)',
        'evidence:Start',
        'SEARCH_FOR_SUPER_SIGNAL:Pass',
        'SEARCH_FOR_SUPER_SIGNAL:Start',
        'EXIT_SIGNAL:Start',
        'ENTRY_SIGNAL:Pass',
        'Pass!',
        'INIT_SIGNAL:Pass',
        '<- Queued:(0) Deferred:(0)']
  assert result == target 

@pytest.mark.pinx
def test_pinx_failing_condition():
  '''
  Here we test the dynamics of a serialized function which injects a payload
  into our event.  The chart posts and event with a payload, given to it by a
  dilled function.  The receiving code has a guard condition which will block
  a transition to "Pass" unless "e.payload.proof == False".  In the XML the code to
  make the python "e.payload.proof == False" is "event.payload.proof == False".
  The XML code is written this way to make it easy to remember, since the
  "event" attribute name is used in the "send" tag.

  This test confirms a failing guard can block a transition to another state.

  '''
  path = data_path / 'pinx_test_2.scxml'
  miros_code_path = data_path / 'pinx_test_2.py'
  xml_chart = XmlToMiros(path, miros_code_path=miros_code_path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(1.10)
  result = ao.spy()
  target = \
    ['START',
     'SEARCH_FOR_SUPER_SIGNAL:Start',
     'ENTRY_SIGNAL:Start',
     'POST_FIFO:evidence',
     'INIT_SIGNAL:Start',
     '<- Queued:(1) Deferred:(0)',
     'evidence:Start',
     'evidence:Start:HOOK',
     '<- Queued:(0) Deferred:(0)',
     'to_fail:Start',
     'SEARCH_FOR_SUPER_SIGNAL:Fail',
     'SEARCH_FOR_SUPER_SIGNAL:Start',
     'EXIT_SIGNAL:Start',
     'ENTRY_SIGNAL:Fail',
     'Fail!',
     'INIT_SIGNAL:Fail',
     '<- Queued:(0) Deferred:(0)']
  assert result == target 


@pytest.mark.pinx
def test_pinx_glob_passing_condition():
  '''
  Glob test; guard permitting a transition.  This test is like pinx_test_1, but
  the catching handler looks for an event glob rather than a specific signal
  name.
  '''
  path = data_path / 'pinx_test_3.scxml'
  miros_code_path = data_path / 'pinx_test_3.py'
  xml_chart = XmlToMiros(path, miros_code_path=miros_code_path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(0.10)
  result = ao.spy()
  target = \
      ['START',
       'SEARCH_FOR_SUPER_SIGNAL:Start',
       'ENTRY_SIGNAL:Start',
       'POST_FIFO:evidence',
       'INIT_SIGNAL:Start',
       '<- Queued:(1) Deferred:(0)',
       'evidence:Start',
       'SEARCH_FOR_SUPER_SIGNAL:Pass',
       'SEARCH_FOR_SUPER_SIGNAL:Start',
       'EXIT_SIGNAL:Start',
       'ENTRY_SIGNAL:Pass',
       'Pass!',
       'INIT_SIGNAL:Pass',
       '<- Queued:(0) Deferred:(0)']
  assert result == target 

@pytest.mark.pinx
def test_pinx_glob_failing_condition():
  '''
  Glob test; guard blocking a transition.  This test is like pinx_test_2, but
  the catching handler looks for an event glob rather than a specific signal
  name.
  '''
  path = data_path / 'pinx_test_4.scxml'
  miros_code_path = data_path / 'pinx_test_4.py'
  xml_chart = XmlToMiros(path, miros_code_path=miros_code_path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(1.10)
  result = ao.spy()
  target = \
    ['START',
     'SEARCH_FOR_SUPER_SIGNAL:Start',
     'ENTRY_SIGNAL:Start',
     'POST_FIFO:evidence',
     'INIT_SIGNAL:Start',
     '<- Queued:(1) Deferred:(0)',
     'evidence:Start',
     'evidence:Start:HOOK',
     '<- Queued:(0) Deferred:(0)',
     'to_fail:Start',
     'SEARCH_FOR_SUPER_SIGNAL:Fail',
     'SEARCH_FOR_SUPER_SIGNAL:Start',
     'EXIT_SIGNAL:Start',
     'ENTRY_SIGNAL:Fail',
     'Fail!',
     'INIT_SIGNAL:Fail',
     '<- Queued:(0) Deferred:(0)']
  assert result == target 

@pytest.mark.pinx
def test_pinx_automatic_passing_condition():
  '''
  Here we add the <assign> feature to the library. The entry condition of the
  'Start' state assigns the result of the dilled_funtion into an attribute of
  the active object which runs the state machine.  The automatic transition from
  "Pass" from "Start" contains a guarding "cond" which allows the transition
  only if python:"self.condition.proof == True", xml: "condition.proof == True".

  '''
  path = data_path / 'pinx_test_5.scxml'
  miros_code_path = data_path / 'pinx_test_5.py'
  xml_chart = XmlToMiros(path, miros_code_path=miros_code_path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(1.00)
  result = ao.spy()
  target = \
      ['START',
          'SEARCH_FOR_SUPER_SIGNAL:Start',
          'ENTRY_SIGNAL:Start',
          'POST_FIFO:SCXML_INIT_SIGNAL',
          'INIT_SIGNAL:Start',
          '<- Queued:(1) Deferred:(0)',
          'SCXML_INIT_SIGNAL:Start',
          'SEARCH_FOR_SUPER_SIGNAL:Pass',
          'SEARCH_FOR_SUPER_SIGNAL:Start',
          'EXIT_SIGNAL:Start',
          'ENTRY_SIGNAL:Pass',
          'Pass!',
          'INIT_SIGNAL:Pass',
          '<- Queued:(0) Deferred:(0)']

  assert result == target 

@pytest.mark.pinx
def test_pinx_automatic_failing_condition():
  '''
  Like pinx_test_5.scxml, but here we block instead of pass.

  '''
  path = data_path / 'pinx_test_6.scxml'
  miros_code_path = data_path / 'pinx_test_6.py'
  xml_chart = XmlToMiros(path, miros_code_path=miros_code_path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(1.00)
  result = ao.spy()
  target = \
      ['START',
          'SEARCH_FOR_SUPER_SIGNAL:Start',
          'ENTRY_SIGNAL:Start',
          'POST_FIFO:SCXML_INIT_SIGNAL',
          'INIT_SIGNAL:Start',
          '<- Queued:(1) Deferred:(0)',
          'SCXML_INIT_SIGNAL:Start',
          'SCXML_INIT_SIGNAL:Start:HOOK',
          '<- Queued:(0) Deferred:(0)']

  assert result == target 


@pytest.mark.pinx
def test_pinx_passing_attribute_variable():
  '''
  Mixing ideas of test pinx_test_2 and pinx_text_5
  Use a dilled function to set a value in an attribute variable.  Later, a
  'caught' event with a guard, permits the transition if
  python:"self.condition.proof, xml"condition.proof == True"

  '''
  path = data_path / 'pinx_test_7.scxml'
  miros_code_path = data_path / 'pinx_test_7.py'
  xml_chart = XmlToMiros(path, miros_code_path=miros_code_path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(1.00)
  result = ao.spy()
  target = \
      ['START',
          'SEARCH_FOR_SUPER_SIGNAL:Start',
          'ENTRY_SIGNAL:Start',
          'POST_FIFO:evidence',
          'INIT_SIGNAL:Start',
          '<- Queued:(1) Deferred:(0)',
          'evidence:Start',
          'SEARCH_FOR_SUPER_SIGNAL:Pass',
          'SEARCH_FOR_SUPER_SIGNAL:Start',
          'EXIT_SIGNAL:Start',
          'ENTRY_SIGNAL:Pass',
          'Pass!',
          'INIT_SIGNAL:Pass',
          '<- Queued:(0) Deferred:(0)']

  assert result == target 

@pytest.mark.pinx
def test_pinx_failing_attribute_variable():
  '''
  Mixing ideas of test pinx_test_3 and pinx_text_6
  Use a dilled function to set a value in an attribute variable.  Later, a
  'caught' event with a guard, permits the transition if
  python:"self.condition.proof == False", xml"condition.proof == False"

  '''
  path = data_path / 'pinx_test_8.scxml'

  miros_code_path = data_path / 'pinx_test_8.py'
  xml_chart = XmlToMiros(path, miros_code_path=miros_code_path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(1.00)
  result = ao.spy()

  target = \
    ['START',
        'SEARCH_FOR_SUPER_SIGNAL:Start',
        'ENTRY_SIGNAL:Start',
        'POST_FIFO:evidence',
        'INIT_SIGNAL:Start',
        '<- Queued:(1) Deferred:(0)',
        'evidence:Start',
        'evidence:Start:HOOK',
        '<- Queued:(0) Deferred:(0)']

  assert result == target 
