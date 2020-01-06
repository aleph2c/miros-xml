# WARNING: Read the pytest.ini file if you want to port these tests to other
# projects (otherwise you will make the same mistakes again and lose hours
# trying to debug your projects):

# The pytest.ini (polluting the top level directory of this project) is full of
# "warnings" about pytest.  As a software project it has been fetishized with a
# lot of on-by-default "features".  The "features" add warning noise and
# highjack python's logging system.

import os
import sys
import time
import miros
import pytest
import logging
from pathlib import Path
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

@pytest.mark.transition
def test_transition_send_to_post_fifo_1():
  logging.shutdown()
  time.sleep(0.1)
  path = data_path / 'test403.scxml'
  xml_chart = XmlToMiros(path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(2.00)
  result = ao.spy()
  target = \
    ['START',
      'SEARCH_FOR_SUPER_SIGNAL:s0',
      'ENTRY_SIGNAL:s0',
      'INIT_SIGNAL:s0',
      'SEARCH_FOR_SUPER_SIGNAL:s01',
      'ENTRY_SIGNAL:s01',
      'INIT_SIGNAL:s01',
      '<- Queued:(0) Deferred:(0)',
      'timeout.token1.token2:s01',
      'SEARCH_FOR_SUPER_SIGNAL:_fail',
      'SEARCH_FOR_SUPER_SIGNAL:s01',
      'EXIT_SIGNAL:s01',
      'ENTRY_SIGNAL:_fail',
      '\n|    FAIL!!!    |\n',
      'INIT_SIGNAL:_fail',
      '<- Queued:(0) Deferred:(0)']

  assert target == result
  time.sleep(0.1)

@pytest.mark.transition
def test_transition_send_to_post_fifo_2():
  logging.shutdown()
  time.sleep(0.1)
  path = data_path / 'test403_eventexpr_delay.scxml'
  xml_chart = XmlToMiros(path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(2.00)
  result = ao.spy()
  target = \
    ['START',
      'SEARCH_FOR_SUPER_SIGNAL:s0',
      'ENTRY_SIGNAL:s0',
      'INIT_SIGNAL:s0',
      'SEARCH_FOR_SUPER_SIGNAL:s01',
      'ENTRY_SIGNAL:s01',
      'INIT_SIGNAL:s01',
      '<- Queued:(0) Deferred:(0)',
      'timeout.token1.token2:s01',
      'SEARCH_FOR_SUPER_SIGNAL:_fail',
      'SEARCH_FOR_SUPER_SIGNAL:s01',
      'EXIT_SIGNAL:s01',
      'ENTRY_SIGNAL:_fail',
      '\n|    FAIL!!!    |\n',
      'INIT_SIGNAL:_fail',
      '<- Queued:(0) Deferred:(0)']

  assert target == result
  time.sleep(0.1)

@pytest.mark.transition
def test_transition_send_to_post_fifo_3():
  logging.shutdown()
  time.sleep(0.1)
  path = data_path / 'test403_eventexpr_delayexpr.scxml'
  xml_chart = XmlToMiros(path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(2.00)
  ao.stop()
  result = ao.spy()
  target = \
    ['START',
      'SEARCH_FOR_SUPER_SIGNAL:s0',
      'ENTRY_SIGNAL:s0',
      'INIT_SIGNAL:s0',
      'SEARCH_FOR_SUPER_SIGNAL:s01',
      'ENTRY_SIGNAL:s01',
      'INIT_SIGNAL:s01',
      '<- Queued:(0) Deferred:(0)',
      'timeout.token1.token2:s01',
      'SEARCH_FOR_SUPER_SIGNAL:_fail',
      'SEARCH_FOR_SUPER_SIGNAL:s01',
      'EXIT_SIGNAL:s01',
      'ENTRY_SIGNAL:_fail',
      '\n|    FAIL!!!    |\n',
      'INIT_SIGNAL:_fail',
      '<- Queued:(0) Deferred:(0)']

  assert target == result
  time.sleep(0.1)

@pytest.mark.transition
def test_transition_send_cancellation_1():
  logging.shutdown()
  time.sleep(0.1)
  path = data_path / 'test208.scxml'
  xml_chart = XmlToMiros(path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(2.00)
  ao.stop()
  #result = get_log_as_stripped_string(data_path / 'test208.log')
  result = ao.spy()
  target = \
    ['START',
      'SEARCH_FOR_SUPER_SIGNAL:s0',
      'ENTRY_SIGNAL:s0',
      'INIT_SIGNAL:s0',
      '<- Queued:(0) Deferred:(0)',
      'event2:s0',
      'SEARCH_FOR_SUPER_SIGNAL:_pass',
      'SEARCH_FOR_SUPER_SIGNAL:s0',
      'EXIT_SIGNAL:s0',
      'ENTRY_SIGNAL:_pass',
      'INIT_SIGNAL:_pass',
      '<- Queued:(0) Deferred:(0)']
  assert target == result
  time.sleep(0.1)

@pytest.mark.transition
def test_transition_send_cancellation_2():
  logging.shutdown()
  time.sleep(0.1)
  path = data_path / 'test208_eventexpr.scxml'
  xml_chart = XmlToMiros(path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(2.00)
  ao.stop()
  result = ao.spy()
  target = \
    ['START',
      'SEARCH_FOR_SUPER_SIGNAL:s0',
      'ENTRY_SIGNAL:s0',
      'INIT_SIGNAL:s0',
      '<- Queued:(0) Deferred:(0)',
      'event2:s0',
      'SEARCH_FOR_SUPER_SIGNAL:_pass',
      'SEARCH_FOR_SUPER_SIGNAL:s0',
      'EXIT_SIGNAL:s0',
      'ENTRY_SIGNAL:_pass',
      'INIT_SIGNAL:_pass',
      '<- Queued:(0) Deferred:(0)']

  assert target == result
  time.sleep(0.1)

@pytest.mark.transition
def test_transition_send_cancellation_3():
  logging.shutdown()
  time.sleep(0.1)
  path = data_path / 'test208_eventexpr_delayexpr.scxml'
  xml_chart = XmlToMiros(path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(2.00)
  ao.stop()
  result = ao.spy()
  target = \
    ['START',
     'SEARCH_FOR_SUPER_SIGNAL:s0',
     'ENTRY_SIGNAL:s0',
     'INIT_SIGNAL:s0',
     '<- Queued:(0) Deferred:(0)',
     'event2:s0',
     'SEARCH_FOR_SUPER_SIGNAL:_pass',
     'SEARCH_FOR_SUPER_SIGNAL:s0',
     'EXIT_SIGNAL:s0',
     'ENTRY_SIGNAL:_pass',
     'INIT_SIGNAL:_pass',
     '<- Queued:(0) Deferred:(0)']

  assert target == result
  time.sleep(0.1)

@pytest.mark.transition
def test_transition_send_cancellation_all_1():
  logging.shutdown()
  time.sleep(0.1)
  path = data_path / 'test208_all.scxml'
  xml_chart = XmlToMiros(path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(2.00)
  ao.stop()
  target = ao.spy()
  result = \
    ['START',
      'SEARCH_FOR_SUPER_SIGNAL:s0',
      'ENTRY_SIGNAL:s0',
      'INIT_SIGNAL:s0',
      '<- Queued:(0) Deferred:(0)',
      'event2:s0',
      'SEARCH_FOR_SUPER_SIGNAL:_pass',
      'SEARCH_FOR_SUPER_SIGNAL:s0',
      'EXIT_SIGNAL:s0',
      'ENTRY_SIGNAL:_pass',
      'INIT_SIGNAL:_pass',
      '<- Queued:(0) Deferred:(0)']

  assert target == result
  time.sleep(0.1)

@pytest.mark.transition
def test_transition_send_cancellation_all_2():
  logging.shutdown()
  time.sleep(0.1)
  path = data_path / 'test208_eventexpr_all.scxml'
  xml_chart = XmlToMiros(path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(2.00)
  ao.stop()
  result = ao.spy()
  target = \
  ['START',
    'SEARCH_FOR_SUPER_SIGNAL:s0',
    'ENTRY_SIGNAL:s0',
    'INIT_SIGNAL:s0',
    '<- Queued:(0) Deferred:(0)',
    'event2:s0',
    'SEARCH_FOR_SUPER_SIGNAL:_pass',
    'SEARCH_FOR_SUPER_SIGNAL:s0',
    'EXIT_SIGNAL:s0',
    'ENTRY_SIGNAL:_pass',
    'INIT_SIGNAL:_pass',
    '<- Queued:(0) Deferred:(0)']

  assert target == result
  time.sleep(0.1)

@pytest.mark.transition
def test_transition_send_cancellation_all_3():
  logging.shutdown()
  time.sleep(0.1)
  path = data_path / 'test208_eventexpr_delayexpr_all.scxml'
  xml_chart = XmlToMiros(path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(2.00)
  ao.stop()
  result = ao.spy()
  target = \
   ['START',
     'SEARCH_FOR_SUPER_SIGNAL:s0',
     'ENTRY_SIGNAL:s0',
     'INIT_SIGNAL:s0',
     '<- Queued:(0) Deferred:(0)',
     'event2:s0',
     'SEARCH_FOR_SUPER_SIGNAL:_pass',
     'SEARCH_FOR_SUPER_SIGNAL:s0',
     'EXIT_SIGNAL:s0',
     'ENTRY_SIGNAL:_pass',
     'INIT_SIGNAL:_pass',
     '<- Queued:(0) Deferred:(0)']
  assert target == result
  time.sleep(0.1)
