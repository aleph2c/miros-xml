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

@pytest.mark.state
def test_initialize_transition_as_attribute():
  path = data_path / 'state_test_initial_as_attribute.scxml'
  xml_chart = XmlToMiros(path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  result = get_log_as_stripped_string(data_path / 'state_test_initial_as_attribute.log')
  target = """
[Scxml] START
[Scxml] SEARCH_FOR_SUPER_SIGNAL:Work
[Scxml] ENTRY_SIGNAL:Work
[Scxml] INIT_SIGNAL:Work
[Scxml] SEARCH_FOR_SUPER_SIGNAL:State1
[Scxml] ENTRY_SIGNAL:State1
[Scxml] POST_FIFO:SCXML_INIT_SIGNAL
[Scxml] Hello!
[Scxml] INIT_SIGNAL:State1
[Scxml] <- Queued:(1) Deferred:(0)
[Scxml] SCXML_INIT_SIGNAL:State1
[Scxml] SCXML_INIT_SIGNAL:State1:HOOK
[Scxml] <- Queued:(0) Deferred:(0)
"""
  assert target == result


@pytest.mark.state
def test_initialize_transition_as_tag():
  path = data_path / 'state_test_initial_as_tag.scxml'
  xml_chart = XmlToMiros(path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  result = get_log_as_stripped_string(data_path / 'state_test_initial_as_tag.log')
  target = """
[Scxml] START
[Scxml] SEARCH_FOR_SUPER_SIGNAL:Work
[Scxml] ENTRY_SIGNAL:Work
[Scxml] INIT_SIGNAL:Work
[Scxml] SEARCH_FOR_SUPER_SIGNAL:State1
[Scxml] ENTRY_SIGNAL:State1
[Scxml] Hello!
[Scxml] INIT_SIGNAL:State1
[Scxml] SEARCH_FOR_SUPER_SIGNAL:State2
[Scxml] ENTRY_SIGNAL:State2
[Scxml] POST_FIFO:SCXML_INIT_SIGNAL
[Scxml] Hello!
[Scxml] INIT_SIGNAL:State2
[Scxml] Illegal in standard but I'm going to allow it
[Scxml] <- Queued:(1) Deferred:(0)
[Scxml] SCXML_INIT_SIGNAL:State2
[Scxml] SCXML_INIT_SIGNAL:State2:HOOK
[Scxml] <- Queued:(0) Deferred:(0)
"""
  assert target == result


@pytest.mark.state
def test_initialize_error_if_in_attribute_of_atomic_state():
  path = data_path / \
    "state_test_error_if_initial_as_attribute_in_atomic_state.scxml"
  xml_chart = XmlToMiros(path, miros_code_path=True)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  with pytest.raises(miros.hsm.HsmTopologyException):
    ao.start()
  assert (data_path /
  "state_test_error_if_initial_as_attribute_in_atomic_state_.py").exists()
