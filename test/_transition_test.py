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

#@pytest.mark.skip
@pytest.mark.transition
def test_transition_send_to_post_fifo():
  path = data_path / 'test403.scxml'
  #import pdb; pdb.set_trace()
  xml_chart = XmlToMiros(path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  result = get_log_as_stripped_string(data_path / 'test403.log')
  time.sleep(0.01)
  target = """
[Test403] START
[Test403] SEARCH_FOR_SUPER_SIGNAL:s0
[Test403] ENTRY_SIGNAL:s0
[Test403] POST_FIFO:SCXML_INIT_SIGNAL
[Test403] INIT_SIGNAL:s0
[Test403] SEARCH_FOR_SUPER_SIGNAL:s01
[Test403] ENTRY_SIGNAL:s01
[Test403] POST_FIFO:SCXML_INIT_SIGNAL
[Test403] INIT_SIGNAL:s01
[Test403] <- Queued:(2) Deferred:(0)
[Test403] SCXML_INIT_SIGNAL:s01
[Test403] SCXML_INIT_SIGNAL:s01:HOOK
[Test403] <- Queued:(1) Deferred:(0)
[Test403] SCXML_INIT_SIGNAL:s01
[Test403] SCXML_INIT_SIGNAL:s01:HOOK
[Test403] <- Queued:(0) Deferred:(0)
"""
  #assert target == result
  time.sleep(0.1)

