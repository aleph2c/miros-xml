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

@pytest.mark.pinx
def test_anything_pinx():
  path = data_path / 'pinx_test_1.scxml'
  miros_code_path = data_path / 'pinx_test_1.py'
  xml_chart = XmlToMiros(path, miros_code_path=miros_code_path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  assert xml_chart.get_name() == "pinx"  

