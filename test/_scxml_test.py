import os
import pytest
from miros_scxml.xml_to_miros import XmlToMiros    
from pathlib import Path
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import ElementTree

import re
import time
import logging
from functools import partial
from collections import deque
from collections import namedtuple

from miros import Event
from miros import spy_on
from miros import signals
from miros import ActiveObject
from miros import return_status

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = Path(dir_path) / '..' / 'data'

@pytest.mark.skip()
@pytest.mark.scxml
def test_scxml_get_name():
  path = data_path / 'scxml_test_1.scxml'
  main = XmlToMiros(path)
  assert main.get_name() == "Scxml"  
  assert isinstance(main.state_chart, ActiveObject)

@pytest.mark.scxml
def test_scxml_confirm_initial_works_1():
  '''
  If there is no "initial" in the attrib of the scxml tag, then start in the
  first state/parellel region of the chart.  The XML used in this test is
  missing the "initial" attrib, so we are confirming that it will start in the
  'Start' state.
  '''
  state_dict = {}
  state_dict['Start'] = {}
  state_dict['Start']['p'] = None
  state_dict['Start']['cl'] = []
  state_dict['Start']['cl'].append(
    {'ENTRY_SIGNAL':"self.scribble('Hello from \'start\')"}
  )
  state_dict['Start']['cl'].append(
    {'INIT_SIGNAL':'status = return_state.HANDLED'}
  )
  state_dict['Start']['cl'].append(
    {'scxml_immediate': "self.trans(Work)"}
  )

  state_dict['Work'] = {}
  state_dict['Work']['p'] = None
  state_dict['Work']['cl'] = []
  state_dict['Work']['cl'].append(
    {'ENTRY_SIGNAL': "self.scribble('Hello from \'work\')"}
  )
  state_dict['Work']['cl'].append(
    {'INIT_SIGNAL':'status = return_state.HANDLED'}
  )
  state_dict['Work']['cl'].append(
    {'scxml_immediate':"status = return_state.HANDLED"}
  )
  state_dict['start_at'] = 'Start'

  path = data_path / 'scxml_test_1.scxml'
  xml_chart = XmlToMiros(path)
  assert 'Start' in xml_chart._state_dict
  assert 'Work' in xml_chart._state_dict
  result = False
  for _dict in xml_chart._state_dict['Start']['cl']:
    if 'ENTRY_SIGNAL' in _dict:
      result = True
      break;



