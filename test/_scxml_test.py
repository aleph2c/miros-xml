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

@pytest.mark.scxml
def test_scxml_get_name():
  path = data_path / 'scxml_test_1.scxml'
  miros_code_path = data_path / 'scxml_test_1.py'
  xml_chart = XmlToMiros(path, miros_code_path=miros_code_path)
  assert xml_chart.get_name() == "Scxml"  

@pytest.mark.scxml
def test_scxml_build_a_small_chart():
  path = data_path / 'scxml_test_1.scxml'
  xml_chart = XmlToMiros(path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(0.2)
  ao.stop()

  result = get_log_as_stripped_string(data_path / 'scxml_test_1.log')
  #print(result)

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

@pytest.mark.scxml
def test_scxml_build_a_small_chart():
  """
  Here we are demonstrating a barebones datamodel with data.  The data binding is
  "early", which means that the variables are initialized when the document is
  read, or there initialization information is put within the __init__ function of
  the ActiveObject.
  
  <scxml datamodel="python" name="Scxml" version="1.0" xmlns="http://www.w3.org/2005/07/scxml">
    <datamodel>
      <data expr="True" id="var_bool"/>
      <data expr="1" id="var_int"/> 
      <data expr="&quot;This is a string!&quot;" id="var_str"/>
      <data expr="[1, 2, 3, 4, 5]" id="var_list"/>
    </datamodel>
    <state id="Start">
      <onentry>
        <log expr='&quot;Hello from \&quot;start\&quot;&quot;'/>
        <log expr='"{} {}".format(var_bool, type(var_bool))'/>
        <log expr='"{} {}".format(var_int, type(var_int))'/>
        <log expr='"{} {}".format(var_str, type(var_str))'/>
        <log expr='"{} {}".format(var_list, type(var_list))'/>
      </onentry>
      <transition target="Work"/>
    </state>
    <state id="Work">
      <onentry>
        <log expr="&quot;Hello from \&quot;work\&quot;&quot;"/>
      </onentry>
    </state>
  </scxml>
  
  Becomes:
  --------
  import re
  import time
  import logging
  from pathlib import Path
  from functools import partial
  from collections import namedtuple
  from collections import OrderedDict
  
  from miros import pp
  from miros import Event
  from miros import spy_on
  from miros import signals
  from miros import ActiveObject
  from miros import return_status
  
  @spy_on
  def Start(self, e):
    status = return_status.UNHANDLED
    if(e.signal == signals.ENTRY_SIGNAL):
      self.post_fifo(Event(signal=signals.SCXML_INIT_SIGNAL))
      self.scribble("{} {}".format(self.var_list, type(self.var_list)))
      self.scribble("{} {}".format(self.var_str, type(self.var_str)))
      self.scribble("{} {}".format(self.var_int, type(self.var_int)))
      self.scribble("{} {}".format(self.var_bool, type(self.var_bool)))
      self.scribble("Hello from \"start\"")
      status = return_status.HANDLED
    elif(e.signal == signals.INIT_SIGNAL):
      status = return_status.HANDLED
    elif(e.signal == signals.SCXML_INIT_SIGNAL):
      status = self.trans(Work)
    elif(e.signal == signals.EXIT_SIGNAL):
      status = return_status.HANDLED
    else:
      self.temp.fun = self.top
      status = return_status.SUPER
    return status
  
  @spy_on
  def Work(self, e):
    status = return_status.UNHANDLED
    if(e.signal == signals.ENTRY_SIGNAL):
      self.scribble("Hello from \"work\"")
      status = return_status.HANDLED
    elif(e.signal == signals.INIT_SIGNAL):
      status = return_status.HANDLED
    elif(e.signal == signals.SCXML_INIT_SIGNAL):
      status = return_status.HANDLED
    elif(e.signal == signals.EXIT_SIGNAL):
      status = return_status.HANDLED
    else:
      self.temp.fun = self.top
      status = return_status.SUPER
    return status
  
  class InstrumentedActiveObject(ActiveObject):
    def __init__(self, name, log_file):
      super().__init__(name)
  
      self.log_file = log_file
  
      logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s',
        filemode='w',
        filename=self.log_file,
        level=logging.DEBUG)
      self.register_live_spy_callback(partial(self.spy_callback))
      self.register_live_trace_callback(partial(self.trace_callback))
  
    def trace_callback(self, trace):
      '''trace without datetimestamp'''
      trace_without_datetime = re.search(r'(\[.+\]) (\[.+\].+)', trace).group(2)
      logging.debug("T: " + trace_without_datetime)
  
    def spy_callback(self, spy):
      '''spy with machine name pre-pending'''
      print(spy)
      logging.debug("S: [%s] %s" % (self.name, spy))
  
    def clear_log(self):
      with open(self.log_file, "w") as fp:
        fp.write("I'm writing")
  
  class ScxmlChart(InstrumentedActiveObject):
    def __init__(self, name, log_file):
      super().__init__(name, log_file)
      self.var_bool = True
      self.var_int = 1
      self.var_str = "This is a string!"
      self.var_list = [1, 2, 3, 4, 5]
  
    def start(self):
      self.start_at(Start)
  
  if __name__ == '__main__':
    ao = ScxmlChart("Scxml", "/mnt/c/github/xml/data/scxml_test_3.log")
    ao.live_spy = True
    ao.start()
    time.sleep(0.01)
  
  """
  path = data_path / 'scxml_test_3.scxml'
  xml_chart = XmlToMiros(path)
  assert xml_chart.binding_type() == 'early'
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(0.2)
  ao.stop()

  result = get_log_as_stripped_string(data_path / 'scxml_test_3.log')
  result = ao.spy()
  target = \
  ['START',
    'SEARCH_FOR_SUPER_SIGNAL:Start',
    'ENTRY_SIGNAL:Start',
    'POST_FIFO:SCXML_INIT_SIGNAL',
    "[1, 2, 3, 4, 5] <class 'list'>",
    "This is a string! <class 'str'>",
    "1 <class 'int'>",
    "True <class 'bool'>",
    'Hello from "start"',
    'INIT_SIGNAL:Start',
    '<- Queued:(1) Deferred:(0)',
    'SCXML_INIT_SIGNAL:Start',
    'SEARCH_FOR_SUPER_SIGNAL:Work',
    'SEARCH_FOR_SUPER_SIGNAL:Start',
    'EXIT_SIGNAL:Start',
    'ENTRY_SIGNAL:Work',
    'Hello from "work"',
    'INIT_SIGNAL:Work',
    '<- Queued:(0) Deferred:(0)']


  assert(target == result)

@pytest.mark.scxml
def test_scxml_early_data_binding():
  path = data_path / 'early_binding.scxml'
  xml_chart = XmlToMiros(path)
  assert xml_chart.binding_type() == 'early'
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(0.2)
  ao.stop()
  result = get_log_as_stripped_string(data_path / 'early_binding.log')
  target = """
[Scxml] START
[Scxml] SEARCH_FOR_SUPER_SIGNAL:Step1
[Scxml] ENTRY_SIGNAL:Step1
[Scxml] POST_FIFO:SCXML_INIT_SIGNAL
[Scxml] 1
[Scxml] INIT_SIGNAL:Step1
[Scxml] <- Queued:(1) Deferred:(0)
[Scxml] SCXML_INIT_SIGNAL:Step1
[Scxml] SEARCH_FOR_SUPER_SIGNAL:Step2
[Scxml] SEARCH_FOR_SUPER_SIGNAL:Step1
[Scxml] EXIT_SIGNAL:Step1
[Scxml] ENTRY_SIGNAL:Step2
[Scxml] INIT_SIGNAL:Step2
[Scxml] <- Queued:(0) Deferred:(0)
"""
  assert(target == result)

@pytest.mark.scxml
def test_scxml_late_data_binding():
  path = data_path / 'late_binding.scxml'
  xml_chart = XmlToMiros(path)
  assert xml_chart.binding_type() == 'late'
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(0.2)
  ao.stop()
  result = get_log_as_stripped_string(data_path / 'late_binding.log')
  target = """
[Scxml] START
[Scxml] SEARCH_FOR_SUPER_SIGNAL:Step1
[Scxml] ENTRY_SIGNAL:Step1
[Scxml] POST_FIFO:SCXML_INIT_SIGNAL
[Scxml] None
[Scxml] INIT_SIGNAL:Step1
[Scxml] <- Queued:(1) Deferred:(0)
[Scxml] SCXML_INIT_SIGNAL:Step1
[Scxml] SEARCH_FOR_SUPER_SIGNAL:Step2
[Scxml] SEARCH_FOR_SUPER_SIGNAL:Step1
[Scxml] EXIT_SIGNAL:Step1
[Scxml] ENTRY_SIGNAL:Step2
[Scxml] INIT_SIGNAL:Step2
[Scxml] <- Queued:(0) Deferred:(0)
"""
  assert(target == result)

@pytest.mark.scxml
def test_scxml_default_init_state():
  path = data_path / 'test355.scxml'
  xml_chart = XmlToMiros(path)
  ao = xml_chart.make()  # like calling ScxmlChart(...)
  ao.live_spy = True
  ao.start()
  time.sleep(0.1)
  ao.stop()
  assert "pass" in ao.state_name
