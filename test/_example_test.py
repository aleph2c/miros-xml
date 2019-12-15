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

def test_file_path_is_path():
  example1 = XmlToMiros(data_path / 'reed.xml')
  assert(isinstance(example1.file_path, Path))
  example2 = XmlToMiros(str(data_path / 'reed.xml'))
  assert(isinstance(example2.file_path, Path))

def test_it_can_open_and_parse_xml():
  example1 = XmlToMiros(data_path / 'reed.xml')
  assert(isinstance(example1.root, Element))

def test_it_can_find_states_final_and_parallel():
  example1 = XmlToMiros(data_path / 'microwave-02.scxml')

  assert(len(example1.find_states()) == 1)
  assert(example1.is_parallel(example1.find_states()[0]))

def test_recursive_function():
  path = data_path / 'microwave-02.scxml'
  example1 = XmlToMiros(path)
  def _print(node, parent):
    if parent is None:
      _parent = None
    else:
      _parent = example1.get_tag_without_namespace(parent)
    _node = example1.get_tag_without_namespace(node)
    print(_node, _parent)

  example1.recurse_scxml(fn=_print)

def test_findall_function():
  path = data_path / 'main.scxml'
  main = XmlToMiros(path)
  def _print(node, parent):
    if parent is None:
      _parent = None
    else:
      _parent = main.get_tag_without_namespace(parent)
    _node = main.get_tag_without_namespace(node)
    print(_node, _parent)
  test2 = main.findall(".//state[@id='Test2']")
  main.recurse_scxml(fn=_print, node=test2[0])

def test_findxi():
  path = data_path / 'main.scxml'
  main = XmlToMiros(path)
  #test2 = main.findall(".//state[@xmlns:xi='http://www.w3.org/2001/XInclude']")
  test2 = main.root.find("*")
  def _print(node, parent):
    for k,v in node.attrib.items():
      print(node, k, v)
      for element in node.iter():
        print("{} {}".format(element.tag, element.attrib))
  main.recurse_scxml(fn=_print, node=test2)
  print(test2)

@pytest.mark.sc
def test_signal_names():
  error = Event(signal="event.execution")
  assert(error.signal_name == "event.execution")


