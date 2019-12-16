import sys
from pathlib import Path
from xml.etree import ElementTree
from functools import partial

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

class ScmChart(ActiveObject):
  def __init__(self, name):
    super().__init__(name)

class XmlToMiros():
  namespaces = {
    'sc': 'http://www.w3.org/2005/07/scxml',
    'xi': 'http://www.w3.org/2001/XInclude'
    }

  def __init__(self, path_to_xml):

    self.file_path = Path(path_to_xml).resolve()
    if isinstance(self.file_path, Path):
      file_path = str(self.file_path)
    else:
      file_path = self.file_path
      self.file_path = Path(self.file_path)

    self.tree = ElementTree.parse(file_path)
    self.root = self.tree.getroot()

    self.state_chart = ScmChart(name=self.get_name())

    # We need to make a list of tags with the namespace prefix
    # so that we can spare our client from having to add the 
    # namespace garbage in their searches
    self.tags = list({elem.tag for elem in self.tree.iter()})
    self.tag_lookup = {}
    for tag in self.tags:
      name = tag.split('}')[-1]
      self.tag_lookup[name] = tag

    states = self.root.findall('sc:parallel', XmlToMiros.namespaces)

    self.find_states = partial(self.finder, 
      ["{"+XmlToMiros.namespaces['sc']+"}state",
      "{"+XmlToMiros.namespaces['sc']+"}parallel",
      "{"+XmlToMiros.namespaces['sc']+"}final"])


    self.is_state = partial(self.is_tag, 'state')
    self.is_parallel = partial(self.is_tag, 'parallel')
    self.is_final = partial(self.is_tag, 'final')

    self.recurse_scxml = partial(self.recurse, self.find_states)

    if sys.version_info < (3,9,0):
      def fn(node, parent):
        include_found, file_name, include_element = self.including_xml(node)
        if include_found:
          print("-------------------------------------------- {}".format(file_name))
          sub_element = ElementTree.parse(file_name).getroot()

          for elem in sub_element.iter():
            if elem.tag == 'xi':
              pass
            else:
              elem.tag = "{"+XmlToMiros.namespaces['sc']+"}" + elem.tag

          # add the subelement to the node
          node.append(sub_element)
          node.remove(include_element)
          a = 1
        
      # xi feature isn't fixed until 3.9 see https://bugs.python.org/issue20928 
      self.recurse_scxml(fn=fn)

    self._state_dict = self.build_statechart_dict(self.root)

  def findall(self, xpath, ns=None, node=None):
    '''find all subnodes of node given the xpath search parameter

    **Args**:
       | ``xpath`` (type1): xpath without namespace clutter
       | ``ns=None`` (type1): optional namespace
       | ``node=None`` (type1): the node to search, root if omitted


    **Returns**:
       (xml.etree.ElementTree.Element): result of search

    **Example(s)**:
      
    .. code-block:: python
       
       main = XmlToMiros('main.scxml')
       result = main.findall(.//state[@id='Test2']")

    '''
    if node is None:
      search_root = self.root

    if ns is None:
      # create a copy of our input
      _xpath = "{}".format(xpath)
      for tag_name_without_ns in self.tag_lookup.keys():
        # Does our origin search has a tag which is protected
        # by a namespace prefix? If so, add the namespace 
        # prefix to our internal version of that xpath
        if tag_name_without_ns in xpath:
          _xpath = "{}".format(_xpath.replace(tag_name_without_ns,
            self.tag_lookup[tag_name_without_ns]))
    else:
      _xpath = xpath

    return search_root.findall(_xpath)

  def finder(self, ns, node=None):
    nodes = []
    if node is None:
      node = self.root

    for arg in ns:
      nodes += node.findall(arg)
    return nodes

  def get_tag_without_namespace(self, node):
    return node.tag.split('}')[-1]

  def is_tag(self, arg, node):
    return True if self.get_tag_without_namespace(node) == arg else False

  def finder_in_depth(self, ns, node=None):
    elements = node.find('//'+ns)
    return elements

  def recurse(self, child_function=None, fn=None, node=None, parent=None):
    if child_function is None:
      child_function = self.find_states

    if node is None:
      node = self.root

    _parent = None
    if parent != None:
      for sibling in child_function(parent):
        if node.attrib['id'] == sibling.attrib['id']:
          _parent = node
          break;
      
    children = child_function(node)

    for child in children:
      
      if type(fn) == list:
        fns = fn
        [fn(child, _parent) for fn in fns]
      else:
        fn(child, _parent)

      self.recurse(
        child_function=child_function,
        fn=fn,
        node=child,
        parent=_parent)

  def get_name(self):
    return self.root.attrib['name']

  @staticmethod
  def including_xml(node):
    result, file_name, ie = False, None, None
    include_element = node.findall('{'+XmlToMiros.namespaces['xi']+'}include')
    if len(include_element) != 0:
      result = True
      file_name = include_element[0].attrib['href']
      ie = include_element[0]
    return result, file_name, ie

  def build_statechart_dict(self, node):
    state_dict = {}
    def state_to_dict(node, parent):
      name = node.attrib['id']
      parent = None if parent == None else \
        self.get_tag_without_namespace(parent)
      state_dict[name] = {}
      state_dict[name]['p'] = parent
      state_dict[name]['cl'] = []
      state_dict[name]['cl'].append({'ENTRY_SIGNAL': None})

    self.recurse_scxml(fn=state_to_dict)
    return state_dict


