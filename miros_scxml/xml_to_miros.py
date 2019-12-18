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

from miros import pp
from miros import Event
from miros import spy_on
from miros import signals
from miros import ActiveObject
from miros import return_status

class ScxmlChart(ActiveObject):
  def __init__(self, name):
    super().__init__(name)

class XmlToMiros():
  namespaces = {
    'sc': 'http://www.w3.org/2005/07/scxml',
    'xi': 'http://www.w3.org/2001/XInclude'
  }
  supported_tags = [
    'scxml',
    'state',
    'parallel',
    'transition',
    'initial',
    'final',
    'onentry',
    'onexit',
    'history',
    'raise',
    'if',
    'else',
    'elseif',
    'foreach',
    'log',
    'datamodel',
    'data',
    'assign',
    'donedata',
    'content',
    'param',
    'script',
    'send',
    'cancel',
    'invoke',
    'finalize'
  ]
  def __init__(self, path_to_xml):

    self.file_path = Path(path_to_xml).resolve()
    if isinstance(self.file_path, Path):
      file_path = str(self.file_path)
    else:
      file_path = self.file_path
      self.file_path = Path(self.file_path)

    self.tree = ElementTree.parse(file_path)
    self.root = self.tree.getroot()

    #self.state_chart = ScxmlChart(name=self.get_name())

    # We need to make a list of tags with the namespace prefix
    # so that we can spare our client from having to add the 
    # namespace garbage in their searches
    self.tags = list({elem.tag for elem in self.tree.iter()})
    self.tag_lookup = {}
    for tag in self.tags:
      name = tag.split('}')[-1]
      self.tag_lookup[name] = tag

    states = self.root.findall('sc:parallel', XmlToMiros.namespaces)

    self.find_states = partial(self.findall_multiple_tags, 
      ["{"+XmlToMiros.namespaces['sc']+"}state",
      "{"+XmlToMiros.namespaces['sc']+"}parallel",
      "{"+XmlToMiros.namespaces['sc']+"}final"])


    self.is_state = partial(self.is_tag, 'state')
    self.is_parallel = partial(self.is_tag, 'parallel')
    self.is_final = partial(self.is_tag, 'final')

    # Create functions which can find through the namespace garbage
    # self.findall_fn['onentry']
    # self.findall_fn['onexit']
    # ..
    # see XmlToMiros.supported_tags
    self.findall_fn = \
    { 
      name : partial(self._findall, "{"+XmlToMiros.namespaces['sc']+"}"+name) \
        for name \
        in XmlToMiros.supported_tags
    }


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
    self._code = self.write_code()


  def write_to_file(self, file_name):
    with open(str(file_name), "w") as fp:
      fp.write(self._code)

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

  def findall_multiple_tags(self, ns, node=None):
    nodes = []
    if node is None:
      node = self.root

    for arg in ns:
      nodes += node.findall(arg)
    return nodes

  def _findall(self, arg, node=None):
    return node.findall(arg)

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

    if 'initial' in node.attrib:
      state_dict['start_at'] = node.attrib['initial']
    else:
      state_dict['start_at'] = \
        self.find_states(node)[0].attrib['id']

    def index_for_signal(_list, signal_name):
      index = -1
      for i, item in enumerate(_list):
        if signal_name in item:
          index = i
          break
      return index

    def state_to_dict(node, parent):
      name = node.attrib['id']
      parent = None if parent == None else \
        self.get_tag_without_namespace(parent)
      state_dict[name] = {}
      state_dict[name]['p'] = parent
      state_dict[name]['cl'] = []
      state_dict[name]['cl'].append(
        {'ENTRY_SIGNAL': 'status = return_status.HANDLED'}
      )
      state_dict[name]['cl'].append(
        {'INIT_SIGNAL': 'status = return_status.HANDLED'}
      )
       
      state_dict[name]['cl'].append(
        {'SCXML_INIT_SIGNAL': 'status = return_status.HANDLED'}
      )
      state_dict[name]['cl'].append(
        {'EXIT_SIGNAL': 'status = return_state.HANDLED'}
      )

      def prepend_log(node, signal_name):
        log_nodes = self.findall_fn['log'](node)
        for log_node in log_nodes:
          string = "self.scribble(\'{}\')\n".format(
            self.escape_quotes(log_node.attrib['expr'])
            )
          index = index_for_signal(state_dict[name]['cl'], signal_name)
          string += state_dict[name]['cl'][index][signal_name]
          state_dict[name]['cl'][index] = {signal_name:string}

      entry_nodes = self.findall_fn['onentry'](node)
      for entry_node in entry_nodes:
        prepend_log(entry_node, 'ENTRY_SIGNAL')

      init_nodes = self.findall_fn['initial'](node)
      for init_node in init_nodes:
        prepend_log(init_node, 'INIT_SIGNAL')

      immediate_transition_nodes = self.findall_fn['transition'](node)
      for node in immediate_transition_nodes:
        code_string = "status = self.trans({})".format(node.attrib['target'])
        index = index_for_signal(state_dict[name]['cl'], 'SCXML_INIT_SIGNAL')
        state_dict[name]['cl'][index] = \
          {'SCXML_INIT_SIGNAL': code_string}
        prepend_log(node, 'SCXML_INIT_SIGNAL')

      exit_nodes = self.findall_fn['onexit'](node)
      for exit_node in exit_nodes:
        prepend_log(exit_node, 'EXIT_SIGNAL')

        #log_nodes = self.findall_fn['log'](entry_node)

        #for log_node in log_nodes:
        #  string = "self.scribble(\'{}\')\n".format(
        #    self.escape_quotes(log_node.attrib['expr'])
        #    )
        #  index = index_for_signal(state_dict[name]['cl'], 'ENTRY_SIGNAL')
        #  string += state_dict[name]['cl'][index]['ENTRY_SIGNAL']
        #  state_dict[name]['cl'][index] = {'ENTRY_SIGNAL':string}

    # recursively build the state_dict
    self.recurse_scxml(fn=state_to_dict)
    return state_dict

  @staticmethod
  def escape_quotes(string):
    _trans_dict = {
      "'": "\\'", 
      '"': '\\"'
    }
    result = string.translate(str.maketrans(_trans_dict)) 
    return result

  def write_code(self, indent_amount=None, custom_imports=None):

    if indent_amount is None:
      indent_amount = "  "

    start_code_template = \
'''
ao = ScxmlChart(name={})
ao.live_spy=True
ao.live_trace=True
ao.start_at({starting_state})
'''

    if custom_imports is None:
      imports = ""
    else:
      imports = "\n".split(custom_imports)

    pre_instantiation_template = """
import re
import time
import logging
from functools import partial
from collections import namedtuple

from miros import pp
from miros import Event
from miros import spy_on
from miros import signals
from miros import ActiveObject
from miros import return_status
{custom_imports}
{state_code}
"""
    instantiation_template = """
{pre_instantiation_code}
class ScxmlChart(ActiveObject):
{i}def __init__(self, name):
{i}{i}super().__init__(name)

ao = ScxmlChart(\"{name}\")
ao.live_spy = True
"""

    post_instantiation_code = """
{instantiation_code}
{start_code}
"""
    file_code = ""
    state_code = ""
    start_at = ""
    name = self.get_name()
    for (state_name, v) in self._state_dict.items():
      if state_name != 'start_at':
        state_code += self._write_state_code(state_name, v, indent_amount) + "\n"
      else:
        starting_state = v

    start_code = \
      "ao.start_at({starting_state})".format(starting_state=starting_state)

    pre_instantiation_code = \
      pre_instantiation_template.format(custom_imports=imports,
        state_code=state_code)
    instantiation_code = instantiation_template.format(
      pre_instantiation_code=pre_instantiation_code,
      name=name,
      i=indent_amount)
    code = post_instantiation_code.format(
      instantiation_code=instantiation_code,
      start_code=start_code)

    return code

  # {
  #   'cl': 
  #     [
  #       {'ENTRY_SIGNAL': 
  #         'self.scribble(\'Hello from \\"start\\"\')\nstatus = return_status.HANDLED'},
  #       {'INIT_SIGNAL': 'status = return_status.HANDLED'},
  #       {'SCXML_INIT_SIGNAL': 'status = self.trans(Work)'},
  #       {'EXIT_SIGNAL': 'status = return_state.HANDLED'}],
  #   'p': None
  # }

  @staticmethod
  def _write_state_code(state_name, state_dict, indent_amount=None):
    if indent_amount is None:
      indent_amount = "  "

    state_template = '''@spy_on
def {state_name}(self, e):
{i}status = return_status.UNHANDLED
{cls}{i}else:
{i}{i}self.temp.fun = {parent_state}
{i}{i}status = return_status.SUPER
{i}return status
'''
    first_cl_template = '''{i}if(e.signal == signals.{signal_name}):
{event_code}'''

    following_cl_template = '''{i}elif(e.signal == signals.{signal_name}):
{event_code}'''
    cls = ""
    signal_name, event_code = next(iter(state_dict.items()))
    for index, catch_dict in enumerate(state_dict['cl']):
      signal_name, event_code = next(iter(catch_dict.items()))
      
      tokens = event_code.split("\n")
      event_code = "{i}{i}{code}\n".format(i=indent_amount, code=tokens[0])
      new_tokens = ["{i}{i}{code}\n".format(i=indent_amount, code=token) for token in tokens[1:]]
      event_code += "\n".join(new_tokens)
      if index == 0:
        cls = first_cl_template.format(
          i=indent_amount,
          signal_name=signal_name,
          event_code= event_code)
      else:
        cls += following_cl_template.format(
          i=indent_amount,
          signal_name=signal_name,
          event_code= event_code)

    parent_state = "self.top" if state_dict['p'] is None else state_dict['p']
    state_code = state_template.format(
        i=indent_amount,
        state_name=state_name,
        cls=cls,
        parent_state=parent_state)
    return state_code

