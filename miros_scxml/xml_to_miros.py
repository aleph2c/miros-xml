# import re
import sys
# import time
import uuid
# import logging
import os as risky_os
from pathlib import Path
from functools import partial
# from collections import deque
from xml.etree import ElementTree
# from collections import namedtuple


# from miros import pp
# from miros import Event
# from miros import spy_on
# from miros import signals
# from miros import ActiveObject
# from miros import return_status

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
    'debug',
    # 'foreach',
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
  def __init__(self, path_to_xml, log_file=None, unique=None):

    self.debugger = False
    self.file_path = Path(path_to_xml).resolve()
    if isinstance(self.file_path, Path):
      file_path = str(self.file_path)
    else:
      file_path = self.file_path
      self.file_path = Path(self.file_path)

    if log_file is None:
      full_path = Path(path_to_xml)
      file_name = full_path.stem
      directory = full_path.resolve().parent
      self.log_file = str(directory / ("%s.log" % file_name))

    self.tree = ElementTree.parse(file_path)
    self.root = self.tree.getroot()

    # We want to ensure our ScxmlChart class name is unique hasn't been named
    self.chart_suffix = "" if unique is None else  str(uuid.uuid4())[0:8]
    self.scxml_chart_class  = 'ScxmlChart' + self.chart_suffix
    self.scxml_chart_superclass  = 'InstrumentedActiveObject' + self.chart_suffix

    # We need to make a list of tags with the namespace prefix
    # so that we can spare our client from having to add the 
    # namespace garbage in their searches
    self.tags = list({elem.tag for elem in self.tree.iter()})
    self.tag_lookup = {}
    for tag in self.tags:
      name = tag.split('}')[-1]
      self.tag_lookup[name] = tag

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

    # One level include feature forced into this package.
    # etree will be fixed in Python 3.9.0 but for now we have to manually add 
    # included XML
    if sys.version_info < (3,9,0):
      def fn(node, parent):
        include_found, file_name, include_element = self._including_xml(node)
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
        
      # xi feature isn't fixed until 3.9 see https://bugs.python.org/issue20928 
      self.recurse_scxml(fn=fn)

    self._state_dict = self.build_statechart_dict(self.root)
    self._code = self._create_miros_code()


  def make(self):
    '''Make the statechart

    **Note**:
       Executing scxml in your program is like running any other program.  If
       you let some outside party write code and run it on your machine, be away
       they have full access and they can do damage or take your secrets.

       If the <debug/> tag is included in a supported tag of the SCXML file,
       python debug code will be added in the same directory as the SCXML file
       and the debugger will be called in the location specified in the SCXML
       file.  The python debug file will be left, and you will have to manually
       delete it when you are done debugging your code.

    **Returns**:
       (ActiveObject): An instantiated instrumented statechart

    **Example(s)**:
      
    .. code-block:: python
       
       from pathlib import Path
       xml_chart = XmlToMiros(path, unique=True)
       ao = xml_chart.make()
       ao.start()  # to start the statechart (starts a thread)

    '''
    file_name, module_name, directory, statename = self._write_to_file()
    sys.path.append(directory)

    exec("from {module_name} import {statename} as ScxmlChart".format(
      module_name=module_name, statename=statename
      ), globals()
    )

    if not self.debugger:
      risky_os.remove(file_name)

    # If you run a linter it will say that this ScxmChart class is not
    # defined, but it is defined and imported in the above exec call.  Linters
    # aren't smart enough to 'see into' this kind of exec code.
    return ScxmlChart(name=self.get_name(), log_file=self.log_file)

  def _write_to_file(self, file_name=None, indent_amount=None):
    '''Write the SCXML chart written as miros python code to a file.

    **Note**:
       Do this not that recommendation

    **Args**:
       | ``file_name=None`` (Path|str): The file_name, defaults to the placeing
       |                                the resulting file with the same
       |                                basename of the SCXML file used in the
       |                                same directory that file was found.
       | ``indent_amount=None`` (str): How many indents you want in your python
       |                               file


    **Returns**:
       (tuple): (file_name, module_name, directory, statechart_class_name)

    **Example(s)**:
      
    .. code-block:: python
       
      file_name, module_name, directory, statename = self._write_to_file()
      sys.path.append(directory)

      exec("from {module_name} import {statename} as ScxmlChart".format(
        module_name=module_name, statename=statename
        ), globals()
      )

      if not self.debugger:
        os.remove(file_name)

      return ScxmlChart(name=self.get_name(), log_file=self.log_file)

    '''
    if indent_amount is None:
      indent_amount = "  "

    if file_name is None:
      path_to_xml = Path(self.file_path)
      module_base = path_to_xml.stem
      directory = path_to_xml.resolve().parent
      module_name = str("{}_{}".format(module_base, self.chart_suffix))
      file_name = str(
        directory / "{}.py".format(module_name)
      )

    directory = str(directory)

    class_code = self._code
    instantiation_code = self.instantiation_template().format(
      class_code=class_code,
      scxml_chart_class=self.scxml_chart_class,
      log_file=self.log_file,
      i=indent_amount,
      name=self.get_name())

    code = self.post_instantiation_template().format(
      i=indent_amount,
      instantiation_code=instantiation_code)

    with open(str(file_name), "w") as fp:
      fp.write(code)

    return file_name, module_name, directory, self.scxml_chart_class

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
    '''A base function to search for multiple tag types in XML, from which to
       build useful partial functions.

    **Note**:
       Not intended for direct use (partial function utility; see example)

    **Args**:
       | ``ns`` (dict): namespace dictionary
       | ``node=None`` (Element): The Element to search


    **Returns**:
       (list): list of Elements nodes which match the search

    **Example(s)**:
      
    .. code-block:: python
       
      self.find_states = partial(self.findall_multiple_tags, 
        ["{"+XmlToMiros.namespaces['sc']+"}state",
        "{"+XmlToMiros.namespaces['sc']+"}parallel",
        "{"+XmlToMiros.namespaces['sc']+"}final"])

    '''
    nodes = []
    if node is None:
      node = self.root

    for arg in ns:
      nodes += node.findall(arg)
    return nodes

  def _findall(self, arg, node=None):
    '''A partial template function to build other find functions from

    **Args**:
       | ``arg`` (str): Xpath search string
       | ``node=None`` (Element): XML element


    **Returns**:
       (list): list of matching child Element nodes

    **Example(s)**:
      
    .. code-block:: python
       
      self.findall_fn = \
      { 
        name : partial(self._findall, 
          "{"+XmlToMiros.namespaces['sc']+"}"+name) \
          for name \
          in XmlToMiros.supported_tags
      }
      log_nodes = self.findall_fn['log'](node)

    '''
    return node.findall(arg)

  def get_tag_without_namespace(self, node):
    '''Get the tag name without the namespace information prepended to it

    **Args**:
       | ``node`` (Element): The element to search


    **Returns**:
       (str): The tag name as a string

    '''
    return node.tag.split('}')[-1]

  def is_tag(self, arg, node):
    return True if self.get_tag_without_namespace(node) == arg else False

  def finder_in_depth(self, ns, node=None):
    elements = node.find('//'+ns)
    return elements

  def recurse(self, child_function=None, fn=None, node=None, parent=None):
    '''Recurse the XML structure with a custom search function and one or more
    customer worker functions.

    The search function is used to find specific nodes to act upon and the
    worker function(s) determine what to do once the node are found.

    **Args**:
       | ``child_function=None`` (fn): A function used to find the next batch of
       |                               children from the XML document
       | ``fn=None`` (list|fn):        A list of function or a function to run
       |                               on every node that matched against the
       |                               child_function in the XML document
       | ``node=None`` (Element):      An etree Element of an XML doc
       | ``parent=None`` (Element):    The parent node of the node

    **Example(s)**:
      
    .. code-block:: python
      
      # build a recursive function using the self.find_states technique
      self.recurse_scxml = partial(self.recurse, self.find_states)

      # create the statechart dictionary structure
      self.recurse_scxml(fn=state_to_dict)

    '''
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
    '''get the name of the Statechart from the provided SCXML document.'''
    return self.root.attrib['name']

  @staticmethod
  def _including_xml(node):
    '''Include one level of xml into the document if the xi:include tag is seen.

    **Note**:
       This function shouldn't exist in this library, it should be a part of the
       etree module.  Accounding to external docs, this include capability will
       be fixed in Python 3.9

    **Args**:
       | ``node`` (Element): node to check for includes

    **Returns**:
       (tuple): include_element(bool), file_name(str), include_element(element)


    **Example(s)**:
      
    .. code-block:: XML

      <state id="Test2" xmlns:xi="http://www.w3.org/2001/XInclude">
        <initial>
          <transition target="Test2Sub1"/>
        </initial>

        <!-- This time we reference a state 
             defined in an external file.   -->
         <xi:include href="data/Test2Sub1.xml" parse="text"/>

      </state>

    .. code-block:: python
       
      include_found, file_name, include_element = self._including_xml(node)

      if include_found:
        print("-------------------------------------------- {}".format(file_name))
        sub_element = ElementTree.parse(file_name).getroot()

        for elem in sub_element.iter():
          if elem.tag == 'xi':
            pass
          else:
            elem.tag = "{"+XmlToMiros.namespaces['sc']+"}" + elem.tag

      # add the sub-element to the node
      node.append(sub_element)
      node.remove(include_element)

    '''
    result, file_name, ie = False, None, None
    include_element = node.findall('{'+XmlToMiros.namespaces['xi']+'}include')
    if len(include_element) != 0:
      result = True
      file_name = include_element[0].attrib['href']
      ie = include_element[0]
    return result, file_name, ie

  def build_statechart_dict(self, node):
    '''Build a statechart dict from a given node

    **Args**:
       | ``node`` (Element): The element from which to begin building the
                             statechart dictionary


    **Returns**:
       (dict): The statechart dictionary

    **Example(s)**:
      
    .. code-block:: python
       
      self._state_dict = self.build_statechart_dict(self.root)

    '''
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

      def prepend_debugger(node, signal_name):
        log_nodes = self.findall_fn['debug'](node)
        for log_node in log_nodes:
          self.debugger = True
          string = "import pdb; pdb.set_trace()\n"
          index = index_for_signal(state_dict[name]['cl'], signal_name)
          string += state_dict[name]['cl'][index][signal_name]
          state_dict[name]['cl'][index] = {signal_name:string}

      entry_nodes = self.findall_fn['onentry'](node)
      for entry_node in entry_nodes:
        prepend_debugger(entry_node, 'ENTRY_SIGNAL')
        prepend_log(entry_node, 'ENTRY_SIGNAL')

      init_nodes = self.findall_fn['initial'](node)
      for init_node in init_nodes:
        prepend_debugger(entry_node, 'INIT_SIGNAL')
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

  def pre_instantiation_template(self):
    return """# autogenerated from {filepath}
import re
import time
import logging
from pathlib import Path
from functools import partial
from collections import namedtuple

from miros import pp
from miros import Event
from miros import spy_on
from miros import signals
from miros import ActiveObject
from miros import return_status
{custom_imports}
{state_code}"""

  def logging_class_definition_template(self):
    return """
{pre_instantiation_code}

class {scxml_chart_superclass}(ActiveObject):
{i}def __init__(self, name, log_file):
{i}{i}super().__init__(name)

{i}{i}self.log_file = log_file

#{i}{i}self.clear_log()

{i}{i}logging.basicConfig(
{i}{i}{i}format='%(asctime)s %(levelname)s:%(message)s',
{i}{i}{i}filemode='w',
{i}{i}{i}filename=self.log_file,
{i}{i}{i}level=logging.DEBUG)
{i}{i}self.register_live_spy_callback(partial(self.spy_callback))
{i}{i}self.register_live_trace_callback(partial(self.trace_callback))

{i}def trace_callback(self, trace):
{i}{i}'''trace without datetimestamp'''
{i}{i}trace_without_datetime = re.search(r'(\[.+\]) (\[.+\].+)', trace).group(2)
{i}{i}logging.debug("T: " + trace_without_datetime)

{i}def spy_callback(self, spy):
{i}{i}'''spy with machine name pre-pending'''
{i}{i}print(spy)
{i}{i}logging.debug("S: [%s] %s" % (self.name, spy))

{i}def clear_log(self):
{i}{i}with open(self.log_file, "w") as fp:
{i}{i}{i}fp.write("I'm writing")"""

  def class_definition_template(self):
    return """{logging_class_code}

class {scxml_chart_class}({scxml_chart_superclass}):
{i}def __init__(self, name, log_file):
{i}{i}super().__init__(name, log_file)

{i}def start(self):
{i}{i}self.start_at({starting_state})"""


  def instantiation_template(self):
    return """{class_code}

if __name__ == '__main__':
{i}ao = {scxml_chart_class}(\"{name}\", \"{log_file}\")
{i}ao.live_spy = True"""

  def post_instantiation_template(self):
    return """{instantiation_code}
{i}ao.start()"""

  def _create_miros_code(self, indent_amount=None, custom_imports=None):
    '''create the python code which can manifest the statechart

    **Args**:
       | ``indent_amount=None`` (str): A string of spaces
       | ``custom_imports=None`` (type1): A string of custome imports


    **Returns**:
       (str): Python code which can run a statechart.

    '''
    if indent_amount is None:
      indent_amount = "  "

    if custom_imports is None:
      imports = ""
    else:
      imports = "\n".split(custom_imports)

    state_code = ""

    for (state_name, v) in self._state_dict.items():
      if state_name != 'start_at':
        state_code += self._write_state_code(state_name, v, indent_amount) + "\n"
      else:
        starting_state = v

    pre_instantiation_code = \
      self.pre_instantiation_template().format(
          log_file=self.log_file,
          filepath=str(self.file_path),
          uuid=self.chart_suffix,
        custom_imports=imports,
        state_code=state_code)

    logging_class_code = self.logging_class_definition_template().format(
      file_path=str(self.file_path),
      scxml_chart_superclass=self.scxml_chart_superclass,
      scxml_chart_class=self.scxml_chart_class,
      pre_instantiation_code=pre_instantiation_code,
      i=indent_amount)

    class_code = self.class_definition_template().format(
      logging_class_code=logging_class_code,
      file_path=str(self.file_path),
      scxml_chart_superclass=self.scxml_chart_superclass,
      scxml_chart_class=self.scxml_chart_class,
      pre_instantiation_code=pre_instantiation_code,
      i=indent_amount,
      starting_state=starting_state)

    return class_code
  @staticmethod
  def _write_state_code(state_name, state_dict, indent_amount=None):
    if indent_amount is None:
      indent_amount = "  "

    state_template = '''
@spy_on
def {state_name}(self, e):
{i}status = return_status.UNHANDLED
{cls}{i}else:
{i}{i}self.temp.fun = {parent_state}
{i}{i}status = return_status.SUPER
{i}return status'''
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

