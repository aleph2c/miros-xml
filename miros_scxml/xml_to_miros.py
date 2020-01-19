import re
import sys
import uuid
import time
import os as risky_os
from pathlib import Path
from functools import partial
from xml.etree import ElementTree
from collections import OrderedDict

from miros import pp
from miros.event import signals as miros_signals

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

  def __init__(self, path_to_xml, log_file=None, miros_code_path=None,
      unique=None, indent_amount=None):
    '''Convert scxml into a miro python statechart

    **Args**:
       | ``path_to_xml`` (path|str): location of the scxml file
       | ``log_file=None`` (str): name of the log file generated from your
       |                          statechart
       | ``miros_code_path=None`` (path|str|bool):  Keep the resulting miros python file.  
       |                          If a path/string is specified, keep the file there.
       |                          If miros_code_path is None, do not keep the file.
       | ``unique=None`` (bool):  to create a unique namespace for the generated
       |                          statechart objects


    **Returns**:
       (XmlToMiros): an object of this class

    **Example(s)**:
      
    .. code-block:: python
       
       path = data_path / 'scxml_test_3.scxml'
       xml_chart = XmlToMiros(path)
       ao = xml_chart.make()  # like calling ScxmlChart(...)
       ao.live_spy = True
       ao.start()
       time.sleep(0.1)

    '''
    self.debugger = False
    self.file_path = Path(path_to_xml).resolve()
    if isinstance(self.file_path, Path):
      file_path = str(self.file_path)
    else:
      file_path = self.file_path
      self.file_path = Path(self.file_path)

    self.keep_code = False if miros_code_path is None else True

    # keys are xml ids and values are miros ids
    self.send_id_dict = {}

    if indent_amount is None:
      self.indent_amount = "  "
    else:
      self.indent_amount = indent_amount

    if isinstance(miros_code_path, Path) or isinstance(miros_code_path, str):
      self.python_file_name = miros_code_path
    else:
      self.python_file_name = None

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

    self.find_states = partial(self._x_findall_multiple_tags, 
      ["{"+XmlToMiros.namespaces['sc']+"}state",
      "{"+XmlToMiros.namespaces['sc']+"}parallel",
      "{"+XmlToMiros.namespaces['sc']+"}final"])

    self.is_state = partial(self._x_is_tag, 'state')
    self.is_parallel = partial(self._x_is_tag, 'parallel')
    self.is_final = partial(self._x_is_tag, 'final')

    # Create functions which can find through the namespace garbage
    # self.findall_fn['onentry']
    # self.findall_fn['onexit']
    # ..
    # see XmlToMiros.supported_tags
    self.findall_fn = \
    { 
      name : partial(self._x_findall, "{"+XmlToMiros.namespaces['sc']+"}"+name) \
        for name \
        in XmlToMiros.supported_tags
    }

    self.recurse_scxml = partial(self.recurse, self.find_states)

    # One level include feature forced into this package.
    # etree will be fixed in Python 3.9.0 but for now we have to manually add 
    # included XML
    if sys.version_info < (3,9,0):
      def fn(node, parent):
        include_found, file_name, include_element = self._x_include_xml(node)
        if include_found:
          print("-------------------------------------------- {}".format(file_name))
          sub_element = ElementTree.parse(file_name).getroot()

          for elem in sub_element.iter():
            if elem.tag == 'xi':
              pass
            else:
              # TODO: fix this weird way of using namespaces when you have a
              # test for this feature
              elem.tag = "{"+XmlToMiros.namespaces['sc']+"}" + elem.tag

          # add the subelement to the node
          node.append(sub_element)
          node.remove(include_element)
        
      # xi feature isn't fixed until 3.9 see https://bugs.python.org/issue20928 
      self.recurse_scxml(fn=fn)

    # Build up the data dictionary structure which will be uses as a flat
    # reference from which to write the miros code
    #{ 'states': {
    #     'Start': 
    #      {
    #        'cl': 
    #        [
    #          {'ENTRY_SIGNAL':
    #             'self.post_fifo(Event(signal=signals.SCXML_INIT_SIGNAL))\n'
    #             'self.scribble(\'Hello from \\"start\\"\')\n'
    #             'self.scribble(\"{} {}\".format(self.var_bool, type(self.var_bool)))\n'
    #             'self.scribble(\"{} {}\".format(self.var_int, type(self.var_int)))\n'
    #             'self.scribble(\"{} {}\".format(self.var_str, type(self.var_str)))\n'
    #             'self.scribble(\"{} {}\".format(self.var_list, type(self.var_list)))\n'
    #             'self.scribble(\"{} {}\".format(self.var_dict, type(self.var_dict)))\n'
    #             'status = return_status.HANDLED'
    #          },
    #          {'INIT_SIGNAL':
    #            'status = return_status.HANDLED'
    #          },
    #          {'SCXML_INIT_SIGNAL':
    #            'status = self.trans(Work)'
    #          },
    #          {'EXIT_SIGNAL':
    #            'status = return_status.HANDLED'}
    #       ],
    #       'p': None
    #       '*': False
    #      },
    #     'Work':
    #     {
    #       'cl':
    #         [
    #           {'ENTRY_SIGNAL':
    #              "self.scribble('Hello from \\'work\\'')\n"
    #              'status = return_status.HANDLED'
    #           },
    #           {'INIT_SIGNAL':
    #             'status = return_status.HANDLED'
    #           },
    #           {'SCXML_INIT_SIGNAL':
    #             'status = return_status.HANDLED'
    #           },
    #           {'EXIT_SIGNAL':
    #             'status = return_status.HANDLED'
    #           }
    #        ],
    #      'p': None
    #       '*': False
    #     },
    #  }
    #  'start_at': 'Start'
    #  'datamodel' : 'python',
    #  'data' :
    #    [
    #      "self.var_bool = True"
    #      "self.var_int = 1"
    #      "self.var_str = \"This is a string\""
    #      "self.var_list = [1, 2, 3, 4, 5]"
    #      "self.var_dict = {\"key_1\":\"value_1\", \"key_2\":\"value_2\"}"
    #    ],
    #   'early_binding' : True
    #}
    self._state_chart_dict = self._d_initialize_state_chart_dict()
    self._state_chart_dict['datamodel'] = self._d_build_datamodel(self.root)
    self._state_chart_dict['early_binding'] = self._d_datamodel_binding_type(self.root)
    self._state_chart_dict['data'], self.datamodel_variables = self._d_build_data(self.root)
    self._state_chart_dict['states'] = self._d_build_statechart_dict(self.root)
    pp(self._state_chart_dict['states'])
    self._states_dict = {}
    self._code = self._sc_create_miros_code()

  def _d_build_datamodel(self, node):
    if 'datamodel' in node.attrib:
      datamodel = node.attrib['datamodel']
      assert datamodel == "python"
    else:
      datamodel = None
    return datamodel

  def _d_datamodel_binding_type(self, node):
    early_binding = True
    if 'binding' in node.attrib:
      binding_type = node.attrib['binding']
      assert binding_type == "early" or binding_type == "late"
      early_binding = True if binding_type == "early" else False
    return early_binding

  def binding_type(self):
    early_binding = self._state_chart_dict['early_binding']
    return 'early' if early_binding else 'late'

  def _d_build_data(self, node):
    # 'data' :
    #   [
    #     "self.var_bool = True"
    #     "self.var_int = 1"
    #     "self.var_str = \"This is a string\""
    #     "self.var_list = [1, 2, 3, 4, 5]"
    #     "self.var_dict = {\"key_1\":\"value_1\", \"key_2\":\"value_2\"}"
    #   ]
    variable_list = [] 
    data_list = []
    data_nodes = node.findall(".//sc:datamodel/sc:data", XmlToMiros.namespaces)

    for data in data_nodes:
      variable_list.append(data.attrib['id'])
      variable_name = "self." + data.attrib['id']
      if self.binding_type() == "early":
        contents = data.attrib['expr'].replace("quot;", '\"')
        data_list.append("{} = {}".format(variable_name, contents))
      else:
        data_list.append("{} = None".format(variable_name))

    return data_list, variable_list
    

  def _d_initialize_state_chart_dict(self):
    #{ 'states': {
    #     'Start': 
    #      {
    #        'cl': 
    #        [
    #          {'ENTRY_SIGNAL':
    #             'self.post_fifo(Event(signal=signals.SCXML_INIT_SIGNAL))\n'
    #             'self.scribble(\'Hello from \\"start\\"\')\n'
    #             'self.scribble(\"{} {}\".format(self.var_bool, type(self.var_bool)))\n'
    #             'self.scribble(\"{} {}\".format(self.var_int, type(self.var_int)))\n'
    #             'self.scribble(\"{} {}\".format(self.var_str, type(self.var_str)))\n'
    #             'self.scribble(\"{} {}\".format(self.var_list, type(self.var_list)))\n'
    #             'self.scribble(\"{} {}\".format(self.var_dict, type(self.var_dict)))\n'
    #             'status = return_status.HANDLED'
    #          },
    #          {'INIT_SIGNAL':
    #            'status = return_status.HANDLED'
    #          },
    #          {'SCXML_INIT_SIGNAL':
    #            'status = self.trans(Work)'
    #          },
    #          {'EXIT_SIGNAL':
    #            'status = return_status.HANDLED'}
    #       ],
    #       'p': None
    #       '*': False
    #      },
    #      'Work':
    #      {
    #        'cl':
    #        [
    #            {'ENTRY_SIGNAL':
    #               "self.scribble('Hello from \\'work\\'')\n"
    #               'status = return_status.HANDLED'
    #            },
    #            {'INIT_SIGNAL':
    #              'status = return_status.HANDLED'
    #            },
    #            {'SCXML_INIT_SIGNAL':
    #              'status = return_status.HANDLED'
    #            },
    #            {'EXIT_SIGNAL':
    #              'status = return_status.HANDLED'
    #            }
    #         ],
    #         'p': None
    #         '*': False
    #      },
    #  }
    #  'start_at': 'Start'
    #  'datamodel' : 'python',
    #  'data' :
    #    [
    #      "self.var_bool = True"
    #      "self.var_int = 1"
    #      "self.var_str = \"This is a string\""
    #      "self.var_list = [1, 2, 3, 4, 5]"
    #      "self.var_dict = {\"key_1\":\"value_1\", \"key_2\":\"value_2\"}"
    #    ],
    #   'early_binding' : True,
    #}
    state_chart_dict = {}
    state_chart_dict['name'] = None
    state_chart_dict['states'] = {}
    state_chart_dict['start_at'] = None
    state_chart_dict['datamodel'] = None
    state_chart_dict['data'] = []
    state_chart_dict['early_binding'] = True
    return state_chart_dict

  @staticmethod
  def index_for_signal(_list, signal_name):
    index = -1
    for i, item in enumerate(_list):
      if signal_name in item:
        index = i
        break
    return index

  def _d_escape_quotes(self, string):
    _trans_dict = {
      "'": "\\'", 
      '"': '\\"'
    }
    result = string.translate(str.maketrans(_trans_dict)) 
    return result

  def _d_prepend_log(self, name, node, signal_name):
    '''convert <log>{contents}</log> to self.scribble({contents})'''
    log_nodes = self.findall_fn['log'](node)
    for log_node in log_nodes:
      log_type = None
      if self._state_chart_dict['datamodel'] == 'python' and 'type' in log_node.attrib:
        if log_node.attrib['type'] == 'code':
          log_type = 'code'
        else:
          log_type = 'string'

      if log_type == 'code':
        for datamodel_variable in self.datamodel_variables:
          expression = expression.replace(datamodel_variable, "self." + datamodel_variable)
        string = "self.scribble({})\n".format(expression)

      expression = log_node.attrib['expr']
      if log_type == 'code':
        for datamodel_variable in self.datamodel_variables:
          expression = expression.replace(datamodel_variable, "self." + datamodel_variable)
        string = "self.scribble({})\n".format(expression)
      else:
        string = "self.scribble(\'{}\')\n".format(
          self._d_escape_quotes(log_node.attrib['expr'])
          )
      index = self.index_for_signal(self._states_dict[name]['cl'], signal_name)
      string += self._states_dict[name]['cl'][index][signal_name]
      self._states_dict[name]['cl'][index] = {signal_name:string}

  def _d_prepend_debugger(self, name, node, signal_name):
    '''convert <debugger\> to import pdb; pdb.set_trace()'''
    log_nodes = self.findall_fn['debug'](node)
    for log_node in log_nodes:
      self.debugger = True
      string = "import pdb; pdb.set_trace()\n"
      index = self.index_for_signal(self._states_dict[name]['cl'], signal_name)
      string += self._states_dict[name]['cl'][index][signal_name]
      self._states_dict[name]['cl'][index] = {signal_name:string}

  def _d_inject_payload(self, name, node, signal_name, event_string):
    new_string = event_string
    param_nodes = self.findall_fn['param'](node)
    for param in param_nodes:
      if param.attrib['name'] != 'payload':
        raise NameError('library only supports param names called "payload"')
      right = param.attrib['expr']
      if self._state_chart_dict['datamodel'] == 'python':
        for datamodel_variable in self.datamodel_variables:
          right = right.replace(datamodel_variable, "self." + datamodel_variable)
          right = right.replace("self.self.", "self.")
      if right != None and len(right) != 0:
        new_string = re.sub(
          r'(Event.+?)\)',
          r'\1, payload={code})'.format(code=right),
          event_string
        )
    return new_string

  def _d_prepend_shooter(self, name, node, signal_name):
    send_nodes = self.findall_fn['send'](node)
    for send_node in send_nodes:
      sendid = None
      if 'id' in send_node.attrib:
        sendid = send_node.attrib['id']

      if 'delay' in send_node.attrib or 'delayexpr' in send_node.attrib:

        if 'event' in send_node.attrib and \
          'eventexpr' in send_node.attrib:
          raise ValueError("you can not specify both a 'event' and 'eventexpr' as attributes of the <send> tag")

        if 'delay' in send_node.attrib and \
          'delayexpr' in send_node.attrib:
          raise ValueError("you can not specify both a 'delay' and 'delayexpr' as attributes of the <send> tag")

        if 'delay' in send_node.attrib:
          time_in_seconds = self._sc_get_time(send_node.attrib['delay'])
        if 'event' in send_node.attrib:
          if sendid:
            eventexpr = "self.post_fifo_with_sendid(\"{sendid}\", ".format(sendid=sendid)
          else:
            eventexpr = "self.post_fifo("
          send_signal_name = send_node.attrib['event']
          string = """{eventexpr}Event(signal=\"{signal_name}\"),
{i}times=1,
{i}period={time_in_seconds},
{i}deferred=True)\n""".format(
            eventexpr=eventexpr,
            i=self.indent_amount,
            signal_name=send_signal_name,
            time_in_seconds=time_in_seconds)
        elif 'eventexpr' in send_node.attrib and 'delay' in send_node.attrib:
          eventexpr = send_node.attrib['eventexpr'][0:-1]
          if sendid:
            eventexpr = eventexpr.replace("post_fifo(",
              "self.post_fifo_with_sendid(\"{sendid}\", ".format(sendid=sendid))
            eventexpr = eventexpr.replace("post_lifo(",
              "self.post_lifo_with_sendid(\"{sendid}\", ".format(sendid=sendid))
          else:
            eventexpr = eventexpr.replace("post_fifo(", "self.post_fifo(")
            eventexpr = eventexpr.replace("post_lifo(", "self.post_lifo(")
          string = """{eventexpr},
{i}times=1,
{i}period={time_in_seconds},
{i}deferred=True)\n""".format(
            i=self.indent_amount,
            eventexpr=eventexpr,
            time_in_seconds=time_in_seconds)
        elif 'eventexpr' in send_node.attrib and 'delayexpr' in send_node.attrib:
          eventexpr = send_node.attrib['eventexpr'][0:-1]
          if sendid:
            eventexpr = eventexpr.replace("post_fifo(",
              "self.post_fifo_with_sendid(\"{sendid}\", ".format(sendid=sendid))
            eventexpr = eventexpr.replace("post_lifo(",
              "self.post_lifo_with_sendid(\"{sendid}\", ".format(sendid=sendid))
          else:
            eventexpr = eventexpr.replace("post_fifo(", "self.post_fifo(")
            eventexpr = eventexpr.replace("post_lifo(", "self.post_lifo(")
          delayexpr = send_node.attrib['delayexpr']
          delayexpr = delayexpr.replace('delay', 'period')
          string = """{eventexpr}, {delayexpr})\n""".format(
            i=self.indent_amount,
            eventexpr=eventexpr,
            delayexpr=delayexpr)
      else:
        eventexpr = "self.post_fifo("
        string = \
          """{eventexpr}Event(signal=\"{signal_name}\"))\n""".format(
            eventexpr=eventexpr,
            signal_name=send_node.attrib['event'])

      index = self.index_for_signal(self._states_dict[name]['cl'],
        signal_name)
      print('---')
      string = self._d_inject_payload(name, send_node, signal_name, string)
      string += self._states_dict[name]['cl'][index][signal_name]
      self._states_dict[name]['cl'][index] = {signal_name:string}

  def _sc_get_time(self, string):
    time_in_seconds = 0
    if string[-1] == 's':
      time_in_seconds = float(string[0:-1])
    elif string[-1] == 'm':
      time_in_seconds = 60.0 * float(string[0:-1])
    elif string[-1] == 'h':
      time_in_seconds = 60.0 * 60.0 * float(string[0:-1])
    elif string[-1] == 'd':
      time_in_seconds = 60.0 * 60.0 * 24 * float(string[0:-1])
    else:
      try:
        time_in_seconds = float(string)
      except:
        raise NotImplementedError("time format not implemented")
    return time_in_seconds

  def _d_prepend_canceller(self, name, node, signal_name):
    cancel_nodes = self.findall_fn['cancel'](node)

    for cancel_node in cancel_nodes:
      if 'sendid' in cancel_node.attrib and \
         'sendexpr' in cancel_node.attrib:
        exception_message = "you can not specify both a 'sendid' and 'sendexpr' "
        exception_message += "as attributes of the <cancel> tag"
        raise ValueError(exception_message)

      if 'sendid' in cancel_node.attrib:
        sendid = cancel_node.attrib['sendid']
        string ="self.cancel_with_sendid(sendid=\"{sendid}\")\n".format(sendid=sendid)
      elif 'sendexpr' in cancel_node.attrib:
        sendexpr = cancel_node.attrib['sendexpr']
        string = "self.{sendexpr}\n".format(sendexpr=sendexpr)
      else:
        exception_message = "either 'sendid' of 'sendexpr' required in <cancel> tag"
        raise ValueError(exception_message)


      index = self.index_for_signal(self._states_dict[name]['cl'], signal_name)
      string += self._states_dict[name]['cl'][index][signal_name]
      self._states_dict[name]['cl'][index] = {signal_name:string}

  def _d_prepend_posting_of_scxml_init_in_entry_condition(self, name, node):
    '''add self.post_fifo(Event(signal=signals.SCXML_INIT_SIGNAL)) to the
    entry condition when the SCXML requires an immediate transition from one
    state to another'''
    entry_index = self.index_for_signal(self._states_dict[name]['cl'], "ENTRY_SIGNAL")
    string = "self.post_fifo(Event(signal=signals.SCXML_INIT_SIGNAL))\n"
    string += self._states_dict[name]['cl'][entry_index]["ENTRY_SIGNAL"]
    self._states_dict[name]['cl'][entry_index] = {"ENTRY_SIGNAL":string}

  # pull these methods into their own class?
  # initialize the state dict
  # draw a picture of what to expect in a comment
  # break out the state to dict function
  # break out and clean up the prepend functions

  def _d_build_statechart_dict(self, node):
    if 'initial' in node.attrib:
      self._state_chart_dict['start_at'] = node.attrib['initial']
    else:
      self._state_chart_dict['start_at'] = \
        self.find_states(node)[0].attrib['id']

    # recursively build the state_dict
    self._states_dict = OrderedDict()
    self.recurse_scxml(fn=partial(self._d_state_to_dict))
    return self._states_dict

  def _d_state_to_dict(self, node, parent):
    name = node.attrib['id']
    _parent = None if parent == None else \
      parent.attrib['id']

    self._states_dict[name] = {}
    self._states_dict[name]['p'] = _parent
    self._states_dict[name]['cl'] = []

    # globs are searched for in _d_transition
    self._states_dict[name]['*'] = False

    self._d_entry(name, node, parent)
    self._d_init(name, node, parent)
    self._d_scxml_init(name, node, parent)
    self._d_exit(name, node, parent)
    self._d_transition(name, node, parent)

    return self._states_dict

  def _d_entry(self, name, node, parent):
    self._states_dict[name]['cl'].append(
      {'ENTRY_SIGNAL': 'status = return_status.HANDLED'}
    )
    entry_nodes = self.findall_fn['onentry'](node)
    for entry_node in entry_nodes:
      self._d_prepend_canceller(name, entry_node, 'ENTRY_SIGNAL')
      self._d_prepend_debugger(name, entry_node, 'ENTRY_SIGNAL')
      self._d_prepend_log(name, entry_node, 'ENTRY_SIGNAL')
      self._d_prepend_shooter(name, entry_node, 'ENTRY_SIGNAL')
      self._d_prepend_assign(name, entry_node, 'ENTRY_SIGNAL')

  def _d_init(self, name, node, parent):
    self._states_dict[name]['cl'].append(
      {'INIT_SIGNAL': 'status = return_status.HANDLED'}
    )

    if 'initial' in node.attrib:
      expression  = "status = self.trans({})".format(node.attrib['initial'])
      index = self.index_for_signal(self._states_dict[name]['cl'], 'INIT_SIGNAL')
      self._states_dict[name]['cl'][index] = {'INIT_SIGNAL':expression}

    init_nodes = self.findall_fn['initial'](node)
    for init_node in init_nodes:
      index = self.index_for_signal(self._states_dict[name]['cl'], 'INIT_SIGNAL')
      if self._states_dict[name]['cl'][index]['INIT_SIGNAL'] != 'status = return_status.HANDLED':
        raise SyntaxError('"initial" keyword can be used as a tag or attribute, but not both')
      transition = init_node.findall('./sc:transition', XmlToMiros.namespaces)

      # TODO: condition here
      if len(transition) > 0:
        target = transition[0].attrib['target']
        expression = "status = self.trans({})".format(target)
        self._states_dict[name]['cl'][index]['INIT_SIGNAL'] = expression
      self._d_prepend_canceller(name, init_node, 'INIT_SIGNAL')
      self._d_prepend_debugger(name, init_node, 'INIT_SIGNAL')
      self._d_prepend_log(name, init_node, 'INIT_SIGNAL')
      self._d_prepend_shooter(name, init_node, 'INIT_SIGNAL')
      self._d_prepend_assign(name, init_node, 'INIT_SIGNAL')

  def _d_prepend_assign(self, name, node, signal_name):
    '''convert <assign>location="{contents1}'" expr="{content2}"</assign> to
    self.{contents1} = {contenxt2}'''
    log_nodes = self.findall_fn['assign'](node)
    for assign_node in log_nodes:
      assign_type = None
      if self._state_chart_dict['datamodel'] != 'python':
        raise NameError('library only supports assign with python datamodel')

      left = assign_node.attrib['location']
      right = assign_node.attrib['expr']

      for datamodel_variable in self.datamodel_variables:
        left = left.replace(datamodel_variable, "self." + datamodel_variable)
        right = right.replace(datamodel_variable, "self." + datamodel_variable)

      string = "{left} = {right}\n".format(left=left, right=right)

      index = self.index_for_signal(self._states_dict[name]['cl'], signal_name)
      string += self._states_dict[name]['cl'][index][signal_name]
      self._states_dict[name]['cl'][index] = {signal_name:string}

  def _d_scxml_init(self, name, node, parent):
    # this can be over-written by a transition, see _d_transition
    self._states_dict[name]['cl'].append(
      {'SCXML_INIT_SIGNAL': 'status = return_status.HANDLED'}
    )

  def _d_exit(self, name, node, parent):
    self._states_dict[name]['cl'].append(
      {'EXIT_SIGNAL': 'status = return_status.HANDLED'}
    )
    exit_nodes = self.findall_fn['onexit'](node)
    for exit_node in exit_nodes:
      self._d_prepend_canceller(name, exit_node, 'EXIT_SIGNAL')
      self._d_prepend_log(name, exit_node, 'EXIT_SIGNAL')
      self._d_prepend_shooter(name, exit_node, 'EXIT_SIGNAL')
      self._d_prepend_assign(name, exit_node, 'EXIT_SIGNAL')

  def _d_transition(self, name, node, parent):
    transition_nodes = self.findall_fn['transition'](node)

    # globs handled here
    self._states_dict[name]['*'] = False
    for transition in transition_nodes:
      if 'event' in transition.attrib:
        event_type = transition.attrib['event']
        code = "status = self.trans({})".format(transition.attrib['target'])
        if 'cond' in transition.attrib:
          code = 'status = return_status.HANDLED\n'
          if 'event.' in transition.attrib['cond'] or 'e.' in transition.attrib['cond']:
            snippet = transition.attrib['cond'].replace('event.', 'e.')
          else:
            snippet = 'self.' + transition.attrib['cond']
          code += "if({}):\n".format(snippet) 
          code += self.indent_amount + "status = self.trans({})".format(transition.attrib['target'])
        self._states_dict[name]['cl'].append(
          {event_type: code}
        )
        self._states_dict[name]['*'] |= False if event_type != '*' else True
        self._d_prepend_assign(name, transition, name)

    # manage SCXML_INIT_SIGNAL
    transition_nodes = self.findall_fn['transition'](node)
    for node in transition_nodes:
      if not 'target' in node.attrib:
        raise NameError('transition without target not supported yet')

      code_string = "status = self.trans({})".format(node.attrib['target'])
      if not "event" in node.attrib:
        if 'cond' in transition.attrib:
          code_string = 'status = return_status.HANDLED\n'
          snippet = 'self.' + transition.attrib['cond']
          code_string += "if({}):\n".format(snippet) 
          code_string += self.indent_amount + "status = self.trans({})".format(transition.attrib['target'])

        self._d_prepend_log(name, node, 'SCXML_INIT_SIGNAL')
        # force the posting of the SCXML_INIT_SIGNAL event in the entry code
        # of this state
        self._d_prepend_posting_of_scxml_init_in_entry_condition(name, node)

      index = self.index_for_signal(self._states_dict[name]['cl'], 'SCXML_INIT_SIGNAL')
      self._states_dict[name]['cl'][index] = {'SCXML_INIT_SIGNAL': code_string}

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
    file_name, module_name, directory, statename = \
      self._sc_write_to_file(file_name=self.python_file_name)
    sys.path.append(directory)

    exec("from {module_name} import {statename} as ScxmlChart".format(
      module_name=module_name, statename=statename
      ), globals()
    )

    #if not self.debugger:
    #  risky_os.remove(file_name)

    # If you run a linter it will say that this ScxmChart class is not
    # defined, but it is defined and imported in the above exec call.  Linters
    # aren't smart enough to 'see into' this kind of exec code.
    return ScxmlChart(name=self.get_name(), log_file=self.log_file)

  def _sc_write_to_file(self, file_name=None, indent_amount=None):
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
       
      file_name, module_name, directory, statename = self._sc_write_to_file()
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
      file_name = self._sc_default_python_file_name()

    Path.touch(Path(file_name))
    directory = str(Path(file_name).resolve().parent)
    module_name = Path(file_name).stem

    class_code = self._code
    instantiation_code = self._s_instantiation_template().format(
      class_code=class_code,
      scxml_chart_class=self.scxml_chart_class,
      log_file=self.log_file,
      i=indent_amount,
      name=self.get_name())

    code = self._s_post_instantiation_template().format(
      i=indent_amount,
      instantiation_code=instantiation_code)

    with open(str(file_name), "w") as fp:
      fp.write(code)

    return file_name, module_name, directory, self.scxml_chart_class
  
  def _sc_default_python_file_name(self):
    path_to_xml = Path(self.file_path)
    module_base = path_to_xml.stem
    directory = path_to_xml.resolve().parent
    module_name = str("{}_{}".format(module_base, self.chart_suffix))
    file_name = str(
      directory / "{}.py".format(module_name)
    )
    return file_name

  #def _x_findall(self, xpath, ns=None, node=None):
  #  '''find all subnodes of node given the xpath search parameter

  #  **Args**:
  #     | ``xpath`` (type1): xpath without namespace clutter
  #     | ``ns=None`` (type1): optional namespace
  #     | ``node=None`` (type1): the node to search, root if omitted


  #  **Returns**:
  #     (xml.etree.ElementTree.Element): result of search

  #  **Example(s)**:
  #    
  #  .. code-block:: python
  #     
  #     main = XmlToMiros('main.scxml')
  #     result = main.findall(.//state[@id='Test2']")

  #  '''
  #  if node is None:
  #    search_root = self.root

  #  if ns is None:
  #    # create a copy of our input
  #    _xpath = "{}".format(xpath)
  #    for tag_name_without_ns in self.tag_lookup.keys():
  #      # Does our origin search has a tag which is protected
  #      # by a namespace prefix? If so, add the namespace 
  #      # prefix to our internal version of that xpath
  #      if tag_name_without_ns in xpath:
  #        _xpath = "{}".format(_xpath.replace(tag_name_without_ns,
  #          self.tag_lookup[tag_name_without_ns]))
  #  else:
  #    _xpath = xpath

  #  return search_root.findall(_xpath)

  def _x_findall_multiple_tags(self, ns, node=None):
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
       
      self.find_states = partial(self._x_findall_multiple_tags, 
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

  def _x_findall(self, arg, node=None):
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
        name : partial(self._x_findall, 
          "{"+XmlToMiros.namespaces['sc']+"}"+name) \
          for name \
          in XmlToMiros.supported_tags
      }
      log_nodes = self.findall_fn['log'](node)

    '''
    return node.findall(arg)

  def _x_get_tag_without_namespace(self, node):
    '''Get the tag name without the namespace information prepended to it

    **Args**:
       | ``node`` (Element): The element to search


    **Returns**:
       (str): The tag name as a string

    '''
    return node.tag.split('}')[-1]

  def _x_is_tag(self, arg, node):
    return True if self._x_get_tag_without_namespace(node) == arg else False

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

    # if your parent is no
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
        parent=node)

  def get_name(self):
    '''get the name of the Statechart from the provided SCXML document.'''
    name = None
    if 'name' in self.root.attrib:
      name = self.root.attrib['name']
    else:
      name = self.chart_suffix
    return name

  @staticmethod
  def _x_include_xml(node):
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
       
      include_found, file_name, include_element = self._x_include_xml(node)

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
    # TODO: remove this stupid way of using a namespace when you build
    # a test for this feature
    include_element = node.findall('{'+XmlToMiros.namespaces['xi']+'}include')
    if len(include_element) != 0:
      result = True
      file_name = include_element[0].attrib['href']
      ie = include_element[0]
    return result, file_name, ie

  @staticmethod
  def _d_escape_quotes(string):
    _trans_dict = {
      "'": "\\'", 
      '"': '\\"'
    }
    result = string.translate(str.maketrans(_trans_dict)) 
    return result

  def _s_pre_instantiation_template(self):
    return """# autogenerated from {filepath}
import re
import dill
import time
import logging
from pathlib import Path
from functools import partial
from functools import lru_cache
from collections import namedtuple
from collections import OrderedDict

from miros import pp
from miros import Event
from miros import spy_on
from miros import signals
from miros import ActiveObject
from miros import return_status
{custom_imports}
{state_code}"""

  def _s_logging_class_definition_template(self):
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
{i}{i}print(trace_without_datetime)
{i}{i}logging.debug("T: " + trace_without_datetime)

{i}def spy_callback(self, spy):
{i}{i}'''spy with machine name pre-pending'''
{i}{i}print(spy)
{i}{i}logging.debug("S: [%s] %s" % (self.name, spy))

{i}def clear_log(self):
{i}{i}with open(self.log_file, "w") as fp:
{i}{i}{i}fp.write("")

"""

  def _s_class_definition_template(self):
    return """{logging_class_code}

STXRef = namedtuple('STXRef', ['send_id', 'thread_id'])

class {scxml_chart_class}({scxml_chart_superclass}):
{i}def __init__(self, name, log_file):
{i}{i}super().__init__(name, log_file)
{i}{i}self.shot_lookup = {_dict_}
{i}{i}{data}
{i}def start(self):
{i}{i}self.start_at({starting_state})

{i}@lru_cache(maxsize=32)
{i}def tockenize(self, signal_name):
{i}{i}return set(signal_name.split("."))

{i}@lru_cache(maxsize=32)
{i}def token_match(self, resident, other):
{i}{i}alien_set = self.tockenize(other)
{i}{i}resident_set = self.tockenize(resident)
{i}{i}result = True if len(resident_set.intersection(alien_set)) >= 1 else False
{i}{i}return result

{i}def post_fifo_with_sendid(self, sendid, e, period=None, times=None, deferred=None):
{i}{i}thread_id = self.post_fifo(e, period, times, deferred)
{i}{i}if thread_id is not None:
{i}{i}{i}self.shot_lookup[e.signal_name] = \\
{i}{i}{i}{i}STXRef(thread_id=thread_id,send_id=sendid)

{i}def post_lifo_with_sendid(self, sendid, e, period=None, times=None, deferred=None):
{i}{i}thread_id = super().post_lifo(e, period, times, deferred)
{i}{i}if thread_id is not None:
{i}{i}{i}self.shot_lookup[e.signal_name] = \\
{i}{i}{i}{i}STXRef(thread_id=thread_id,send_id=sendid)

{i}def cancel_with_sendid(self, sendid):
{i}{i}thread_id = None
{i}{i}for k, v in self.shot_lookup.items():
{i}{i}{i}if v.send_id == sendid:
{i}{i}{i}{i}thread_id = v.thread_id
{i}{i}{i}{i}break
{i}{i}if thread_id is not None:
{i}{i}{i}self.cancel_event(thread_id)

{i}def cancel_all(self, e):
{i}{i}token = e.signal_name
{i}{i}for k, v in self.shot_lookup.items():
{i}{i}{i}if self.token_match(token, k):
{i}{i}{i}{i}self.cancel_events(Event(signal=k))
{i}{i}{i}{i}break

"""

  def _s_instantiation_template(self):
    return """{class_code}

if __name__ == '__main__':
{i}ao = {scxml_chart_class}(\"{name}\", \"{log_file}\")
{i}ao.live_spy = True"""

  def _s_post_instantiation_template(self):
    return """{instantiation_code}
{i}ao.start()
{i}time.sleep(0.01)"""

  def _sc_create_miros_code(self, state_chart_dict=None, custom_imports=None):
    '''create the python code which can manifest the statechart

    **Args**:
       | ``custom_imports=None`` (type1): A string of custome imports


    **Returns**:
       (str): Python code which can run a statechart.

    '''
    if state_chart_dict is None:
      state_chart_dict = self._state_chart_dict

    indent_amount = self.indent_amount

    if custom_imports is None:
      imports = ""
    else:
      imports = "\n".split(custom_imports)

    state_code = ""

    # if there is not 'start_at' in our dictionary, start at the first state
    if state_chart_dict['start_at'] is None:
      starting_state = state_chart_dict['state'].keys()[0]
    else:
      starting_state = state_chart_dict['start_at']

    # globs handled here
    for (state_name, v) in state_chart_dict['states'].items():
      if v['*']:
        state_code += self._sc_write_state_code_with_globs(state_name, v,
          indent_amount) + "\n"
      else:
        state_code += self._sc_write_state_code(state_name, v, indent_amount) + "\n"

    pre_instantiation_code = \
      self._s_pre_instantiation_template().format(
          log_file=self.log_file,
          filepath=str(self.file_path),
          uuid=self.chart_suffix,
        custom_imports=imports,
        state_code=state_code)

    logging_class_code = self._s_logging_class_definition_template().format(
      file_path=str(self.file_path),
      scxml_chart_superclass=self.scxml_chart_superclass,
      scxml_chart_class=self.scxml_chart_class,
      pre_instantiation_code=pre_instantiation_code,
      i=indent_amount)

    data = ""
    for index, data_item in enumerate(self._state_chart_dict['data']):
      if index == 0:
        data += data_item + '\n'
      else:
        data += "{i}{i}{data_item}\n".format(i=indent_amount, data_item=data_item)

    class_code = self._s_class_definition_template().format(
      logging_class_code=logging_class_code,
      file_path=str(self.file_path),
      scxml_chart_superclass=self.scxml_chart_superclass,
      scxml_chart_class=self.scxml_chart_class,
      pre_instantiation_code=pre_instantiation_code,
      data=data,
      i=indent_amount,
      starting_state=starting_state,
      _dict_="{}")

    return class_code

  @staticmethod
  def _sc_write_state_code(state_name, state_dict, indent_amount):

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

    internal_cl_template = '''{i}elif(e.signal == signals.{signal_name}):
{event_code}'''

    external_cl_template = '''{i}elif(self.token_match(e.signal_name, \"{signal_name}\")):
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
        if miros_signals.is_inner_signal(signal_name):
          cls += internal_cl_template.format(
            i=indent_amount,
            signal_name=signal_name,
            event_code= event_code)
        else:
          cls += external_cl_template.format(
            i=indent_amount,
            signal_name=signal_name,
            event_code= event_code)

    # remove distracting double new lines
    cls = cls.replace("\n\n", "\n")

    parent_state = "self.top" if state_dict['p'] is None else state_dict['p']
    state_code = state_template.format(
        i=indent_amount,
        state_name=state_name,
        cls=cls,
        parent_state=parent_state)
    return state_code

  @staticmethod
  def _sc_write_state_code_with_globs(state_name, state_dict, indent_amount):

    state_template = '''
@spy_on
def {state_name}(self, e):
{i}status = return_status.UNHANDLED
{cls}{i}elif(signals.is_inner_signal(e.signal)):
{i}{i}self.temp.fun = {parent_state}
{i}{i}status = return_status.SUPER
{i}else:  # "*"
{catch_all}
{i}return status'''

    first_cl_template = '''{i}if(e.signal == signals.{signal_name}):
{event_code}'''

    internal_cl_template = '''{i}elif(e.signal == signals.{signal_name}):
{event_code}'''

    external_cl_template = '''{i}elif(self.token_match(e.signal_name, \"{signal_name}\")):
{event_code}'''

    glob_template = '''{i}{i}{event_code}'''
    
    cls = ""
    signal_name, event_code = next(iter(state_dict.items()))
    for index, catch_dict in enumerate(state_dict['cl']):
      signal_name, event_code = next(iter(catch_dict.items()))

      if signal_name != "*":
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
          if miros_signals.is_inner_signal(signal_name):
            cls += internal_cl_template.format(
              i=indent_amount,
              signal_name=signal_name,
              event_code= event_code)
          else:
            cls += external_cl_template.format(
              i=indent_amount,
              signal_name=signal_name,
              event_code= event_code)
      else:
        #event_code.replace('\n', '{i}{i}{i}\n'.format(i=indent_amount))
        code = ""
        for event_code_snippet in event_code.split('\n'):
          code += glob_template.format(i=indent_amount, event_code=event_code_snippet) + '\n'
        catch_all = code

    # remove distracting double new lines
    cls = cls.replace("\n\n", "\n")

    parent_state = "self.top" if state_dict['p'] is None else state_dict['p']
    state_code = state_template.format(
      i=indent_amount,
      state_name=state_name,
      cls=cls,
      catch_all=catch_all,
      parent_state=parent_state)
    return state_code
