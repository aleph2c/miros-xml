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

def test_datamodel_1():
  path = data_path / 'data_model_1.scxml'
  main = XmlToMiros(path)
  

