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

event_1 = Event(signal="event.name1")
event_2 = Event(signal="event.name2")
event_1a = event_1
event_3 = Event(signal=signals.event_3)

if __name__ == '__main__':
  print(event_1.signal_name)

  assert(event_3.signal == signals.event_3)
  assert(event_1.signal_name == event_1a.signal_name)
  assert(event_2.signal_name != event_1a.signal_name)
  pp(signals)

  print(event_3.signal)
  print(signals.event.name1)

