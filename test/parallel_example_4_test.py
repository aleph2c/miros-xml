#import os
#import sys
#import time
#import pytest
#from pathlib import Path
#
#dir_path = os.path.dirname(os.path.realpath(__file__))
#sys.path.insert(0, dir_path)
#
#from conftest import get_log_as_stripped_string
#experiment_path = Path(dir_path) / '..' / 'experiment'
#sys.path.insert(0, str(experiment_path))
#
#from parallel_example_4 import ScxmlChart
#
#log_file = str((experiment_path / "parallel_example_4.log").resolve())
#
#@pytest.mark.exp
#@pytest.mark.experiment_start
#def test_experiment_4_start():
#  time.sleep(0.1)
#  example = ScxmlChart(
#    name="parallel",
#    log_file=log_file,
#    live_trace=True,
#    live_spy=True
#  )
#  example.instrumented = True
#  example.start()
#  time.sleep(0.2)
#
#  result = example.spy()
#  target = \
#      ['START',
#          'SEARCH_FOR_SUPER_SIGNAL:outer_state',
#          'ENTRY_SIGNAL:outer_state',
#          'INIT_SIGNAL:outer_state',
#          '<- Queued:(0) Deferred:(0)']
#
#  assert target == result
#  time.sleep(0.1)
