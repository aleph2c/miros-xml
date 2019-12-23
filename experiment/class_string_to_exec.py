from pathlib import Path

class_string = """
class Bob():

  def __init__(self, path):
    self.name = str(path)

  def to_s(self):
    return self.name
"""
b, loc = None, locals()
exec(class_string, loc)
Bob = loc['Bob']
b = Bob(Path(".")/".."/"data"/"calc.scxml")
# b.name with break debugger, but print(b.name) will work in pdb
print(b.name)


