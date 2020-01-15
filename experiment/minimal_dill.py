import dill

def somefun(string):
  print(string)

dill.dump(somefun, open("save_somefun.p", "wb"))
fun_from_binary = dill.load(open("save_somefun.p", "rb"))
fun_from_binary('hello world')
