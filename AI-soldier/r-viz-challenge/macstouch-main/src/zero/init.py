# -*- coding: utf-8 -*-
from rbsys import *

#This is sample program for initial settings.

if name == "main":
  rbs = RobSys()
  rbs.open()
  a = rbs.cmd_reset()
  print(a)
  rbs.assign_din(run=0)

  k=rbs.set_robtask("test_bottle.py")
  rbs.assign_din(run=0, stop=1, err_reset=2, pause=3 )
  rbs.assign_dout(running=16, svon=17, emo=18, in_pause=19)
  print("test ", k)

# rbs.cmd_run("test_bottle.py")
  print(rbs.req_mcmd()[4])
  rbs.close()
