# 2024 R-Biz Challenge Soomac ZeroBurger

Bottle Flip >

Controller
1. python BottleFlip.py

   
ZeroBurger >

Arduino
1. cd src/workspace
2. Upload macstouch_loadcell.ino

Controller
1. python robot_server.py

PC
(conda env) /src/requirements.txt
1. roscore
2. rosrun macstouch main.py
3. rosrun macstouch vision_node.py
4. rosrun macstouch control.py
5. rosrun macstouch pc_client.py
6. rosrun rosserial_python serial_node.py _port:=/dev/ttyACM0 _baud:=57600

(Conda X) /src/gui/requirements.txt
0. cd src/gui
1. python first_page.py
2. rosrun macstouch menu_list_node.py
3. rosrun macstouch error_node.py
