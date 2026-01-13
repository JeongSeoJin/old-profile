#!/usr/bin/env python3

from tkinter import Tk, Canvas, Button, PhotoImage, Frame
from pathlib import Path
import second_page
import os
import subprocess
from time import sleep
import rospy
from std_msgs.msg import String
from selected_menu import menu_index


# 경로 설정
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH_1 = OUTPUT_PATH / Path(r"/home/mac/catkin_ws/src/macstouch/src/gui/build_figma/assets/frame2")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH_1 / Path(path)

# 이미지 객체를 전역 변수로 유지
global images
images = {} 

def publish_order(menu_indx):
    rospy.init_node('order_publisher_node', anonymous=True)
    
    order_pub = rospy.Publisher('/menu_index', String, queue_size=10, latch=True)
    # rospy.sleep(1)
    order_msg = str(menu_indx)
    order_pub.publish(order_msg)
    
    rospy.loginfo(f"Published message: {order_msg}")

def open_first_page(window):
    subprocess.Popen(['python', '/home/mac/catkin_ws/src/macstouch/src/gui/first_page.py'])  # 세 번째 페이지 실행
    sleep(1)
    window.destroy()

# 첫 번째 페이지 구성
def create_fourth_page():
    global images  # 전역 변수로 이미지 객체를 선언
    window = Tk()
    window.geometry("1080x1920")
    window.configure(bg = "#FFFFFF")


    canvas = Canvas(
        window,
        bg = "#FFFFFF",
        height = 1920,
        width = 1080,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
    )

    canvas.place(x = 0, y = 0)
    canvas.create_text(
        361.0,
        684.0,
        anchor="nw",
        text="주문 완료",
        fill="#000000",
        font=("Inter", 90 * -1)
    )

    button_image_1 = PhotoImage(
        file=relative_to_assets("button_1.png"))
    button_1 = Button(
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: open_first_page(window),
        relief="flat"
    )
    button_1.place(
        x=193.0,
        y=1037.0,
        width=693.0,
        height=146.0
    )
    window.resizable(False, False)
    window.mainloop()

# 첫 번째 페이지 생성 실행
if __name__ == "__main__":
    publish_order(menu_index)
    create_fourth_page()
