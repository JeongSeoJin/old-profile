from tkinter import Tk, Canvas, Button, PhotoImage, Frame
from pathlib import Path
import second_page
import os
import subprocess
from time import sleep


# 경로 설정
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH_1 = OUTPUT_PATH / Path(r"/home/mac/catkin_ws/src/macstouch/src/gui/build_figma/assets/frame3")

def relative_to_assets_1(path: str) -> Path:
    return ASSETS_PATH_1 / Path(path)

# 이미지 객체를 전역 변수로 유지
global images
images = {}

# 페이지 전환 함수
def open_second_page(window):
    subprocess.Popen(['python', '/home/mac/catkin_ws/src/macstouch/src/gui/second_page.py'])
    sleep(1)
    window.destroy()

# 첫 번째 페이지 구성
    
def create_first_page():
    global images  # 전역 변수로 이미지 객체를 선언
    window = Tk()
    window.geometry("1080x1920")
    window.configure(bg="#FFFFFF")

    # 첫 번째 페이지 Frame
    page1 = Frame(window, bg="#FFFFFF")
    page1.place(x=0, y=0, width=1080, height=1920)

    canvas1 = Canvas(
        page1,
        bg="#FFFFFF",
        height=1920,
        width=1080,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )
    canvas1.place(x=0, y=0)

    # 첫 번째 페이지 구성 요소
    images["image_1"] = PhotoImage(file=relative_to_assets_1("image_1.png"))
    canvas1.create_image(
        539.0,
        962.0,
        image=images["image_1"]
    )

    canvas1.create_text(
        199.0,
        497.0,
        anchor="nw",
        text="MAC’S TOUCH",
        fill="#000000",
        font=("InriaSans Regular", 90 * -1)
    )

    button_image_1 = PhotoImage(file=relative_to_assets_1("button_1.png"))
    button_1 = Button(
        page1,
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: open_second_page(window),  # 두 번째 페이지 호출
        relief="flat"
    )
    button_1.place(x=195.0, y=1031.0, width=693.0, height=146.0)

    window.resizable(False, False)
    window.mainloop()

# 첫 번째 페이지 생성 실행
if __name__ == "__main__":
    create_first_page()
