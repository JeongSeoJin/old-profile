from tkinter import Tk, Canvas, Button, Frame
from pathlib import Path
from PIL import Image, ImageTk

# 경로 설정
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH_2 = OUTPUT_PATH / Path(r"/home/seojin/catkin_ws/src/macstouch/src/gui/build_figma/assets/frame0")

def relative_to_assets_2(path: str) -> Path:
    return ASSETS_PATH_2 / Path(path)

def resize_image(image_path, scale=2/3):
    """이미지의 크기를 scale 비율만큼 줄임"""
    img = Image.open(image_path)
    new_size = (int(img.width * scale), int(img.height * scale))
    img = img.resize(new_size, Image.ANTIALIAS)
    return ImageTk.PhotoImage(img)

def create_third_page():
    window = Tk()
    window.geometry("1080x1920")
    window.configure(bg="#FFFFFF")

    # 두 번째 페이지 Frame
    page2 = Frame(window, bg="#FFFFFF")
    page2.place(x=0, y=0, width=1080, height=1920)

    canvas2 = Canvas(
        page2,
        bg="#FFFFFF",
        height=1920,
        width=1080,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )
    canvas2.place(x=0, y=0)

    # 이미지 리사이즈 (2/3 크기)
    button_image = resize_image(relative_to_assets_2("button_1.png"))

    # 버튼 크기 및 배치 설정
    button_width = 340  # 버튼 너비 (화면 크기에 맞춰 조정)
    button_height = 335  # 버튼 높이 (화면 크기에 맞춰 조정)
    padding_x = 30  # 좌우 여백
    padding_y = 30  # 위아래 여백

    positions = [
        (30, 316),  # 첫 번째 줄 첫 번째 버튼 좌표
        (375, 316),  # 첫 번째 줄 두 번째 버튼 좌표
        (720, 316),  # 첫 번째 줄 세 번째 버튼 좌표
        (30, 661),  # 두 번째 줄 첫 번째 버튼 좌표
        (375, 661),  # 두 번째 줄 두 번째 버튼 좌표
        (720, 661),  # 두 번째 줄 세 번째 버튼 좌표
        (30, 1006),  # 세 번째 줄 첫 번째 버튼 좌표
        (375, 1006),  # 세 번째 줄 두 번째 버튼 좌표
        (720, 1006)  # 세 번째 줄 세 번째 버튼 좌표
    ]

    # 9개의 버튼을 생성 및 배치
    buttons = []
    for i in range(9):
        button = Button(
            page2,
            image=button_image,
            borderwidth=0,
            highlightthickness=0,
            command=lambda i=i: print(f"button {i+1} clicked"),
            relief="flat"
        )
        button.place(
            x=positions[i][0],
            y=positions[i][1],
            width=button_width,
            height=button_height
        )
        buttons.append(button)

    window.resizable(False, False)
    window.mainloop()

# 두 번째 페이지 생성 실행
if __name__ == "__main__":
    create_third_page()
