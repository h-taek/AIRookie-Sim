"""로봇 카메라 화면을 딴 창에 그대로 띄운다 (사양 4-7). 사람들에게 보여 주는 창이다.

dearpygui 는 한 프로세스에 창 하나다. 창은 딴 프로세스가 띄우고, 앱이 그린 로봇 시점 버퍼(박스 포함)를
공유 메모리로 넘긴다. 앱은 덮어쓰기만 하고 기다리지 아니한다.

공유 메모리 앞 8 바이트는 쓴 횟수다. 쓰는 동안은 홀수다. 창은 짝수일 때 복사하고, 복사 뒤에도
같은 값이어야 받는다 — 반쯤 쓴 그림을 띄우지 않는다.
"""

from __future__ import annotations

import multiprocessing
from multiprocessing import shared_memory

import numpy as np

창_제목 = "AI Rookie — 로봇 카메라"


class Mirror:
    def __init__(self, 너비: int, 높이: int) -> None:
        self.size = (너비, 높이)
        self._shm = shared_memory.SharedMemory(create=True, size=8 + 높이 * 너비 * 4 * 4)
        self._횟수 = np.ndarray((1,), dtype=np.int64, buffer=self._shm.buf)
        self._그림 = np.ndarray((높이 * 너비 * 4,), dtype=np.float32, buffer=self._shm.buf, offset=8)
        self._횟수[0] = 0
        self._그림[:] = 0.0
        self._그림[3::4] = 1.0
        # 창 프로세스는 앱을 들이지 않는다. 이 모듈과 numpy · dearpygui 만 쓴다.
        self._proc = multiprocessing.get_context("spawn").Process(
            target=_창을_띄운다, args=(self._shm.name, 너비, 높이), daemon=True,
        )
        self._proc.start()

    @property
    def alive(self) -> bool:
        return self._proc.is_alive()

    def put(self, 버퍼: np.ndarray) -> None:
        """앱의 로봇 시점 텍스처 버퍼(RGBA 평면)를 넘긴다."""
        self._횟수[0] += 1
        np.copyto(self._그림, 버퍼)
        self._횟수[0] += 1

    def close(self) -> None:
        if self._proc.is_alive():
            self._proc.terminate()
        self._proc.join(timeout=1.0)
        # 배열이 버퍼를 물고 있으면 닫히지 아니한다.
        del self._횟수, self._그림
        self._shm.close()
        self._shm.unlink()


def _창을_띄운다(이름: str, 너비: int, 높이: int) -> None:
    import time

    import dearpygui.dearpygui as dpg

    # 지우는 것은 만든 앱이다. 여기서 추적하면 창을 닫을 때 앱이 쓰던 메모리를 지운다.
    shm = shared_memory.SharedMemory(name=이름, track=False)
    횟수 = np.ndarray((1,), dtype=np.int64, buffer=shm.buf)
    그림 = np.ndarray((높이 * 너비 * 4,), dtype=np.float32, buffer=shm.buf, offset=8)
    버퍼 = 그림.copy()
    본_횟수 = -1

    dpg.create_context()
    with dpg.texture_registry():
        dpg.add_raw_texture(너비, 높이, 버퍼, format=dpg.mvFormat_Float_rgba, tag="tex")
    with dpg.theme() as 검정:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (0, 0, 0))
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0)
    with dpg.window(tag="root", no_scrollbar=True):
        dpg.add_image("tex", tag="img")
    dpg.bind_item_theme("root", 검정)
    dpg.set_primary_window("root", True)

    def 맞춘다(*_) -> None:
        """비율을 지켜 창에 꽉 차게 가운데 둔다."""
        창_폭, 창_높이 = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
        배율 = min(창_폭 / 너비, 창_높이 / 높이)
        폭, 높 = int(너비 * 배율), int(높이 * 배율)
        dpg.configure_item("img", width=폭, height=높, pos=((창_폭 - 폭) // 2, (창_높이 - 높) // 2))

    with dpg.handler_registry():
        dpg.add_key_press_handler(dpg.mvKey_F, callback=lambda: dpg.toggle_viewport_fullscreen())

    dpg.create_viewport(title=창_제목, width=너비, height=높이, vsync=False)
    dpg.set_viewport_resize_callback(맞춘다)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    맞춘다()
    try:
        while dpg.is_dearpygui_running():
            지금 = int(횟수[0])
            if 지금 != 본_횟수 and 지금 % 2 == 0:
                np.copyto(버퍼, 그림)
                if int(횟수[0]) == 지금:
                    본_횟수 = 지금
                    dpg.set_value("tex", 버퍼)
            dpg.render_dearpygui_frame()
            time.sleep(0.01)
    finally:
        dpg.destroy_context()
        del 횟수, 그림
        shm.close()
