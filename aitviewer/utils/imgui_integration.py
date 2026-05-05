# Copyright (C) 2022-2026  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos

import ctypes

import imgui
import moderngl
from moderngl_window.integrations.imgui import ModernglWindowRenderer


def scale_imgui_style(scale):
    """Scale imgui style sizes proportionally, mirroring ImGuiStyle::ScaleAllSizes, because it is not available
    in pyimgui."""
    if scale < 1.0:
        return

    s = imgui.get_style()

    def f(v):
        return v * scale

    def v2(v):
        return v[0] * scale, v[1] * scale

    s.window_padding = v2(s.window_padding)
    s.window_rounding = f(s.window_rounding)
    s.window_border_size = f(s.window_border_size)
    s.window_min_size = v2(s.window_min_size)
    s.child_rounding = f(s.child_rounding)
    s.child_border_size = f(s.child_border_size)
    s.popup_rounding = f(s.popup_rounding)
    s.popup_border_size = f(s.popup_border_size)
    s.frame_padding = v2(s.frame_padding)
    s.frame_rounding = f(s.frame_rounding)
    s.frame_border_size = f(s.frame_border_size)
    s.item_spacing = v2(s.item_spacing)
    s.item_inner_spacing = v2(s.item_inner_spacing)
    s.cell_padding = v2(s.cell_padding)
    s.touch_extra_padding = v2(s.touch_extra_padding)
    s.indent_spacing = f(s.indent_spacing)
    s.columns_min_spacing = f(s.columns_min_spacing)
    s.scrollbar_size = f(s.scrollbar_size)
    s.scrollbar_rounding = f(s.scrollbar_rounding)
    s.grab_min_size = f(s.grab_min_size)
    s.grab_rounding = f(s.grab_rounding)
    s.log_slider_deadzone = f(s.log_slider_deadzone)
    s.tab_rounding = f(s.tab_rounding)
    s.tab_border_size = f(s.tab_border_size)
    tmw = s.tab_min_width_for_close_button
    if 0 < tmw < 1e37:
        s.tab_min_width_for_close_button = f(tmw)
    s.display_window_padding = v2(s.display_window_padding)
    s.display_safe_area_padding = v2(s.display_safe_area_padding)
    s.mouse_cursor_scale = f(s.mouse_cursor_scale)


class ImGuiRenderer(ModernglWindowRenderer):
    def __init__(self, window, window_type):
        self.window_type = window_type
        super().__init__(window)

    def key_event(self, key, action, modifiers):
        if self.window_type == "pyqt5":
            # HACK: we remap Qt.Key_Enter (numpad enter key) to the normal enter key.
            from PyQt5.QtCore import Qt

            if key == Qt.Key_Enter:
                key = self.wnd.keys.ENTER

        super().key_event(key, action, modifiers)

    def mouse_scroll_event(self, x_offset, y_offset):
        # HACK: pyimgui does not support horizontal scroll
        self.io.mouse_wheel = y_offset

    def render(self, draw_data):
        # HACK: we set the modifiers here every frame because key_event is not called when
        # the window loses and regains focus (e.g. when changing focus with alt+tab)
        self.io.key_alt = self.wnd.modifiers.alt
        self.io.key_ctrl = self.wnd.modifiers.ctrl
        self.io.key_shift = self.wnd.modifiers.shift

        io = self.io
        display_width, display_height = io.display_size
        fb_width = int(display_width * io.display_fb_scale[0])
        fb_height = int(display_height * io.display_fb_scale[1])

        if fb_width == 0 or fb_height == 0:
            return

        self.projMat.value = (
            2.0 / display_width,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 / -display_height,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            -1.0,
            1.0,
            0.0,
            1.0,
        )

        draw_data.scale_clip_rects(*io.display_fb_scale)

        self.ctx.enable_only(moderngl.BLEND)
        self.ctx.blend_equation = moderngl.FUNC_ADD

        # HACK: we set the blend func for the alpha channel to one here because
        # on some linux platforms the alpha channel is used by the window manager
        # for blending with the desktop and we want to keep the window opaque.
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA,
            moderngl.ONE,
            moderngl.ONE,
        )

        self._font_texture.use()

        for commands in draw_data.commands_lists:
            # Write the vertex and index buffer data without copying it
            vtx_type = ctypes.c_byte * commands.vtx_buffer_size * imgui.VERTEX_SIZE
            idx_type = ctypes.c_byte * commands.idx_buffer_size * imgui.INDEX_SIZE
            vtx_arr = (vtx_type).from_address(commands.vtx_buffer_data)
            idx_arr = (idx_type).from_address(commands.idx_buffer_data)
            self._vertex_buffer.write(vtx_arr)
            self._index_buffer.write(idx_arr)

            idx_pos = 0
            for command in commands.commands:
                texture = self._textures.get(command.texture_id)
                if texture is None:
                    raise ValueError(
                        (
                            "Texture {} is not registered. Please add to renderer using "
                            "register_texture(..). "
                            "Current textures: {}".format(command.texture_id, list(self._textures))
                        )
                    )

                texture.use(0)

                x, y, z, w = command.clip_rect
                self.ctx.scissor = int(x), int(fb_height - w), int(z - x), int(w - y)
                self._vao.render(moderngl.TRIANGLES, vertices=command.elem_count, first=idx_pos)
                idx_pos += command.elem_count

        self.ctx.scissor = None
