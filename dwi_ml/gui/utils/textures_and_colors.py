import dearpygui.dearpygui as dpg


def manage_textures():
    dpg.add_texture_registry(label="Demo Texture Container",
                             tag="__demo_texture_container")
    _create_static_textures()
    _create_dynamic_textures()


def manage_colors():
    dpg.add_colormap_registry(label="Demo Colormap Registry",
                              tag="__demo_colormap_registry")
    with dpg.theme(tag="__demo_hyperlinkTheme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, [0, 0, 0, 0])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [0, 0, 0, 0])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,
                                [29, 151, 236, 25])
            dpg.add_theme_color(dpg.mvThemeCol_Text, [29, 151, 236])


def _create_static_textures():
    ## create static textures
    texture_data1 = []
    for i in range(100 * 100):
        texture_data1.append(255 / 255)
        texture_data1.append(0)
        texture_data1.append(255 / 255)
        texture_data1.append(255 / 255)

    texture_data2 = []
    for i in range(50 * 50):
        texture_data2.append(255 / 255)
        texture_data2.append(255 / 255)
        texture_data2.append(0)
        texture_data2.append(255 / 255)

    texture_data3 = []
    for row in range(50):
        for column in range(50):
            texture_data3.append(255 / 255)
            texture_data3.append(0)
            texture_data3.append(0)
            texture_data3.append(255 / 255)
        for column in range(50):
            texture_data3.append(0)
            texture_data3.append(255 / 255)
            texture_data3.append(0)
            texture_data3.append(255 / 255)
    for row in range(50):
        for column in range(50):
            texture_data3.append(0)
            texture_data3.append(0)
            texture_data3.append(255 / 255)
            texture_data3.append(255 / 255)
        for column in range(50):
            texture_data3.append(255 / 255)
            texture_data3.append(255 / 255)
            texture_data3.append(0)
            texture_data3.append(255 / 255)

    dpg.add_static_texture(100, 100, texture_data1,
                           parent="__demo_texture_container",
                           tag="__demo_static_texture_1",
                           label="Static Texture 1")
    dpg.add_static_texture(50, 50, texture_data2,
                           parent="__demo_texture_container",
                           tag="__demo_static_texture_2",
                           label="Static Texture 2")
    dpg.add_static_texture(100, 100, texture_data3,
                           parent="__demo_texture_container",
                           tag="__demo_static_texture_3",
                           label="Static Texture 3")


def _create_dynamic_textures():
    ## create dynamic textures
    texture_data1 = []
    for i in range(100 * 100):
        texture_data1.append(255 / 255)
        texture_data1.append(0)
        texture_data1.append(255 / 255)
        texture_data1.append(255 / 255)

    texture_data2 = []
    for i in range(50 * 50):
        texture_data2.append(255 / 255)
        texture_data2.append(255 / 255)
        texture_data2.append(0)
        texture_data2.append(255 / 255)

    dpg.add_dynamic_texture(100, 100, texture_data1,
                            parent="__demo_texture_container",
                            tag="__demo_dynamic_texture_1")
    dpg.add_dynamic_texture(50, 50, texture_data2,
                            parent="__demo_texture_container",
                            tag="__demo_dynamic_texture_2")

