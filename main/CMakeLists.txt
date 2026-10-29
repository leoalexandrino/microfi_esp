# MicroFI 0.4

idf_component_register(
    SRCS main.cc main_functions.cc microfi.cc image_input.cc model.cc
    PRIV_REQUIRES spi_flash esp_partition nvs_flash esp_timer
    INCLUDE_DIRS "")
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error)