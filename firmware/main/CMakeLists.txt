idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES
        driver
        esp_driver_ledc
        esp_driver_gpio
        esp_driver_i2c
        esp_driver_rmt
        nvs_flash
        bt
        esp_event
        esp_timer
)
