idf_component_register(
    SRCS "main.c" "policy.c"
    INCLUDE_DIRS "."
    REQUIRES
        driver
        esp_driver_ledc
        esp_driver_gpio
        esp_driver_i2c
        nvs_flash
        bt
        esp_event
        esp_timer
)
