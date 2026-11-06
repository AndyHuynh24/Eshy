/*
 * Eshy — ESP-IDF firmware for XIAO ESP32-S3.
 *
 * Servo index order: 0=FL, 1=FR, 2=BL, 3=BR (pins below).
 *
 * Robot frame: X+ forward (head), Y+ right, Z+ up.
 * A positive robot-frame hip angle swings the foot forward on every leg.
 * Left/right servos are mirrored in hardware, so:
 *   servo_deg = 90 + SERVO_SIGN[i] * robot_deg
 *
 * BLE commands (Nordic UART Service, newline-terminated):
 *   c                    center all servos
 *   s                    sweep all servos 60..120
 *   i                    IMU-only (servos held at center)
 *   M:fl,fr,bl,br        raw servo degrees (0..180)
 *   mN:a                 raw servo degrees for one leg (N=0..3)
 *   A:fl,fr,bl,br        robot-frame degrees — policy / sequence playback
 *   O:fl,fr,bl,br        per-leg trim offsets (±30)
 *   R:N                  slew limit for A: commands, degrees per call
 *
 * Telemetry @ 50 Hz over BLE:
 *   R:roll P:pitch Y:yaw S:fl,fr,bl,br G:gx,gy,gz Q:qw,qx,qy,qz\n
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "driver/ledc.h"
#include "driver/gpio.h"
#include "driver/i2c.h"

#include "esp_log.h"
#include "esp_err.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "driver/rmt_tx.h"
#include "driver/rmt_encoder.h"

#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/ble_uuid.h"
#include "host/util/util.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

#define TAG "ROBOT"

/* ================================================================= */
/* Pin + timing configuration                                        */
/* ================================================================= */
#define FL_PIN  GPIO_NUM_1   /* D0 */
#define FR_PIN  GPIO_NUM_2   /* D1 */
#define BL_PIN  GPIO_NUM_3   /* D2 */
#define BR_PIN  GPIO_NUM_4   /* D3 */

#define I2C_PORT   I2C_NUM_0
#define I2C_SDA    GPIO_NUM_9
#define I2C_SCL    GPIO_NUM_8
#define MPU_ADDR   0x68

#define SERVO_FREQ_HZ 50
#define SERVO_RES     LEDC_TIMER_13_BIT
#define SERVO_MAX     ((1 << 13) - 1)

static const gpio_num_t      servo_pins[4]    = {FL_PIN, FR_PIN, BL_PIN, BR_PIN};
static const ledc_channel_t  servo_channels[4] = {
    LEDC_CHANNEL_0, LEDC_CHANNEL_1, LEDC_CHANNEL_2, LEDC_CHANNEL_3
};
static const ledc_timer_t    servo_timers[4]   = {
    LEDC_TIMER_0, LEDC_TIMER_1, LEDC_TIMER_2, LEDC_TIMER_3
};
static const char *servo_names[4] = {"FL", "FR", "BL", "BR"};

static const int SERVO_SIGN[4] = { -1, 1, -1, 1 };  /* FL, FR, BL, BR */
#define ACTION_RANGE_DEG 90   /* hard clip on A: commands */

/* Open-loop sequence playback doesn't need the IMU. With this set, the MPU
 * is never initialised and telemetry reports identity quat + zero gyro, so
 * closed-loop policy inference will not work until it is cleared. */
#define DISABLE_MPU 1

/* Slew limit on A: commands (robot-frame degrees per call). Kept low because
 * the 5V boost browns out on large simultaneous steps. Override with R:N. */
#define SLEW_DEFAULT_DEG_PER_TICK   3
static int slew_max_deg_per_tick = SLEW_DEFAULT_DEG_PER_TICK;
static int last_robot_deg[4] = { 0, 0, 0, 0 };

/* Per-leg mechanical trim (degrees) applied in servo_write(), so that a
 * logical 90 puts each leg straight down. Tune live with O:, then hardcode. */
static int servo_offset_deg[4] = { 0, 0, 0, 0 };  /* FL, FR, BL, BR */

/* ================================================================= */
/* Shared state                                                      */
/* ================================================================= */
typedef enum {
    MODE_CENTER = 0,
    MODE_SWEEP,
    MODE_IMU,
    MODE_MANUAL,
} robot_mode_t;

static SemaphoreHandle_t state_mtx;
static robot_mode_t robot_mode = MODE_CENTER;
static int current_angles[4] = {90, 90, 90, 90};

static float q_w = 1.0f, q_x = 0.0f, q_y = 0.0f, q_z = 0.0f;
static const float madgwick_beta = 0.1f;

/* Rest-pose anchor captured after boot calibration. Telemetry reports
 * q_current ⊗ conj(q_boot), so the rest pose reads as identity regardless of
 * IMU mounting tilt — matching sim, which resets to identity every episode. */
static float qb_w = 1.0f, qb_x = 0.0f, qb_y = 0.0f, qb_z = 0.0f;

static float last_gx = 0.0f, last_gy = 0.0f, last_gz = 0.0f;   /* rad/s */
static float gyro_bias_x = 0.0f, gyro_bias_y = 0.0f, gyro_bias_z = 0.0f;

static uint8_t  own_addr_type = 0;
static uint16_t conn_handle   = 0;          /* 0 = not connected */
static uint16_t tx_val_handle = 0;
static bool     tx_subscribed = false;

/* ================================================================= */
/* Servo driver                                                      */
/* ================================================================= */
static uint32_t angle_to_duty(int angle)
{
    if (angle < 0)   angle = 0;
    if (angle > 180) angle = 180;
    int pulse_us = 500 + (angle * 2000) / 180;           /* 500..2500 us */
    return ((uint32_t)pulse_us * SERVO_MAX) / 20000;     /* 20 ms period */
}

static void servo_write(int i, int angle)
{
    if (i < 0 || i > 3) return;
    int physical = angle + servo_offset_deg[i];
    if (physical < 0)   physical = 0;
    if (physical > 180) physical = 180;
    ledc_set_duty(LEDC_LOW_SPEED_MODE, servo_channels[i], angle_to_duty(physical));
    ledc_update_duty(LEDC_LOW_SPEED_MODE, servo_channels[i]);

    xSemaphoreTake(state_mtx, portMAX_DELAY);
    /* Report the logical angle so trim stays invisible to the host. */
    current_angles[i] = angle;
    xSemaphoreGive(state_mtx);
}

static void servos_init(void)
{
    /* One timer per channel so a write to one servo can't affect another. */
    for (int i = 0; i < 4; i++) {
        ledc_timer_config_t timer = {
            .speed_mode      = LEDC_LOW_SPEED_MODE,
            .timer_num       = servo_timers[i],
            .duty_resolution = SERVO_RES,
            .freq_hz         = SERVO_FREQ_HZ,
            .clk_cfg         = LEDC_AUTO_CLK,
        };
        ESP_ERROR_CHECK(ledc_timer_config(&timer));

        ledc_channel_config_t ch = {
            .gpio_num   = servo_pins[i],
            .speed_mode = LEDC_LOW_SPEED_MODE,
            .channel    = servo_channels[i],
            .timer_sel  = servo_timers[i],
            .duty       = 0,            /* no pulses until centered below */
            .hpoint     = 0,
        };
        ESP_ERROR_CHECK(ledc_channel_config(&ch));

        ESP_LOGI(TAG, "  %s on GPIO %d -> channel %d, timer %d",
                 servo_names[i], servo_pins[i], servo_channels[i], servo_timers[i]);
    }
    for (int i = 0; i < 4; i++) servo_write(i, 90);
}

static void robot_write(int i, int robot_deg)
{
    servo_write(i, 90 + SERVO_SIGN[i] * robot_deg);
}

/* Apply a robot-frame A: command with clip + slew limit, physically driving
 * only ONE servo per call: the 5V boost can't supply inrush for several
 * MG90S moving at once. Diagonal order (FL, BR, FR, BL) keeps the body
 * balanced while legs catch up. Each servo updates at 1/4 of the call rate,
 * so stream at 100 Hz for a 25 Hz per-servo rate. */
static void robot_write_4_staggered(const int rd[4])
{
    static const int order[4] = { 0, 3, 1, 2 };  /* FL, BR, FR, BL */
    static int order_idx = 0;

    int target[4];
    for (int i = 0; i < 4; i++) {
        int t = rd[i];
        if (t >  ACTION_RANGE_DEG) t =  ACTION_RANGE_DEG;
        if (t < -ACTION_RANGE_DEG) t = -ACTION_RANGE_DEG;
        int delta = t - last_robot_deg[i];
        if (delta >  slew_max_deg_per_tick) delta =  slew_max_deg_per_tick;
        if (delta < -slew_max_deg_per_tick) delta = -slew_max_deg_per_tick;
        target[i] = last_robot_deg[i] + delta;
    }

    int i = order[order_idx];
    order_idx = (order_idx + 1) & 3;
    robot_write(i, target[i]);
    last_robot_deg[i] = target[i];
}

/* ================================================================= */
/* WS2812B status LEDs                                               */
/* ================================================================= */
/* 3-pixel WS2812B strip driven directly by RMT at 10 MHz (100 ns/tick):
 *   bit0 = 0.4 us high + 0.85 us low  -> (4, 8)
 *   bit1 = 0.8 us high + 0.45 us low  -> (8, 4)
 * Blue = advertising, green = connected. */
#define LED_PIN       GPIO_NUM_6  /* D5 */
#define LED_COUNT     3
#define LED_BRIGHT    26          /* ~10% of 255 */

static rmt_channel_handle_t led_chan    = NULL;
static rmt_encoder_handle_t led_encoder = NULL;

static void leds_fill(uint8_t r, uint8_t g, uint8_t b)
{
    if (led_chan == NULL) return;
    uint8_t frame[LED_COUNT * 3];
    for (int i = 0; i < LED_COUNT; i++) {
        frame[i*3 + 0] = g;          /* WS2812 byte order is GRB */
        frame[i*3 + 1] = r;
        frame[i*3 + 2] = b;
    }
    rmt_transmit_config_t tx = { .loop_count = 0 };
    rmt_transmit(led_chan, led_encoder, frame, sizeof(frame), &tx);
    rmt_tx_wait_all_done(led_chan, pdMS_TO_TICKS(100));
}

static void leds_init(void)
{
    rmt_tx_channel_config_t tx_cfg = {
        .gpio_num          = LED_PIN,
        .clk_src           = RMT_CLK_SRC_DEFAULT,
        .resolution_hz     = 10 * 1000 * 1000,  /* 100 ns / tick */
        .mem_block_symbols = 64,
        .trans_queue_depth = 4,
    };
    ESP_ERROR_CHECK(rmt_new_tx_channel(&tx_cfg, &led_chan));

    rmt_bytes_encoder_config_t enc_cfg = {
        .bit0 = { .level0 = 1, .duration0 = 4, .level1 = 0, .duration1 = 8 },
        .bit1 = { .level0 = 1, .duration0 = 8, .level1 = 0, .duration1 = 4 },
        .flags.msb_first = 1,
    };
    ESP_ERROR_CHECK(rmt_new_bytes_encoder(&enc_cfg, &led_encoder));
    ESP_ERROR_CHECK(rmt_enable(led_chan));

    leds_fill(0, 0, LED_BRIGHT);
    ESP_LOGI(TAG, "LEDs: %d x WS2812B on GPIO %d", LED_COUNT, LED_PIN);
}

/* ================================================================= */
/* MPU-6050 + Madgwick                                               */
/* ================================================================= */
static void mpu_read(float *ax, float *ay, float *az,
                     float *gx, float *gy, float *gz);
static void madgwick_update(float ax, float ay, float az,
                            float gx, float gy, float gz, float dt);

static esp_err_t mpu_write_reg(uint8_t reg, uint8_t data)
{
    uint8_t buf[2] = {reg, data};
    return i2c_master_write_to_device(I2C_PORT, MPU_ADDR, buf, 2, pdMS_TO_TICKS(100));
}

static void mpu_init(void)
{
    i2c_config_t conf = {
        .mode             = I2C_MODE_MASTER,
        .sda_io_num       = I2C_SDA,
        .scl_io_num       = I2C_SCL,
        .sda_pullup_en    = GPIO_PULLUP_ENABLE,
        .scl_pullup_en    = GPIO_PULLUP_ENABLE,
        .master.clk_speed = 400000,
    };
    ESP_ERROR_CHECK(i2c_param_config(I2C_PORT, &conf));
    ESP_ERROR_CHECK(i2c_driver_install(I2C_PORT, I2C_MODE_MASTER, 0, 0, 0));
    vTaskDelay(pdMS_TO_TICKS(100));

    uint8_t reg = 0x75;          /* WHO_AM_I */
    uint8_t who = 0;
    i2c_master_write_read_device(I2C_PORT, MPU_ADDR, &reg, 1, &who, 1,
                                  pdMS_TO_TICKS(100));
    ESP_LOGI(TAG, "MPU6050 WHO_AM_I: 0x%02X %s",
             who, who == 0x68 ? "(OK)" : "(ERROR!)");

    mpu_write_reg(0x6B, 0x00);   /* wake up */
    mpu_write_reg(0x1B, 0x08);   /* gyro   ±500 dps  */
    mpu_write_reg(0x1C, 0x00);   /* accel  ±2g       */
    mpu_write_reg(0x1A, 0x05);   /* DLPF ~10 Hz — less yaw random-walk */

    /* Gyro bias: average 200 samples (~1 s) while the robot is still. */
    ESP_LOGI(TAG, "Hold robot still — calibrating gyro bias (~1 s)...");
    vTaskDelay(pdMS_TO_TICKS(500));  /* let boot centering finish */
    float sx = 0, sy = 0, sz = 0;
    const int N = 200;
    for (int i = 0; i < N; i++) {
        float ax, ay, az, gx, gy, gz;
        mpu_read(&ax, &ay, &az, &gx, &gy, &gz);
        sx += gx; sy += gy; sz += gz;
        vTaskDelay(pdMS_TO_TICKS(5));
    }
    gyro_bias_x = sx / N;
    gyro_bias_y = sy / N;
    gyro_bias_z = sz / N;
    ESP_LOGI(TAG, "Gyro bias: x=%+.4f y=%+.4f z=%+.4f rad/s",
             gyro_bias_x, gyro_bias_y, gyro_bias_z);

    /* Let Madgwick gravity-align for ~1 s from identity. */
    q_w = 1.0f; q_x = 0.0f; q_y = 0.0f; q_z = 0.0f;
    const int SETTLE = 200;
    for (int i = 0; i < SETTLE; i++) {
        float ax, ay, az, gx, gy, gz;
        mpu_read(&ax, &ay, &az, &gx, &gy, &gz);
        madgwick_update(ax, ay, az, gx, gy, gz, 0.005f);
        vTaskDelay(pdMS_TO_TICKS(5));
    }

    qb_w = q_w; qb_x = q_x; qb_y = q_y; qb_z = q_z;
    ESP_LOGI(TAG, "Anchor pose: w=%+.3f x=%+.3f y=%+.3f z=%+.3f",
             qb_w, qb_x, qb_y, qb_z);
}

/* Returns accel in g, gyro in rad/s. */
static void mpu_read(float *ax, float *ay, float *az,
                     float *gx, float *gy, float *gz)
{
    uint8_t reg = 0x3B;
    uint8_t d[14];
    i2c_master_write_read_device(I2C_PORT, MPU_ADDR, &reg, 1,
                                  d, 14, pdMS_TO_TICKS(100));
    int16_t rax = (int16_t)((d[0]  << 8) | d[1]);
    int16_t ray = (int16_t)((d[2]  << 8) | d[3]);
    int16_t raz = (int16_t)((d[4]  << 8) | d[5]);
    int16_t rgx = (int16_t)((d[8]  << 8) | d[9]);   /* d[6..7] = temp */
    int16_t rgy = (int16_t)((d[10] << 8) | d[11]);
    int16_t rgz = (int16_t)((d[12] << 8) | d[13]);

    /* IMU is mounted upside-down relative to the sim body frame: flip Z. */
    *ax =  rax / 16384.0f;
    *ay =  ray / 16384.0f;
    *az = -raz / 16384.0f;
    *gx =  (rgx / 65.5f) * (float)M_PI / 180.0f - gyro_bias_x;
    *gy =  (rgy / 65.5f) * (float)M_PI / 180.0f - gyro_bias_y;
    *gz = -(rgz / 65.5f) * (float)M_PI / 180.0f - gyro_bias_z;
}

static void madgwick_update(float ax, float ay, float az,
                            float gx, float gy, float gz, float dt)
{
    float qw = q_w, qx = q_x, qy = q_y, qz = q_z;

    float norm = sqrtf(ax*ax + ay*ay + az*az);
    if (norm < 1e-5f) return;
    ax /= norm; ay /= norm; az /= norm;

    float f1 = 2.0f*(qx*qz - qw*qy) - ax;
    float f2 = 2.0f*(qw*qx + qy*qz) - ay;
    float f3 = 2.0f*(0.5f - qx*qx - qy*qy) - az;

    float sw = -2.0f*qy*f1 + 2.0f*qx*f2;
    float sx =  2.0f*qz*f1 + 2.0f*qw*f2 - 4.0f*qx*f3;
    float sy = -2.0f*qw*f1 + 2.0f*qz*f2 - 4.0f*qy*f3;
    float sz =  2.0f*qx*f1 + 2.0f*qy*f2;

    norm = sqrtf(sw*sw + sx*sx + sy*sy + sz*sz);
    if (norm > 1e-5f) { sw /= norm; sx /= norm; sy /= norm; sz /= norm; }

    float qdot_w = 0.5f*(-qx*gx - qy*gy - qz*gz) - madgwick_beta*sw;
    float qdot_x = 0.5f*( qw*gx + qy*gz - qz*gy) - madgwick_beta*sx;
    float qdot_y = 0.5f*( qw*gy - qx*gz + qz*gx) - madgwick_beta*sy;
    float qdot_z = 0.5f*( qw*gz + qx*gy - qy*gx) - madgwick_beta*sz;

    qw += qdot_w * dt;
    qx += qdot_x * dt;
    qy += qdot_y * dt;
    qz += qdot_z * dt;

    norm = sqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
    if (norm > 1e-5f) { qw /= norm; qx /= norm; qy /= norm; qz /= norm; }

    q_w = qw; q_x = qx; q_y = qy; q_z = qz;
}

static void quat_to_rpy(float *roll, float *pitch, float *yaw)
{
    float sinr = 2.0f * (q_w*q_x + q_y*q_z);
    float cosr = 1.0f - 2.0f * (q_x*q_x + q_y*q_y);
    *roll = atan2f(sinr, cosr) * 180.0f / (float)M_PI;

    float sinp = 2.0f * (q_w*q_y - q_z*q_x);
    if (fabsf(sinp) >= 1.0f) *pitch = copysignf(90.0f, sinp);
    else                     *pitch = asinf(sinp) * 180.0f / (float)M_PI;

    float siny = 2.0f * (q_w*q_z + q_x*q_y);
    float cosy = 1.0f - 2.0f * (q_y*q_y + q_z*q_z);
    *yaw = atan2f(siny, cosy) * 180.0f / (float)M_PI;
}

/* ================================================================= */
/* Command parser                                                    */
/* ================================================================= */
static void center_all(void)
{
    for (int i = 0; i < 4; i++) servo_write(i, 90);
}

static void handle_single_char(char c)
{
    switch (c) {
    case 'c': case 'C':
        xSemaphoreTake(state_mtx, portMAX_DELAY);
        robot_mode = MODE_CENTER;
        xSemaphoreGive(state_mtx);
        center_all();
        ESP_LOGI(TAG, "Mode: CENTER");
        break;
    case 's': case 'S':
        xSemaphoreTake(state_mtx, portMAX_DELAY);
        robot_mode = MODE_SWEEP;
        xSemaphoreGive(state_mtx);
        ESP_LOGI(TAG, "Mode: SWEEP");
        break;
    case 'i': case 'I':
        xSemaphoreTake(state_mtx, portMAX_DELAY);
        robot_mode = MODE_IMU;
        xSemaphoreGive(state_mtx);
        center_all();
        ESP_LOGI(TAG, "Mode: IMU-only");
        break;
    default: break;
    }
}

/* Parse one line of input (no trailing newline). */
static void handle_line(const char *raw, int len)
{
    while (len > 0 && (*raw == ' ' || *raw == '\t' || *raw == '\r')) { raw++; len--; }
    while (len > 0 && (raw[len-1] == ' ' || raw[len-1] == '\t' || raw[len-1] == '\r')) len--;
    if (len <= 0) return;

    if (len == 1) { handle_single_char(raw[0]); return; }

    /* "M:fl,fr,bl,br" — raw servo degrees, all four */
    if (len >= 2 && (raw[0] == 'M' || raw[0] == 'm') && raw[1] == ':') {
        int vals[4];
        xSemaphoreTake(state_mtx, portMAX_DELAY);
        for (int i = 0; i < 4; i++) vals[i] = current_angles[i];
        xSemaphoreGive(state_mtx);

        const char *p = raw + 2;
        const char *end = raw + len;
        int idx = 0;
        while (p < end && idx < 4) {
            char *ep;
            long v = strtol(p, &ep, 10);
            if (ep == p) break;
            vals[idx++] = (int)v;
            p = ep;
            if (p < end && *p == ',') p++;
        }
        xSemaphoreTake(state_mtx, portMAX_DELAY);
        robot_mode = MODE_MANUAL;
        xSemaphoreGive(state_mtx);
        for (int i = 0; i < 4; i++) servo_write(i, vals[i]);
        ESP_LOGI(TAG, "Manual ALL: FL=%d FR=%d BL=%d BR=%d",
                 vals[0], vals[1], vals[2], vals[3]);
        return;
    }

    /* "mN:a" — raw servo degrees, one leg */
    if (len >= 4 && (raw[0] == 'm' || raw[0] == 'M') &&
        raw[1] >= '0' && raw[1] <= '3' && raw[2] == ':') {
        int idx = raw[1] - '0';
        int angle = atoi(raw + 3);
        xSemaphoreTake(state_mtx, portMAX_DELAY);
        robot_mode = MODE_MANUAL;
        xSemaphoreGive(state_mtx);
        servo_write(idx, angle);
        ESP_LOGI(TAG, "Manual %s=%d", servo_names[idx], angle);
        return;
    }

    /* "O:fl,fr,bl,br" — per-leg trim offsets, clamped to ±30 */
    if (len >= 2 && (raw[0] == 'O' || raw[0] == 'o') && raw[1] == ':') {
        int vals[4] = {0, 0, 0, 0};
        const char *p = raw + 2;
        const char *end = raw + len;
        int idx = 0;
        while (p < end && idx < 4) {
            char *ep;
            long v = strtol(p, &ep, 10);
            if (ep == p) break;
            vals[idx++] = (int)v;
            p = ep;
            if (p < end && *p == ',') p++;
        }
        for (int i = 0; i < 4; i++) {
            if (vals[i] >  30) vals[i] =  30;
            if (vals[i] < -30) vals[i] = -30;
            servo_offset_deg[i] = vals[i];
        }
        /* Re-issue current angles so the new trim applies immediately. */
        int cur[4];
        xSemaphoreTake(state_mtx, portMAX_DELAY);
        for (int i = 0; i < 4; i++) cur[i] = current_angles[i];
        xSemaphoreGive(state_mtx);
        for (int i = 0; i < 4; i++) servo_write(i, cur[i]);
        ESP_LOGI(TAG, "Offsets set: FL=%+d FR=%+d BL=%+d BR=%+d",
                 servo_offset_deg[0], servo_offset_deg[1],
                 servo_offset_deg[2], servo_offset_deg[3]);
        return;
    }

    /* "A:a0,a1,a2,a3" — robot-frame degrees, + = swing forward */
    if (len >= 2 && (raw[0] == 'A' || raw[0] == 'a') && raw[1] == ':') {
        int rd[4] = {0, 0, 0, 0};
        const char *p = raw + 2;
        const char *end = raw + len;
        int idx = 0;
        while (p < end && idx < 4) {
            char *ep;
            long v = strtol(p, &ep, 10);
            if (ep == p) break;
            rd[idx++] = (int)v;
            p = ep;
            if (p < end && *p == ',') p++;
        }
        xSemaphoreTake(state_mtx, portMAX_DELAY);
        robot_mode = MODE_MANUAL;
        xSemaphoreGive(state_mtx);
        robot_write_4_staggered(rd);
        ESP_LOGI(TAG, "A: rd=[%d,%d,%d,%d] -> last=[%d,%d,%d,%d] (slew=%d)",
                 rd[0], rd[1], rd[2], rd[3],
                 last_robot_deg[0], last_robot_deg[1],
                 last_robot_deg[2], last_robot_deg[3],
                 slew_max_deg_per_tick);
        return;
    }

    /* "R:N" — slew limit for A: commands, degrees per call (1..90) */
    if (len >= 2 && (raw[0] == 'R' || raw[0] == 'r') && raw[1] == ':') {
        char *ep;
        long v = strtol(raw + 2, &ep, 10);
        if (ep != raw + 2) {
            if (v < 1)  v = 1;
            if (v > 90) v = 90;
            slew_max_deg_per_tick = (int)v;
            ESP_LOGI(TAG, "Slew limit set to %d deg/tick", slew_max_deg_per_tick);
        }
        return;
    }
}

/* ================================================================= */
/* BLE UART service (Nordic UART Service)                            */
/* ================================================================= */
/* Service 6e400001-b5a3-f393-e0a9-e50e24dcca9e */
static const ble_uuid128_t nus_svc_uuid = BLE_UUID128_INIT(
    0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0,
    0x93, 0xf3, 0xa3, 0xb5, 0x01, 0x00, 0x40, 0x6e);
/* RX (write)  6e400002-... */
static const ble_uuid128_t nus_rx_uuid = BLE_UUID128_INIT(
    0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0,
    0x93, 0xf3, 0xa3, 0xb5, 0x02, 0x00, 0x40, 0x6e);
/* TX (notify) 6e400003-... */
static const ble_uuid128_t nus_tx_uuid = BLE_UUID128_INIT(
    0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0,
    0x93, 0xf3, 0xa3, 0xb5, 0x03, 0x00, 0x40, 0x6e);

static char rx_line_buf[160];
static int  rx_line_len = 0;

static void rx_feed(const uint8_t *data, int len)
{
    for (int i = 0; i < len; i++) {
        char c = (char)data[i];
        if (c == '\n' || c == '\r') {
            if (rx_line_len > 0) {
                rx_line_buf[rx_line_len] = '\0';
                handle_line(rx_line_buf, rx_line_len);
                rx_line_len = 0;
            }
        } else {
            if (rx_line_len < (int)sizeof(rx_line_buf) - 1) {
                rx_line_buf[rx_line_len++] = c;
            } else {
                rx_line_len = 0;   /* overflow: drop line */
            }
        }
    }
}

static int gatt_nus_access(uint16_t c_handle, uint16_t a_handle,
                           struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    (void)c_handle; (void)a_handle; (void)arg;
    if (ctxt->op == BLE_GATT_ACCESS_OP_WRITE_CHR) {
        uint8_t buf[128];
        uint16_t copied = 0;
        int rc = ble_hs_mbuf_to_flat(ctxt->om, buf, sizeof(buf), &copied);
        if (rc == 0 && copied > 0) rx_feed(buf, copied);
    }
    return 0;
}

static const struct ble_gatt_svc_def gatt_svcs[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &nus_svc_uuid.u,
        .characteristics = (struct ble_gatt_chr_def[]) {
            {
                .uuid = &nus_rx_uuid.u,
                .access_cb = gatt_nus_access,
                .flags = BLE_GATT_CHR_F_WRITE | BLE_GATT_CHR_F_WRITE_NO_RSP,
            },
            {
                .uuid = &nus_tx_uuid.u,
                .access_cb = gatt_nus_access,
                .flags = BLE_GATT_CHR_F_NOTIFY,
                .val_handle = &tx_val_handle,
            },
            { 0 }
        },
    },
    { 0 }
};

static void ble_send(const char *msg)
{
    if (conn_handle == 0 || !tx_subscribed) return;
    int len = (int)strlen(msg);
    /* 20-byte chunks are safe at any negotiated MTU. */
    for (int i = 0; i < len; i += 20) {
        int chunk = (len - i < 20) ? (len - i) : 20;
        struct os_mbuf *om = ble_hs_mbuf_from_flat(msg + i, chunk);
        if (!om) break;
        int rc = ble_gatts_notify_custom(conn_handle, tx_val_handle, om);
        if (rc != 0) break;
    }
}

/* ---------- GAP ---------- */
static int  gap_event_cb(struct ble_gap_event *event, void *arg);

static void start_advertising(void)
{
    struct ble_hs_adv_fields fields = {0};
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.name = (uint8_t *)"Eshy";
    fields.name_len = strlen("Eshy");
    fields.name_is_complete = 1;
    fields.tx_pwr_lvl_is_present = 1;
    fields.tx_pwr_lvl = BLE_HS_ADV_TX_PWR_LVL_AUTO;

    int rc = ble_gap_adv_set_fields(&fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "adv_set_fields: %d", rc);
        return;
    }

    struct ble_gap_adv_params params = {0};
    params.conn_mode = BLE_GAP_CONN_MODE_UND;
    params.disc_mode = BLE_GAP_DISC_MODE_GEN;
    rc = ble_gap_adv_start(own_addr_type, NULL, BLE_HS_FOREVER,
                           &params, gap_event_cb, NULL);
    if (rc != 0) ESP_LOGE(TAG, "adv_start: %d", rc);
    else        ESP_LOGI(TAG, "Advertising as 'Eshy'");
}

static int gap_event_cb(struct ble_gap_event *event, void *arg)
{
    (void)arg;
    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
        if (event->connect.status == 0) {
            conn_handle = event->connect.conn_handle;
            rx_line_len = 0;   /* flush stale bytes from the previous session */
            leds_fill(0, LED_BRIGHT, 0);        /* green = connected */
            ESP_LOGI(TAG, "BLE connected (handle %d)", conn_handle);
        } else {
            ESP_LOGW(TAG, "BLE connect failed, restarting adv");
            start_advertising();
        }
        break;
    case BLE_GAP_EVENT_DISCONNECT:
        ESP_LOGI(TAG, "BLE disconnected (reason %d)",
                 event->disconnect.reason);
        conn_handle   = 0;
        tx_subscribed = false;
        rx_line_len   = 0;
        center_all();   /* park the robot if the host drops */
        /* Re-anchor: the next session's rest pose becomes the new zero. */
        q_w = 1.0f; q_x = 0.0f; q_y = 0.0f; q_z = 0.0f;
        qb_w = 1.0f; qb_x = 0.0f; qb_y = 0.0f; qb_z = 0.0f;
        leds_fill(0, 0, LED_BRIGHT);            /* blue = advertising */
        start_advertising();
        break;
    case BLE_GAP_EVENT_SUBSCRIBE:
        if (event->subscribe.attr_handle == tx_val_handle) {
            tx_subscribed = event->subscribe.cur_notify;
            ESP_LOGI(TAG, "TX subscribed=%d", tx_subscribed);
        }
        break;
    case BLE_GAP_EVENT_MTU:
        ESP_LOGI(TAG, "MTU update, conn=%d mtu=%d",
                 event->mtu.conn_handle, event->mtu.value);
        break;
    default: break;
    }
    return 0;
}

static void on_sync(void)
{
    int rc = ble_hs_util_ensure_addr(0);
    if (rc) { ESP_LOGE(TAG, "ensure_addr: %d", rc); return; }
    rc = ble_hs_id_infer_auto(0, &own_addr_type);
    if (rc) { ESP_LOGE(TAG, "infer_auto: %d", rc); return; }
    start_advertising();
}

static void on_reset(int reason) { ESP_LOGW(TAG, "BLE reset: %d", reason); }

static void ble_host_task(void *param)
{
    (void)param;
    nimble_port_run();               /* blocks until nimble_port_stop() */
    nimble_port_freertos_deinit();
}

static void ble_start(void)
{
    ESP_ERROR_CHECK(nimble_port_init());

    ble_hs_cfg.sync_cb  = on_sync;
    ble_hs_cfg.reset_cb = on_reset;

    ble_svc_gap_init();
    ble_svc_gatt_init();

    ESP_ERROR_CHECK(ble_gatts_count_cfg(gatt_svcs));
    ESP_ERROR_CHECK(ble_gatts_add_svcs(gatt_svcs));
    ESP_ERROR_CHECK(ble_svc_gap_device_name_set("Eshy"));

    nimble_port_freertos_init(ble_host_task);
}

/* ================================================================= */
/* Main control loop                                                 */
/* ================================================================= */
static void control_task(void *param)
{
    (void)param;

    const TickType_t period = pdMS_TO_TICKS(10);   /* 100 Hz */
    TickType_t last_wake = xTaskGetTickCount();
    int64_t last_us = esp_timer_get_time();

    int   broadcast_ticks = 0;
    float sweep_angle = 90.0f;
    float sweep_dir   = 1.0f;

    while (1) {
        vTaskDelayUntil(&last_wake, period);

        int64_t now_us = esp_timer_get_time();
        float dt = (now_us - last_us) / 1e6f;
        if (dt <= 0.0f || dt > 0.2f) dt = 0.01f;
        last_us = now_us;

#if !DISABLE_MPU
        float ax, ay, az, gx, gy, gz;
        mpu_read(&ax, &ay, &az, &gx, &gy, &gz);
        madgwick_update(ax, ay, az, gx, gy, gz, dt);
        last_gx = gx; last_gy = gy; last_gz = gz;
#else
        last_gx = 0.0f; last_gy = 0.0f; last_gz = 0.0f;
#endif

        float r, p, y;
        quat_to_rpy(&r, &p, &y);

        xSemaphoreTake(state_mtx, portMAX_DELAY);
        robot_mode_t m = robot_mode;
        xSemaphoreGive(state_mtx);

        if (m == MODE_SWEEP) {
            sweep_angle += sweep_dir * 1.0f;      /* 1 deg per 10 ms tick */
            if (sweep_angle >= 120.0f) sweep_dir = -1.0f;
            if (sweep_angle <= 60.0f)  sweep_dir =  1.0f;
            int a = (int)sweep_angle;
            for (int i = 0; i < 4; i++) servo_write(i, a);
        }

        if (++broadcast_ticks >= 2) {   /* 50 Hz telemetry */
            broadcast_ticks = 0;
            xSemaphoreTake(state_mtx, portMAX_DELAY);
            int a0 = current_angles[0], a1 = current_angles[1];
            int a2 = current_angles[2], a3 = current_angles[3];
            float cw = q_w, cx = q_x, cy = q_y, cz = q_z;
            float bw = qb_w, bx = qb_x, by = qb_y, bz = qb_z;
            float gxs = last_gx, gys = last_gy, gzs = last_gz;
            xSemaphoreGive(state_mtx);

            /* q_rel = q_current ⊗ conj(q_boot) */
            float bwc = bw, bxc = -bx, byc = -by, bzc = -bz;
            float qw = cw*bwc - cx*bxc - cy*byc - cz*bzc;
            float qx = cw*bxc + cx*bwc + cy*bzc - cz*byc;
            float qy = cw*byc - cx*bzc + cy*bwc + cz*bxc;
            float qz = cw*bzc + cx*byc - cy*bxc + cz*bwc;

            char buf[192];
            int n = snprintf(buf, sizeof(buf),
                             "R:%+.1f P:%+.1f Y:%+.1f S:%d,%d,%d,%d "
                             "G:%+.2f,%+.2f,%+.2f "
                             "Q:%+.3f,%+.3f,%+.3f,%+.3f\n",
                             r, p, y, a0, a1, a2, a3,
                             gxs, gys, gzs, qw, qx, qy, qz);
            if (n > 0) {
                /* Serial gets servo state only; full frame goes over BLE. */
                printf("S:%d,%d,%d,%d\n", a0, a1, a2, a3);
                fflush(stdout);
                ble_send(buf);
            }
        }
    }
}

/* ================================================================= */
/* app_main                                                          */
/* ================================================================= */
void app_main(void)
{
    esp_log_level_set("*", ESP_LOG_WARN);
    esp_log_level_set(TAG, ESP_LOG_INFO);

    ESP_LOGI(TAG, "Eshy firmware starting");

    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    state_mtx = xSemaphoreCreateMutex();

    servos_init();
    leds_init();
#if !DISABLE_MPU
    mpu_init();      /* robot must be still during gyro calibration */
#else
    ESP_LOGW(TAG, "MPU disabled (DISABLE_MPU=1) — telemetry IMU will be zero.");
#endif

    ble_start();

    xTaskCreatePinnedToCore(control_task, "control", 6144, NULL, 5, NULL, 1);

    ESP_LOGI(TAG, "Commands: c / s / i / M: / mN: / A: / O: / R:");
    ESP_LOGI(TAG, "Mode: CENTER — waiting for BLE connection");
}
