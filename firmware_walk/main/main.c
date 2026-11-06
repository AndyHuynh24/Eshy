/*
 * Eshy — on-device policy firmware (XIAO ESP32-S3, ESP-IDF).
 *
 * Runs the exported PPO walk policy (model_weights.h) at 25 Hz in a
 * closed loop on IMU + servo state. Same BLE service and telemetry format
 * as ../firmware.
 *
 * BLE commands:
 *   w                start walking
 *   c / x            stop and center
 *   i                IMU only (servos centered, policy idle)
 *   A:a0,a1,a2,a3    robot-frame degree override
 *
 * Telemetry @ 50 Hz:
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
#include "esp_rom_sys.h"
#include "driver/gpio.h"
#include "driver/i2c.h"

#include "esp_log.h"
#include "esp_err.h"
#include "esp_timer.h"
#include "nvs_flash.h"

#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/ble_uuid.h"
#include "host/util/util.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

#include "policy.h"

#define TAG "WALK"

/* ================================================================= */
/* Pin + timing configuration                                        */
/* ================================================================= */
#define FL_PIN  GPIO_NUM_1   /* D0 */
#define FR_PIN  GPIO_NUM_2   /* D1 */
#define BL_PIN  GPIO_NUM_4   /* D3 */
#define BR_PIN  GPIO_NUM_3   /* D2 */

#define I2C_PORT   I2C_NUM_0
#define I2C_SDA    GPIO_NUM_9
#define I2C_SCL    GPIO_NUM_8
#define MPU_ADDR   0x68

#define SERVO_FREQ_HZ 50
#define SERVO_RES     LEDC_TIMER_13_BIT
#define SERVO_MAX     ((1 << 13) - 1)

/* Must match the env the exported policy was trained with (eshy_env.py and
 * the task subclass's JOINT_RANGE_DEG). */
#define ACTION_RANGE_DEG   32     /* JOINT_RANGE_DEG */
#define POLICY_HZ          25     /* CONTROL_HZ      */
#define GAIT_FREQ_HZ       1.0f   /* GAIT_FREQ_HZ    */

static const gpio_num_t      servo_pins[4]     = {FL_PIN, FR_PIN, BL_PIN, BR_PIN};
static const ledc_channel_t  servo_channels[4] = {
    LEDC_CHANNEL_0, LEDC_CHANNEL_1, LEDC_CHANNEL_2, LEDC_CHANNEL_3
};
static const ledc_timer_t    servo_timers[4]   = {
    LEDC_TIMER_0, LEDC_TIMER_1, LEDC_TIMER_2, LEDC_TIMER_3
};
static const char *servo_names[4] = {"FL", "FR", "BL", "BR"};
static const int   SERVO_SIGN[4]  = { -1, +1, -1, +1 };

/* Slew limit (5 deg/tick = 125 deg/s @ 25 Hz, ~ the env's
 * MAX_SPEED_DEG_PER_SEC) plus a 1 ms stagger between servo writes so all
 * four MG90S don't draw inrush current at the same instant. */
#define SLEW_DEG_PER_TICK    5
#define SERVO_STAGGER_US     1000
static int last_robot_deg[4] = { 0, 0, 0, 0 };

/* ================================================================= */
/* Shared state                                                      */
/* ================================================================= */
typedef enum {
    MODE_CENTER = 0,
    MODE_IMU,
    MODE_MANUAL,
    MODE_WALK,
} robot_mode_t;

static SemaphoreHandle_t state_mtx;
static robot_mode_t robot_mode = MODE_CENTER;
static int   current_angles[4] = {90, 90, 90, 90};
static float prev_joint_rad[4] = {0, 0, 0, 0};

static float q_w = 1.0f, q_x = 0.0f, q_y = 0.0f, q_z = 0.0f;
static const float madgwick_beta = 0.1f;
static float last_gx = 0.0f, last_gy = 0.0f, last_gz = 0.0f;

static uint8_t  own_addr_type = 0;
static uint16_t conn_handle   = 0;
static uint16_t tx_val_handle = 0;
static bool     tx_subscribed = false;

/* ================================================================= */
/* Servo driver                                                      */
/* ================================================================= */
static uint32_t angle_to_duty(int angle)
{
    if (angle < 0)   angle = 0;
    if (angle > 180) angle = 180;
    int pulse_us = 500 + (angle * 2000) / 180;
    return ((uint32_t)pulse_us * SERVO_MAX) / 20000;
}

static void servo_write(int i, int angle)
{
    if (i < 0 || i > 3) return;
    if (angle < 0)   angle = 0;
    if (angle > 180) angle = 180;
    ledc_set_duty(LEDC_LOW_SPEED_MODE, servo_channels[i], angle_to_duty(angle));
    ledc_update_duty(LEDC_LOW_SPEED_MODE, servo_channels[i]);

    xSemaphoreTake(state_mtx, portMAX_DELAY);
    current_angles[i] = angle;
    xSemaphoreGive(state_mtx);
}

/* Clip + slew-limit 4 robot-frame commands, then write with stagger.
 * Shared by A: overrides and the on-device policy. */
static void robot_write_4_staggered(const int rd[4])
{
    int target[4];
    for (int i = 0; i < 4; i++) {
        int t = rd[i];
        if (t >  ACTION_RANGE_DEG) t =  ACTION_RANGE_DEG;
        if (t < -ACTION_RANGE_DEG) t = -ACTION_RANGE_DEG;
        int delta = t - last_robot_deg[i];
        if (delta >  SLEW_DEG_PER_TICK) delta =  SLEW_DEG_PER_TICK;
        if (delta < -SLEW_DEG_PER_TICK) delta = -SLEW_DEG_PER_TICK;
        target[i] = last_robot_deg[i] + delta;
    }
    for (int i = 0; i < 4; i++) {
        servo_write(i, 90 + SERVO_SIGN[i] * target[i]);
        last_robot_deg[i] = target[i];
        if (i < 3) esp_rom_delay_us(SERVO_STAGGER_US);
    }
}

static void servos_init(void)
{
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
            .duty       = angle_to_duty(90),
            .hpoint     = 0,
        };
        ESP_ERROR_CHECK(ledc_channel_config(&ch));

        ESP_LOGI(TAG, "  %s on GPIO %d -> channel %d, timer %d",
                 servo_names[i], servo_pins[i], servo_channels[i], servo_timers[i]);
    }
    for (int i = 0; i < 4; i++) servo_write(i, 90);
}

static void center_all(void)
{
    for (int i = 0; i < 4; i++) servo_write(i, 90);
}

/* ================================================================= */
/* MPU-6050 + Madgwick                                               */
/* ================================================================= */
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

    uint8_t reg = 0x75, who = 0;
    i2c_master_write_read_device(I2C_PORT, MPU_ADDR, &reg, 1, &who, 1,
                                  pdMS_TO_TICKS(100));
    ESP_LOGI(TAG, "MPU6050 WHO_AM_I: 0x%02X %s",
             who, who == 0x68 ? "(OK)" : "(ERROR!)");

    mpu_write_reg(0x6B, 0x00);
    mpu_write_reg(0x1B, 0x08);
    mpu_write_reg(0x1C, 0x00);
    mpu_write_reg(0x1A, 0x03);
}

static void mpu_read(float *ax, float *ay, float *az,
                     float *gx, float *gy, float *gz)
{
    uint8_t reg = 0x3B, d[14];
    i2c_master_write_read_device(I2C_PORT, MPU_ADDR, &reg, 1, d, 14,
                                  pdMS_TO_TICKS(100));
    int16_t rax = (int16_t)((d[0]  << 8) | d[1]);
    int16_t ray = (int16_t)((d[2]  << 8) | d[3]);
    int16_t raz = (int16_t)((d[4]  << 8) | d[5]);
    int16_t rgx = (int16_t)((d[8]  << 8) | d[9]);
    int16_t rgy = (int16_t)((d[10] << 8) | d[11]);
    int16_t rgz = (int16_t)((d[12] << 8) | d[13]);

    *ax = rax / 16384.0f;
    *ay = ray / 16384.0f;
    *az = raz / 16384.0f;
    *gx = (rgx / 65.5f) * (float)M_PI / 180.0f;
    *gy = (rgy / 65.5f) * (float)M_PI / 180.0f;
    *gz = (rgz / 65.5f) * (float)M_PI / 180.0f;
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

    float qdw = 0.5f*(-qx*gx - qy*gy - qz*gz) - madgwick_beta*sw;
    float qdx = 0.5f*( qw*gx + qy*gz - qz*gy) - madgwick_beta*sx;
    float qdy = 0.5f*( qw*gy - qx*gz + qz*gx) - madgwick_beta*sy;
    float qdz = 0.5f*( qw*gz + qx*gy - qy*gx) - madgwick_beta*sz;

    qw += qdw * dt; qx += qdx * dt; qy += qdy * dt; qz += qdz * dt;
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
static void set_mode(robot_mode_t m)
{
    xSemaphoreTake(state_mtx, portMAX_DELAY);
    robot_mode = m;
    xSemaphoreGive(state_mtx);
}

static void handle_line(const char *raw, int len)
{
    while (len > 0 && (*raw == ' ' || *raw == '\t' || *raw == '\r')) { raw++; len--; }
    while (len > 0 && (raw[len-1] == ' ' || raw[len-1] == '\t' || raw[len-1] == '\r')) len--;
    if (len <= 0) return;

    if (len == 1) {
        switch (raw[0]) {
        case 'c': case 'C': case 'x': case 'X':
            set_mode(MODE_CENTER); center_all();
            ESP_LOGI(TAG, "Mode: CENTER");
            return;
        case 'i': case 'I':
            set_mode(MODE_IMU); center_all();
            ESP_LOGI(TAG, "Mode: IMU");
            return;
        case 'w': case 'W':
            set_mode(MODE_WALK);
            ESP_LOGI(TAG, "Mode: WALK (policy @ %d Hz)", POLICY_HZ);
            return;
        }
        return;
    }

    /* "A:a0,a1,a2,a3" — robot-frame degree override */
    if (len >= 2 && (raw[0] == 'A' || raw[0] == 'a') && raw[1] == ':') {
        int rd[4] = {0, 0, 0, 0};
        const char *p = raw + 2, *end = raw + len;
        int idx = 0;
        while (p < end && idx < 4) {
            char *ep;
            long v = strtol(p, &ep, 10);
            if (ep == p) break;
            rd[idx++] = (int)v;
            p = ep;
            if (p < end && *p == ',') p++;
        }
        set_mode(MODE_MANUAL);
        robot_write_4_staggered(rd);
    }
}

/* ================================================================= */
/* BLE UART (Nordic UART Service)                                    */
/* ================================================================= */
static const ble_uuid128_t nus_svc_uuid = BLE_UUID128_INIT(
    0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0,
    0x93, 0xf3, 0xa3, 0xb5, 0x01, 0x00, 0x40, 0x6e);
static const ble_uuid128_t nus_rx_uuid = BLE_UUID128_INIT(
    0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0,
    0x93, 0xf3, 0xa3, 0xb5, 0x02, 0x00, 0x40, 0x6e);
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
                handle_line(rx_line_buf, rx_line_len);
                rx_line_len = 0;
            }
        } else if (rx_line_len < (int)sizeof(rx_line_buf) - 1) {
            rx_line_buf[rx_line_len++] = c;
        } else {
            rx_line_len = 0;
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
    { .type = BLE_GATT_SVC_TYPE_PRIMARY,
      .uuid = &nus_svc_uuid.u,
      .characteristics = (struct ble_gatt_chr_def[]) {
          { .uuid = &nus_rx_uuid.u, .access_cb = gatt_nus_access,
            .flags = BLE_GATT_CHR_F_WRITE | BLE_GATT_CHR_F_WRITE_NO_RSP },
          { .uuid = &nus_tx_uuid.u, .access_cb = gatt_nus_access,
            .flags = BLE_GATT_CHR_F_NOTIFY, .val_handle = &tx_val_handle },
          { 0 }
      }, },
    { 0 }
};

static void ble_send(const char *msg)
{
    if (conn_handle == 0 || !tx_subscribed) return;
    int len = (int)strlen(msg);
    for (int i = 0; i < len; i += 20) {
        int chunk = (len - i < 20) ? (len - i) : 20;
        struct os_mbuf *om = ble_hs_mbuf_from_flat(msg + i, chunk);
        if (!om) break;
        int rc = ble_gatts_notify_custom(conn_handle, tx_val_handle, om);
        if (rc != 0) break;
    }
}

static int  gap_event_cb(struct ble_gap_event *event, void *arg);

static void start_advertising(void)
{
    struct ble_hs_adv_fields fields = {0};
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.name = (uint8_t *)"TechTalkies";
    fields.name_len = strlen("TechTalkies");
    fields.name_is_complete = 1;
    fields.tx_pwr_lvl_is_present = 1;
    fields.tx_pwr_lvl = BLE_HS_ADV_TX_PWR_LVL_AUTO;

    if (ble_gap_adv_set_fields(&fields) != 0) return;

    struct ble_gap_adv_params params = {0};
    params.conn_mode = BLE_GAP_CONN_MODE_UND;
    params.disc_mode = BLE_GAP_DISC_MODE_GEN;
    int rc = ble_gap_adv_start(own_addr_type, NULL, BLE_HS_FOREVER,
                                &params, gap_event_cb, NULL);
    if (rc != 0) ESP_LOGE(TAG, "adv_start: %d", rc);
    else         ESP_LOGI(TAG, "Advertising as 'TechTalkies'");
}

static int gap_event_cb(struct ble_gap_event *event, void *arg)
{
    (void)arg;
    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
        if (event->connect.status == 0) {
            conn_handle = event->connect.conn_handle;
            ESP_LOGI(TAG, "BLE connected");
        } else {
            start_advertising();
        }
        break;
    case BLE_GAP_EVENT_DISCONNECT:
        conn_handle = 0; tx_subscribed = false;
        set_mode(MODE_CENTER); center_all();
        start_advertising();
        break;
    case BLE_GAP_EVENT_SUBSCRIBE:
        if (event->subscribe.attr_handle == tx_val_handle)
            tx_subscribed = event->subscribe.cur_notify;
        break;
    default: break;
    }
    return 0;
}

static void on_sync(void)
{
    ble_hs_util_ensure_addr(0);
    ble_hs_id_infer_auto(0, &own_addr_type);
    start_advertising();
}
static void on_reset(int reason) { ESP_LOGW(TAG, "BLE reset: %d", reason); }

static void ble_host_task(void *param)
{
    (void)param;
    nimble_port_run();
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
    ESP_ERROR_CHECK(ble_svc_gap_device_name_set("TechTalkies"));
    nimble_port_freertos_init(ble_host_task);
}

/* ================================================================= */
/* Policy step                                                       */
/* ================================================================= */
static float phase = 0.0f;
static const float policy_dt = 1.0f / (float)POLICY_HZ;
static const float joint_range_rad = (float)ACTION_RANGE_DEG * (float)M_PI / 180.0f;

static void policy_step(void)
{
    /* Must match EshyEnv._get_obs exactly. */
    float obs[POLICY_OBS_DIM];

    xSemaphoreTake(state_mtx, portMAX_DELAY);
    int a0 = current_angles[0], a1 = current_angles[1];
    int a2 = current_angles[2], a3 = current_angles[3];
    float qx = q_x, qy = q_y, qz = q_z;
    float gx = last_gx, gy = last_gy, gz = last_gz;
    xSemaphoreGive(state_mtx);

    float jp[4];
    jp[0] = (a0 - 90) * (float)SERVO_SIGN[0] * (float)M_PI / 180.0f;
    jp[1] = (a1 - 90) * (float)SERVO_SIGN[1] * (float)M_PI / 180.0f;
    jp[2] = (a2 - 90) * (float)SERVO_SIGN[2] * (float)M_PI / 180.0f;
    jp[3] = (a3 - 90) * (float)SERVO_SIGN[3] * (float)M_PI / 180.0f;

    float jv[4];
    for (int i = 0; i < 4; i++) {
        jv[i] = (jp[i] - prev_joint_rad[i]) / policy_dt;
        prev_joint_rad[i] = jp[i];
    }

    obs[0]  = jp[0]; obs[1]  = jp[1]; obs[2]  = jp[2]; obs[3]  = jp[3];
    obs[4]  = jv[0]; obs[5]  = jv[1]; obs[6]  = jv[2]; obs[7]  = jv[3];
    obs[8]  = qx;    obs[9]  = qy;    obs[10] = qz;
    obs[11] = gx;    obs[12] = gy;    obs[13] = gz;
    obs[14] = sinf(2.0f * (float)M_PI * phase);
    obs[15] = cosf(2.0f * (float)M_PI * phase);

    float action[POLICY_ACTION_DIM];
    policy_forward(obs, action);

    int rd[4];
    for (int i = 0; i < 4; i++) {
        float a = action[i];
        if (a >  1.0f) a =  1.0f;
        if (a < -1.0f) a = -1.0f;
        rd[i] = (int)(a * joint_range_rad * 180.0f / (float)M_PI);
    }
    robot_write_4_staggered(rd);

    phase += GAIT_FREQ_HZ * policy_dt;
    if (phase >= 1.0f) phase -= 1.0f;
}

/* ================================================================= */
/* Main control loop                                                 */
/* ================================================================= */
static void control_task(void *param)
{
    (void)param;
    const TickType_t period = pdMS_TO_TICKS(10);      /* 100 Hz */
    TickType_t last_wake = xTaskGetTickCount();
    int64_t last_us = esp_timer_get_time();
    int imu_ticks = 0, bc_ticks = 0;
    const int POLICY_DIV = 100 / POLICY_HZ;           /* 100 Hz / 25 Hz = 4 */

    while (1) {
        vTaskDelayUntil(&last_wake, period);

        int64_t now_us = esp_timer_get_time();
        float dt = (now_us - last_us) / 1e6f;
        if (dt <= 0.0f || dt > 0.2f) dt = 0.01f;
        last_us = now_us;

        float ax, ay, az, gx, gy, gz;
        mpu_read(&ax, &ay, &az, &gx, &gy, &gz);
        madgwick_update(ax, ay, az, gx, gy, gz, dt);
        last_gx = gx; last_gy = gy; last_gz = gz;

        float r, p, y; quat_to_rpy(&r, &p, &y);
        xSemaphoreTake(state_mtx, portMAX_DELAY);
        robot_mode_t m = robot_mode;
        xSemaphoreGive(state_mtx);

        if (m == MODE_WALK) {
            if (++imu_ticks >= POLICY_DIV) {
                imu_ticks = 0;
                policy_step();
            }
        } else {
            imu_ticks = 0;
            phase = 0.0f;
        }

        if (++bc_ticks >= 2) {     /* 50 Hz */
            bc_ticks = 0;
            xSemaphoreTake(state_mtx, portMAX_DELAY);
            int s0 = current_angles[0], s1 = current_angles[1];
            int s2 = current_angles[2], s3 = current_angles[3];
            float qw = q_w, qx = q_x, qy = q_y, qz = q_z;
            float _gx = last_gx, _gy = last_gy, _gz = last_gz;
            xSemaphoreGive(state_mtx);

            char buf[192];
            int n = snprintf(buf, sizeof(buf),
                             "R:%+.1f P:%+.1f Y:%+.1f S:%d,%d,%d,%d "
                             "G:%+.2f,%+.2f,%+.2f "
                             "Q:%+.3f,%+.3f,%+.3f,%+.3f\n",
                             r, p, y, s0, s1, s2, s3,
                             _gx, _gy, _gz, qw, qx, qy, qz);
            if (n > 0) {
                fwrite(buf, 1, (size_t)n, stdout);
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
    ESP_LOGI(TAG, "Eshy walk firmware starting");

    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    state_mtx = xSemaphoreCreateMutex();

    servos_init();
    mpu_init();
    vTaskDelay(pdMS_TO_TICKS(500));

    ble_start();

    xTaskCreatePinnedToCore(control_task, "control", 8192, NULL, 5, NULL, 1);

    ESP_LOGI(TAG, "Ready. BLE commands: w=walk, c/x=stop, i=imu-only");
    ESP_LOGI(TAG, "Policy: obs=%d, hidden=%d, action=%d",
             POLICY_OBS_DIM, POLICY_HIDDEN_DIM, POLICY_ACTION_DIM);
}
