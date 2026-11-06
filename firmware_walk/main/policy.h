/*
 * Tiny MLP forward pass for PPO policy inference on-device.
 *
 * Network exported by export_model.py into model_weights.h:
 *   obs(16) -> Linear(128) -> Tanh -> Linear(128) -> Tanh -> Linear(4)
 */
#pragma once

#include <stddef.h>

#define POLICY_OBS_DIM     16
#define POLICY_HIDDEN_DIM  128
#define POLICY_ACTION_DIM  4

/* Returns the raw action mean (shape [POLICY_ACTION_DIM]).  Caller is
 * responsible for clipping to [-1, 1] and scaling to the hip range. */
void policy_forward(const float *obs, float *action_out);
