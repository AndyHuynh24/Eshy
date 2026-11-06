/*
 * MLP forward pass matching the SB3 PPO default MlpPolicy:
 *   h1 = tanh(W1 @ obs + b1)          W1: (128, 16)
 *   h2 = tanh(W2 @ h1  + b2)          W2: (128, 128)
 *   a  = W3 @ h2 + b3                 W3: (4, 128)
 *
 * PyTorch Linear stores weights in (out_features, in_features) order, which
 * is exactly how export_model.py flattens them. Row-major dot product below
 * is therefore W[j*cols + i] * x[i].
 */
#include "policy.h"
#include "model_weights.h"

#include <math.h>
#include <string.h>

static inline float fast_tanhf(float x)
{
    /* Clamp to avoid NaNs from extreme pre-activations. */
    if (x >  10.0f) return  1.0f;
    if (x < -10.0f) return -1.0f;
    return tanhf(x);
}

static void linear(const float *W, const float *b, int rows, int cols,
                   const float *x, float *y)
{
    for (int j = 0; j < rows; j++) {
        float acc = b[j];
        const float *wj = W + (size_t)j * cols;
        for (int i = 0; i < cols; i++) {
            acc += wj[i] * x[i];
        }
        y[j] = acc;
    }
}

void policy_forward(const float *obs, float *action_out)
{
    static float h1[POLICY_HIDDEN_DIM];
    static float h2[POLICY_HIDDEN_DIM];

    linear(policy_0_weight, policy_0_bias,
           policy_0_weight_rows, policy_0_weight_cols,
           obs, h1);
    for (int i = 0; i < POLICY_HIDDEN_DIM; i++) h1[i] = fast_tanhf(h1[i]);

    linear(policy_2_weight, policy_2_bias,
           policy_2_weight_rows, policy_2_weight_cols,
           h1, h2);
    for (int i = 0; i < POLICY_HIDDEN_DIM; i++) h2[i] = fast_tanhf(h2[i]);

    linear(action_weight, action_bias,
           action_weight_rows, action_weight_cols,
           h2, action_out);
}
