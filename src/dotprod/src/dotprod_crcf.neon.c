/*
 * Copyright (c) 2011 Joseph Gaeddert
 * Copyright (c) 2011 Virginia Polytechnic Institute & State University
 *
 * This file is part of liquid.
 *
 * liquid is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * liquid is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with liquid.  If not, see <http://www.gnu.org/licenses/>.
 */

// 
// Complex floating-point dot product using ARM NEON instructions
//

#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>

#include "liquid.internal.h"

#define DEBUG_DOTPROD_CRCF_NEON   0

// basic dot product

void dotprod_crcf_run(float *_h,
                      float complex *_x,
                      unsigned int _n,
                      float complex * _y)
{
    float complex r=0;
    unsigned int i;
    for (i=0; i<_n; i++)
        r += _h[i] * _x[i];
    *_y = r;
}

// (x + yi) u = xu + yu i
void dotprod_crcf_run4(float *_h,
                       float complex *_x,
                       unsigned int _n,
                       float complex * _y)
{
    float32x2_t h_1;
    float32x2_t h_2;
    float32x4_t x_float_vec;
    float32x4_t x_comp_vec;
    float32x4_t acc_vec = vdupq_n_f32(0); /* init to zero */
    float complex res = 0;

    /* process two complex floats at once */
    unsigned int i = 0;
    for (i = 0; i < _n / 2; i++) {
        /* load floats and combine to one */
        h_1 = vld1_dup_f32 (&_h[i * 2]);
        h_2 = vld1_dup_f32 (&_h[(i * 2) + 1]);
        x_float_vec = vcombine_f32(h_1, h_2);
        /* load two complex floats, treat real and imaginary part as float */
        x_comp_vec = vld1q_f32((float *)&_x[i * 2]);
        /* multiply and add them to result vector */
        acc_vec = vmlaq_f32 (acc_vec, x_float_vec, x_comp_vec);
    }
    
    /* take care of the rest */
    if (_n % 4) {
        for (i = _n - (_n % 2) ; i < _n; i++) {
           res += _h[i] * _x[i];
        }
    }

    /* split result and add them up */
    h_1 = vget_high_f32(acc_vec);
    h_2 = vget_low_f32(acc_vec);
    h_2 = vadd_f32(h_1, h_2); 
    res += vget_lane_f32(h_2, 0) + I * vget_lane_f32(h_2, 1);
    
    *_y = res;
}


//
// structured dot product
//

struct dotprod_crcf_s {
    float * h;          // coefficients array
    unsigned int n;     // length
};

// create the structured dotprod object
dotprod_crcf dotprod_crcf_create(float * _h, unsigned int _n)
{
    dotprod_crcf q = (dotprod_crcf) malloc(sizeof(struct dotprod_crcf_s));
    q->n = _n;

    // allocate memory for coefficients
    q->h = (float*) malloc((q->n)*sizeof(float));

    // move coefficients
    memmove(q->h, _h, (q->n)*sizeof(float));

    // return object
    return q;
}

// re-create the structured dotprod object
dotprod_crcf dotprod_crcf_recreate(dotprod_crcf _dp,
                                   float * _h,
                                   unsigned int _n)
{
    // completely destroy and re-create dotprod object
    dotprod_crcf_destroy(_dp);
    _dp = dotprod_crcf_create(_h,_n);
    return _dp;
}

// destroy the structured dotprod object
void dotprod_crcf_destroy(dotprod_crcf _q)
{
    free(_q->h);    // free coefficients memory
    free(_q);       // free main object memory
}

// print the dotprod object
void dotprod_crcf_print(dotprod_crcf _q)
{
    printf("dotprod [%u elements]:\n", _q->n);
    unsigned int i;
    for (i=0; i<_q->n; i++) {
        printf("  %4u: %12.8f + j*%12.8f\n", i,
                                             crealf(_q->h[i]),
                                             cimagf(_q->h[i]));
    }
}

// exectue vectorized structured inner dot product
void dotprod_crcf_execute(dotprod_crcf _q,
                          float complex * _x,
                          float complex * _y)
{
    dotprod_crcf_run4(_q->h, _x, _q->n, _y);
}

