/*
 * Copyright (c) 2007, 2009 Joseph Gaeddert
 * Copyright (c) 2007, 2009 Virginia Polytechnic Institute & State University
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
 * MERCHANTABILITY or FITNESS FOR A PARfloatCULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with liquid.  If not, see <http://www.gnu.org/licenses/>.
 */

// 
// Floating-point dot product using ARM NEON instructions
//

#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>

#include "liquid.internal.h"

#define DEBUG_DOTPROD_RRRF_NEON   0

// basic dot product

void dotprod_rrrf_run(float *_h,
                      float *_x,
                      unsigned int _n,
                      float * _y)
{
    float r=0;
    unsigned int i;
    for (i=0; i<_n; i++)
        r += _h[i] * _x[i];
    *_y = r;
}

void dotprod_rrrf_run4(float *_h,
                       float *_x,
                       unsigned int _n,
                       float * _y)
{
    float32x4_t h_vec;
    float32x4_t x_vec;
    float32x4_t acc_vec = vdupq_n_f32(0); /* init to zero */
    float32_t res = 0;
    unsigned int i;

    for (i = 0; i < _n / 4; i++) {
        /* load input vectors, four floats at once */
        h_vec = vld1q_f32(&_h[i * 4]);
        x_vec = vld1q_f32(&_x[i * 4]);
        /* multiply and add them to result vector */
        acc_vec = vmlaq_f32 (acc_vec, h_vec, x_vec);
    }

    if (_n % 4) {
        for (i = _n - (_n % 4) ; i < _n; i++) {
           res += _h[i] * _x[i];
        }
    }

    res += vgetq_lane_f32(acc_vec, 0);
    res += vgetq_lane_f32(acc_vec, 1);
    res += vgetq_lane_f32(acc_vec, 2);
    res += vgetq_lane_f32(acc_vec, 3);

    *_y = res;
}


//
// structured dot product
//

struct dotprod_rrrf_s {
    float * h;             // coefficients array
    unsigned int n;     // length
};

// create the structured dotprod object
dotprod_rrrf dotprod_rrrf_create(float * _h, unsigned int _n)
{
    dotprod_rrrf q = (dotprod_rrrf) malloc(sizeof(struct dotprod_rrrf_s));
    q->n = _n;

    // allocate memory for coefficients
    q->h = (float*) malloc((q->n)*sizeof(float));

    // move coefficients
    memmove(q->h, _h, (q->n)*sizeof(float));

    // return object
    return q;
}

// re-create the structured dotprod object
dotprod_rrrf dotprod_rrrf_recreate(dotprod_rrrf _dp,
                                   float * _h,
                                   unsigned int _n)
{
    // completely destroy and re-create dotprod object
    dotprod_rrrf_destroy(_dp);
    _dp = dotprod_rrrf_create(_h,_n);
    return _dp;
}

// destroy the structured dotprod object
void dotprod_rrrf_destroy(dotprod_rrrf _q)
{
    free(_q->h);    // free coefficients memory
    free(_q);       // free main object memory
}

// print the dotprod object
void dotprod_rrrf_print(dotprod_rrrf _q)
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
void dotprod_rrrf_execute(dotprod_rrrf _q,
                          float * _x,
                          float * _y)
{
    dotprod_rrrf_run4(_q->h, _x, _q->n, _y);
}

