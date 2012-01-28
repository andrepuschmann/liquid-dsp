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

void dotprod_crcf_run4(float *_h,
                       float complex *_x,
                       unsigned int _n,
                       float complex * _y)
{
    float complex r=0;

    // t = 4*(floor(_n/4))
    unsigned int t=(_n>>2)<<2; 

    // compute dotprod in groups of 4
    unsigned int i;
    for (i=0; i<t; i+=4) {
        r += _h[i]   * _x[i];
        r += _h[i+1] * _x[i+1];
        r += _h[i+2] * _x[i+2];
        r += _h[i+3] * _x[i+3];
    }

    // clean up remaining
    for ( ; i<_n; i++)
        r += _h[i] * _x[i];

    *_y = r;
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

