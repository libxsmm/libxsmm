#include "gimmik.h"

#define NREPS 200000
#define SEP ';'

struct gmm
{
    const char *name;
    void (*fn)(const double * LIBXSMM_RESTRICT, double * LIBXSMM_RESTRICT);
    int beta;
    int m;
    int k;
    int nnz;
};

int main(int argc, char* argv[])
{
    struct gmm mats[] = {
        {"p1/hex/m0", gimmik_p1_hex_m0, 0, 24, 8, 48},
        {"p1/hex/m3", gimmik_p1_hex_m3, 1, 8, 24, 48},
        {"p1/hex/m6", gimmik_p1_hex_m6, 1, 24, 24, 48},
        {"p1/hex/m460", gimmik_p1_hex_m460, 0, 24, 8, 48},
        {"p1/hex/m132", gimmik_p1_hex_m132, 0, 8, 24, 48},
        {"p1/pri/m0", gimmik_p1_pri_m0, 0, 18, 6, 48},
        {"p1/pri/m3", gimmik_p1_pri_m3, 1, 6, 18, 48},
        {"p1/pri/m6", gimmik_p1_pri_m6, 1, 18, 18, 60},
        {"p1/pri/m460", gimmik_p1_pri_m460, 0, 18, 6, 36},
        {"p1/pri/m132", gimmik_p1_pri_m132, 0, 6, 18, 36},
        {"p2/hex/m0", gimmik_p2_hex_m0, 0, 54, 27, 162},
        {"p2/hex/m3", gimmik_p2_hex_m3, 1, 27, 54, 162},
        {"p2/hex/m6", gimmik_p2_hex_m6, 1, 81, 54, 162},
        {"p2/hex/m460", gimmik_p2_hex_m460, 0, 81, 27, 216},
        {"p2/hex/m132", gimmik_p2_hex_m132, 0, 27, 81, 216},
        {"p2/pri/m0", gimmik_p2_pri_m0, 0, 39, 18, 198},
        {"p2/pri/m3", gimmik_p2_pri_m3, 1, 18, 39, 198},
        {"p2/pri/m6", gimmik_p2_pri_m6, 1, 54, 39, 252},
        {"p2/pri/m460", gimmik_p2_pri_m460, 0, 54, 18, 240},
        {"p2/pri/m132", gimmik_p2_pri_m132, 0, 18, 54, 240},
        {"p3/hex/m0", gimmik_p3_hex_m0, 0, 96, 64, 384},
        {"p3/hex/m3", gimmik_p3_hex_m3, 1, 64, 96, 384},
        {"p3/hex/m6", gimmik_p3_hex_m6, 1, 192, 96, 384},
        {"p3/hex/m460", gimmik_p3_hex_m460, 0, 192, 64, 768},
        {"p3/hex/m132", gimmik_p3_hex_m132, 0, 64, 192, 768},
        {"p3/pri/m0", gimmik_p3_pri_m0, 0, 68, 40, 560},
        {"p3/pri/m3", gimmik_p3_pri_m3, 1, 40, 68, 560},
        {"p3/pri/m6", gimmik_p3_pri_m6, 1, 120, 68, 720},
        {"p3/pri/m460", gimmik_p3_pri_m460, 0, 120, 40, 928},
        {"p3/pri/m132", gimmik_p3_pri_m132, 0, 40, 120, 928},
        {"p4/hex/m0", gimmik_p4_hex_m0, 0, 150, 125, 750},
        {"p4/hex/m3", gimmik_p4_hex_m3, 1, 125, 150, 750},
        {"p4/hex/m6", gimmik_p4_hex_m6, 1, 375, 150, 750},
        {"p4/hex/m460", gimmik_p4_hex_m460, 0, 375, 125, 1800},
        {"p4/hex/m132", gimmik_p4_hex_m132, 0, 125, 375, 1800},
        {"p4/pri/m0", gimmik_p4_pri_m0, 0, 105, 75, 1275},
        {"p4/pri/m3", gimmik_p4_pri_m3, 1, 75, 105, 1275},
        {"p4/pri/m6", gimmik_p4_pri_m6, 1, 225, 105, 1650},
        {"p4/pri/m460", gimmik_p4_pri_m460, 0, 225, 75, 2520},
        {"p4/pri/m132", gimmik_p4_pri_m132, 0, 75, 225, 2520},
        {"p5/hex/m0", gimmik_p5_hex_m0, 0, 216, 216, 1296},
        {"p5/hex/m3", gimmik_p5_hex_m3, 1, 216, 216, 1296},
        {"p5/hex/m6", gimmik_p5_hex_m6, 1, 648, 216, 1296},
        {"p5/hex/m460", gimmik_p5_hex_m460, 0, 648, 216, 3888},
        {"p5/hex/m132", gimmik_p5_hex_m132, 0, 216, 648, 3888},
        {"p5/pri/m0", gimmik_p5_pri_m0, 0, 150, 126, 2520},
        {"p5/pri/m3", gimmik_p5_pri_m3, 1, 126, 150, 2520},
        {"p5/pri/m6", gimmik_p5_pri_m6, 1, 378, 150, 3276},
        {"p5/pri/m460", gimmik_p5_pri_m460, 0, 378, 126, 5940},
        {"p5/pri/m132", gimmik_p5_pri_m132, 0, 126, 378, 5940},
        {"p6/hex/m0", gimmik_p6_hex_m0, 0, 294, 343, 2058},
        {"p6/hex/m3", gimmik_p6_hex_m3, 1, 343, 294, 2058},
        {"p6/hex/m6", gimmik_p6_hex_m6, 1, 1029, 294, 2058},
        {"p6/hex/m460", gimmik_p6_hex_m460, 0, 1029, 343, 7056},
        {"p6/hex/m132", gimmik_p6_hex_m132, 0, 343, 1029, 7056},
        {"p6/pri/m0", gimmik_p6_pri_m0, 0, 203, 196, 4508},
        {"p6/pri/m3", gimmik_p6_pri_m3, 1, 196, 203, 4508},
        {"p6/pri/m6", gimmik_p6_pri_m6, 1, 588, 203, 5880},
        {"p6/pri/m460", gimmik_p6_pri_m460, 0, 588, 196, 12096},
        {"p6/pri/m132", gimmik_p6_pri_m132, 0, 196, 588, 12096},
    };
    const int nreps = (1 < argc ? atoi(argv[1]) : (NREPS));
    int i;

    for (i = 0; i < ((int)sizeof(mats) / (int)sizeof(mats[0])); ++i)
    {
        libxsmm_timer_tickint begin, end;
        size_t bsz = sizeof(double)*LDB*mats[i].k;
        size_t csz = sizeof(double)*LDC*mats[i].m;
        double *b = libxsmm_aligned_malloc(bsz, 64);
        double *c = libxsmm_aligned_malloc(csz, 64);
        int j;

        memset(b, 0, bsz);
        memset(c, 0, csz);

        mats[i].fn(b, c);

        begin = libxsmm_timer_tick();
        for (j = 0; j < nreps; ++j) {
            mats[i].fn(b, c);
        }
        end = libxsmm_timer_tick();

        { /* output performance record in CSV format */
          const double dt = libxsmm_timer_duration(begin, end), invdt = 1.0 / dt;
          const double gflops = (size_t)nreps*2*N*(double)mats[i].nnz * invdt / 1e9;
          const double gbytes = (size_t)nreps*(bsz + ((size_t)1 + mats[i].beta)*csz) * invdt / pow(1024, 3);
          printf("%s%c%f%c%f\n", mats[i].name, SEP, gflops, SEP, gbytes);
        }

        libxsmm_free(b);
        libxsmm_free(c);
    }

    return 0;
}
