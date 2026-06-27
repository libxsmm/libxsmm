#include <check.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "include/libxsmm_macros.h"

START_TEST(test_libxsmm_snprintf_bounds)
{
    // Invariant: LIBXSMM_SNPRINTF never writes beyond the declared buffer size
    const char *payloads[] = {
        "%s",                    // Valid format specifier
        "AAAAAAAAAAAAAAAAAAAA",  // Long string exceeding typical buffer
        "%100s",                 // Width specifier that could overflow
    };
    int num_payloads = sizeof(payloads) / sizeof(payloads[0]);

    for (int i = 0; i < num_payloads; i++) {
        const size_t buffer_size = 16;
        char buffer[buffer_size];
        char canary[32];
        
        // Setup canary after buffer to detect overflow
        memset(buffer, 0, buffer_size);
        memset(canary, 0x42, sizeof(canary));
        
        // Call the vulnerable macro with controlled size
        LIBXSMM_SNPRINTF(buffer, buffer_size, payloads[i], "test");
        
        // Verify canary remains unchanged (no overflow)
        for (size_t j = 0; j < sizeof(canary); j++) {
            ck_assert_msg(canary[j] == 0x42, 
                         "Buffer overflow detected! Canary corrupted at byte %zu", j);
        }
        
        // Additional check: ensure null termination within bounds
        int null_found = 0;
        for (size_t j = 0; j < buffer_size; j++) {
            if (buffer[j] == '\0') {
                null_found = 1;
                break;
            }
        }
        ck_assert_msg(null_found || buffer[buffer_size - 1] == '\0',
                     "String not properly terminated within buffer bounds");
    }
}
END_TEST

Suite *security_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Security");
    tc_core = tcase_create("Core");

    tcase_add_test(tc_core, test_libxsmm_snprintf_bounds);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    Suite *s;
    SRunner *sr;

    s = security_suite();
    sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}