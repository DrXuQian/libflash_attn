#!/bin/bash

# Auto monitoring script for compilation and testing
# This script will monitor the compilation, run tests, and save results

BUILD_DIR="/home/qianxu/flash-attention-homemade/libflash_attn/build"
REPO_DIR="/home/qianxu/flash-attention-homemade/libflash_attn"
LOG_FILE="${BUILD_DIR}/auto_monitor.log"
RESULT_FILE="${BUILD_DIR}/test_results.txt"

echo "=== Auto Monitor Started at $(date) ===" | tee -a ${LOG_FILE}
echo "Monitoring compilation in ${BUILD_DIR}" | tee -a ${LOG_FILE}

# Wait for make to complete
while pgrep -f "make.*full_compile.log" > /dev/null; do
    echo "[$(date +%H:%M:%S)] Compilation still running..." | tee -a ${LOG_FILE}
    sleep 30
done

echo "[$(date +%H:%M:%S)] Compilation process finished" | tee -a ${LOG_FILE}

# Check compilation result
if [ -f "${BUILD_DIR}/test_fp8_flash_attn" ]; then
    echo "[$(date +%H:%M:%S)] ✓ Compilation SUCCESS - test_fp8_flash_attn binary created" | tee -a ${LOG_FILE}

    # Run the test
    echo "[$(date +%H:%M:%S)] Running test_fp8_flash_attn..." | tee -a ${LOG_FILE}
    cd ${BUILD_DIR}
    ./test_fp8_flash_attn 2>&1 | tee ${RESULT_FILE}
    TEST_EXIT_CODE=${PIPESTATUS[0]}

    if [ ${TEST_EXIT_CODE} -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ✓ TEST PASSED" | tee -a ${LOG_FILE}
        echo "SUCCESS" > ${BUILD_DIR}/final_status.txt
    else
        echo "[$(date +%H:%M:%S)] ✗ TEST FAILED with exit code ${TEST_EXIT_CODE}" | tee -a ${LOG_FILE}
        echo "TEST_FAILED" > ${BUILD_DIR}/final_status.txt
    fi

    # Commit and push if successful
    if [ ${TEST_EXIT_CODE} -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] Committing test results..." | tee -a ${LOG_FILE}
        cd ${REPO_DIR}
        git add -A
        git commit -m "Complete FP8 Flash Attention implementation and testing

- Added FP8 E4M3 kernel instantiations for head dims 64, 128, 256
- Added test program with reference implementation
- Fixed Flash_fwd_params include path
- Compilation and tests passed successfully

🤖 Generated with Claude Code" || echo "Nothing to commit" | tee -a ${LOG_FILE}

        git push 2>&1 | tee -a ${LOG_FILE}
    fi
else
    echo "[$(date +%H:%M:%S)] ✗ Compilation FAILED - binary not found" | tee -a ${LOG_FILE}
    echo "COMPILE_FAILED" > ${BUILD_DIR}/final_status.txt

    # Check for errors in log
    echo "[$(date +%H:%M:%S)] Last 50 lines of compilation log:" | tee -a ${LOG_FILE}
    tail -50 ${BUILD_DIR}/full_compile.log | tee -a ${LOG_FILE}
fi

echo "=== Auto Monitor Finished at $(date) ===" | tee -a ${LOG_FILE}
echo "Check ${LOG_FILE} for full details"
echo "Check ${RESULT_FILE} for test output (if test ran)"
