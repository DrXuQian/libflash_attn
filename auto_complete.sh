#!/bin/bash

# Auto completion script - monitors compilation and runs tests automatically
REPO_DIR="/home/qianxu/flash-attention-homemade/libflash_attn"
BUILD_DIR="${REPO_DIR}/build"
LOG_FILE="${REPO_DIR}/auto_complete.log"

echo "=== Auto Completion Started at $(date) ===" | tee ${LOG_FILE}

# Wait for compilation to finish (check every minute)
echo "Waiting for compilation to complete..." | tee -a ${LOG_FILE}
while true; do
    if ! pgrep -f "make.*compile_final.log" > /dev/null; then
        echo "[$(date +%H:%M:%S)] Compilation process finished" | tee -a ${LOG_FILE}
        break
    fi
    sleep 60
done

# Check if binary was created
if [ -f "${BUILD_DIR}/test_fp8_flash_attn" ]; then
    echo "[$(date +%H:%M:%S)] ✓ COMPILATION SUCCESS" | tee -a ${LOG_FILE}

    # Run test
    echo "[$(date +%H:%M:%S)] Running test..." | tee -a ${LOG_FILE}
    cd ${BUILD_DIR}
    ./test_fp8_flash_attn 2>&1 | tee test_results.txt
    TEST_EXIT=$?

    if [ $TEST_EXIT -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ✓ TEST PASSED" | tee -a ${LOG_FILE}

        # Commit and push
        cd ${REPO_DIR}
        git add -A
        git commit -m "Complete FP8 Flash Attention - All tests passed

- FP8 E4M3 kernels for head dims 64, 128, 256
- Test program with reference implementation
- Fixed all include dependencies
- Compilation and testing successful

🤖 Generated with Claude Code" 2>&1 | tee -a ${LOG_FILE}
        git push 2>&1 | tee -a ${LOG_FILE}

        echo "SUCCESS" > ${BUILD_DIR}/final_status.txt
    else
        echo "[$(date +%H:%M:%S)] ✗ TEST FAILED (exit code: $TEST_EXIT)" | tee -a ${LOG_FILE}
        echo "TEST_FAILED" > ${BUILD_DIR}/final_status.txt
    fi
else
    echo "[$(date +%H:%M:%S)] ✗ COMPILATION FAILED" | tee -a ${LOG_FILE}
    tail -100 ${BUILD_DIR}/compile_final.log | tee -a ${LOG_FILE}
    echo "COMPILE_FAILED" > ${BUILD_DIR}/final_status.txt
fi

echo "=== Auto Completion Finished at $(date) ===" | tee -a ${LOG_FILE}
echo "Final status saved to ${BUILD_DIR}/final_status.txt"
