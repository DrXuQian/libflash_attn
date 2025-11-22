#!/bin/bash
echo "========================================"
echo "FP8 Flash Attention Build & Test Status"
echo "========================================"
echo ""

if [ -f final_status.txt ]; then
    STATUS=$(cat final_status.txt)
    echo "Final Status: $STATUS"
    echo ""
    
    if [ "$STATUS" == "SUCCESS" ]; then
        echo "✓ Compilation successful"
        echo "✓ Test passed"
        echo ""
        echo "Test Results:"
        cat test_results.txt
    elif [ "$STATUS" == "TEST_FAILED" ]; then
        echo "✓ Compilation successful"
        echo "✗ Test failed"
        echo ""
        echo "Test Output:"
        cat test_results.txt
    elif [ "$STATUS" == "COMPILE_FAILED" ]; then
        echo "✗ Compilation failed"
        echo ""
        echo "Last errors:"
        tail -50 full_compile.log
    fi
else
    echo "Status: Still running or not started"
    echo ""
    
    if pgrep -f "make.*full_compile" > /dev/null; then
        echo "Compilation is still running..."
        if [ -f full_compile.log ]; then
            LAST_LINE=$(tail -1 full_compile.log)
            echo "Last output: $LAST_LINE"
        fi
    else
        echo "Compilation finished, checking results..."
        if [ -f test_fp8_flash_attn ]; then
            echo "✓ Binary exists, tests may be running"
        else
            echo "✗ Binary not found, compilation may have failed"
        fi
    fi
fi

echo ""
echo "Full logs available in:"
echo "  - full_compile.log (compilation)"
echo "  - auto_monitor.log (auto monitor)"
echo "  - test_results.txt (test output)"
echo "========================================"
