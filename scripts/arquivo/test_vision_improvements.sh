#!/bin/bash
echo "Testing vision system improvements..."
echo "Target: Verify NEAR/FAR zone sensory channels and improved lookahead"
echo ""

echo "Running 5-generation test with 10 agents..."
timeout 600 ./target/release/universe --headless --gens 5 --swarm 10 2>&1 | tee test_vision_results.log | grep -E "GEN|Deaths|NET"

echo ""
echo "=== RESULTS ==="
if [ -f test_vision_results.log ]; then
    echo "Generation performance:"
    grep "^📊 GEN" test_vision_results.log
    echo ""
    echo "Elite pool status:"
    grep -E "elite_pool|champion" test_vision_results.log | tail -5 || echo "(No elite pool output)"
fi
