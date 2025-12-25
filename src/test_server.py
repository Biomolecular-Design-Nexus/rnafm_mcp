#!/usr/bin/env python3
"""
Test script for RNA-FM MCP server
"""

import json
import subprocess
import sys
from pathlib import Path

def test_sync_tool():
    """Test a synchronous tool call"""
    print("Testing synchronous RNA embeddings extraction...")

    # Test data
    test_input = Path("../examples/data/example.fasta")
    if not test_input.exists():
        print(f"❌ Test input file not found: {test_input}")
        return False

    # Create a simple MCP client test
    test_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "extract_rna_embeddings",
            "arguments": {
                "input_file": str(test_input),
                "output_file": "/tmp/test_embeddings_mcp",
                "use_mock": True
            }
        }
    }

    try:
        # Start server and send request
        process = subprocess.Popen(
            [sys.executable, "src/server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Send request
        stdout, stderr = process.communicate(json.dumps(test_request) + "\n", timeout=30)

        if stderr:
            print(f"Server error: {stderr}")

        # Parse response
        lines = stdout.strip().split('\n')
        for line in lines:
            if line.strip():
                try:
                    response = json.loads(line)
                    if "result" in response:
                        result = response["result"]
                        if result.get("status") == "success":
                            print("✅ Synchronous tool test passed")
                            print(f"   Processed {result.get('num_sequences', 'unknown')} sequences")
                            return True
                        else:
                            print(f"❌ Tool failed: {result.get('error', 'Unknown error')}")
                            return False
                except json.JSONDecodeError:
                    continue

        print("❌ No valid response received")
        return False

    except subprocess.TimeoutExpired:
        process.kill()
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_job_management():
    """Test job management tools"""
    print("\nTesting job management...")

    # Test list_jobs
    test_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "list_jobs",
            "arguments": {}
        }
    }

    try:
        process = subprocess.Popen(
            [sys.executable, "src/server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(json.dumps(test_request) + "\n", timeout=10)

        lines = stdout.strip().split('\n')
        for line in lines:
            if line.strip():
                try:
                    response = json.loads(line)
                    if "result" in response:
                        result = response["result"]
                        if result.get("status") == "success":
                            print("✅ Job management test passed")
                            print(f"   Found {result.get('total', 0)} jobs")
                            return True
                        else:
                            print(f"❌ Job management failed: {result.get('error', 'Unknown error')}")
                            return False
                except json.JSONDecodeError:
                    continue

        return False

    except Exception as e:
        print(f"❌ Job management test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing RNA-FM MCP Server")
    print("=" * 50)

    tests_passed = 0
    total_tests = 2

    if test_sync_tool():
        tests_passed += 1

    if test_job_management():
        tests_passed += 1

    print("\n" + "=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)