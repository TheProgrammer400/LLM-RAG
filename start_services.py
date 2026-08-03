#!/usr/bin/env python3
"""
start_services.py
=================
Single-command Python launcher for all 5 CDSS microservices.
Boots NLP Service, Retrieval Service, Reasoning Service, Inference Gateway, and Orchestrator.
Manages clean process shutdown on Ctrl+C.
"""

import sys
import os
import signal
import subprocess
import time

SERVICES = [
    {"name": "NLP Service", "port": 8001, "cmd": [sys.executable, "-m", "uvicorn", "services.nlp_service.main:app", "--port", "8001", "--reload"]},
    {"name": "Retrieval Service", "port": 8002, "cmd": [sys.executable, "-m", "uvicorn", "services.retrieval_service.main:app", "--port", "8002", "--reload"]},
    {"name": "Reasoning Service", "port": 8003, "cmd": [sys.executable, "-m", "uvicorn", "services.reasoning_service.main:app", "--port", "8003", "--reload"]},
    {"name": "Inference Gateway", "port": 8004, "cmd": [sys.executable, "-m", "uvicorn", "services.inference_gateway.main:app", "--port", "8004", "--reload"]},
    {"name": "Orchestrator Gateway", "port": 8000, "cmd": [sys.executable, "-m", "uvicorn", "services.orchestrator.main:app", "--port", "8000", "--reload"]},
]

processes = []

def signal_handler(sig, frame):
    print("\n\n[CDSS Launcher] Shutting down all 5 microservices...")
    for proc in processes:
        try:
            proc.terminate()
        except Exception:
            pass
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 60)
    print("🚀 CDSS Microservices Single-Command Launcher")
    print("=" * 60)

    for s in SERVICES:
        print(f"  • Starting {s['name']} on http://localhost:{s['port']}...")
        p = subprocess.Popen(s["cmd"])
        processes.append(p)
        time.sleep(0.8)

    print("\n✅ All 5 CDSS Services Online!")
    print("👉 Main API Gateway: http://localhost:8000/")
    print("👉 Interactive Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C anytime to stop all services.\n")

    for proc in processes:
        proc.wait()

if __name__ == "__main__":
    main()
