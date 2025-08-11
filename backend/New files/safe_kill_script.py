#!/usr/bin/env python3
"""
🔴 EMERGENCY SCRIPT KILLER FOR SSH GPU SERVERS
Use this to safely terminate training scripts without creating zombie processes
or leaving GPU memory allocated.

Usage in SSH:
1. Find your process: ps aux | grep python
2. Run this script: python safe_kill_script.py [PID]
3. Or use: python safe_kill_script.py --kill-all-python

This prevents SSH access revocation due to zombie processes or GPU monopolization.
"""

import os
import sys
import signal
import psutil
import subprocess
import time
import argparse

def cleanup_gpu_memory():
    """Clean up CUDA GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            print("🧼 Cleaning GPU memory...")
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("✅ GPU memory cleaned")
    except ImportError:
        print("PyTorch not available, skipping GPU cleanup")
    except Exception as e:
        print(f"GPU cleanup error: {e}")

def kill_process_tree(pid):
    """Kill a process and all its children safely."""
    try:
        parent = psutil.Process(pid)
        print(f"🎯 Target process: {parent.name()} (PID: {pid})")
        
        # Get all children first
        children = parent.children(recursive=True)
        print(f"📦 Found {len(children)} child processes")
        
        # Kill children first
        for child in children:
            try:
                print(f"  🔸 Killing child PID {child.pid}")
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Wait for children to terminate
        _, still_alive = psutil.wait_procs(children, timeout=5)
        
        # Force kill any stubborn children
        for child in still_alive:
            try:
                print(f"  🔥 Force killing stubborn child PID {child.pid}")
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Kill parent
        try:
            print(f"🔸 Killing parent PID {pid}")
            parent.terminate()
            parent.wait(timeout=5)
        except psutil.TimeoutExpired:
            print(f"🔥 Force killing parent PID {pid}")
            parent.kill()
            
        print(f"✅ Successfully killed process tree for PID {pid}")
        return True
        
    except psutil.NoSuchProcess:
        print(f"❌ Process PID {pid} not found")
        return False
    except Exception as e:
        print(f"❌ Error killing process {pid}: {e}")
        return False

def find_python_processes():
    """Find all Python processes that might be training scripts."""
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if any(keyword in cmdline.lower() for keyword in ['train', 'model', 'deepfake', 'multimodal']):
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return python_processes

def main():
    parser = argparse.ArgumentParser(description='Safely kill Python training scripts')
    parser.add_argument('pid', nargs='?', type=int, help='Process ID to kill')
    parser.add_argument('--kill-all-python', action='store_true', 
                       help='Kill all Python training processes')
    parser.add_argument('--list', action='store_true', 
                       help='List all Python training processes')
    
    args = parser.parse_args()
    
    if args.list:
        print("🔍 Searching for Python training processes...")
        processes = find_python_processes()
        if processes:
            print(f"Found {len(processes)} potential training processes:")
            for proc in processes:
                print(f"  PID {proc['pid']}: {proc['name']} - {proc['cmdline']}")
        else:
            print("No Python training processes found")
        return
    
    if args.kill_all_python:
        print("🚨 KILLING ALL PYTHON TRAINING PROCESSES...")
        processes = find_python_processes()
        if not processes:
            print("No Python training processes found")
            return
            
        for proc in processes:
            print(f"\n🎯 Targeting PID {proc['pid']}: {proc['cmdline']}")
            kill_process_tree(proc['pid'])
        
        # Clean GPU memory after killing all processes
        cleanup_gpu_memory()
        print("\n✅ All Python training processes terminated safely")
        return
    
    if args.pid:
        print(f"🚨 KILLING PROCESS PID {args.pid}...")
        success = kill_process_tree(args.pid)
        if success:
            cleanup_gpu_memory()
            print("\n✅ Process terminated safely")
        else:
            print("\n❌ Failed to terminate process")
        return
    
    # Interactive mode
    print("🔍 Searching for Python training processes...")
    processes = find_python_processes()
    
    if not processes:
        print("No Python training processes found")
        return
    
    print(f"Found {len(processes)} potential training processes:")
    for i, proc in enumerate(processes):
        print(f"  {i+1}. PID {proc['pid']}: {proc['name']} - {proc['cmdline']}")
    
    try:
        choice = input("\nEnter number to kill (or 'all' to kill all): ").strip()
        if choice.lower() == 'all':
            for proc in processes:
                kill_process_tree(proc['pid'])
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(processes):
                kill_process_tree(processes[idx]['pid'])
            else:
                print("Invalid choice")
                return
        
        cleanup_gpu_memory()
        print("\n✅ Process(es) terminated safely")
        
    except KeyboardInterrupt:
        print("\n🛑 Aborted")
    except ValueError:
        print("Invalid input")

if __name__ == "__main__":
    main()

