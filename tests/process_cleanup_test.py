import os
import time
import subprocess
import ctypes
from ctypes import wintypes

def test_job_object():
    if os.name != 'nt':
        print("Test only works on Windows.")
        return

    print("--- Windows Job Object Cleanup Test ---")
    
    # 1. Create Job
    job = ctypes.windll.kernel32.CreateJobObjectW(None, None)
    if not job:
        print(f"FAILED: CreateJobObjectW failed with error {ctypes.GetLastError()}")
        return
    print(f"SUCCESS: Job Object created (handle: {job})")

    # 2. Set Limit Information (Kill on close)
    # JOBOBJECT_EXTENDED_LIMIT_INFORMATION = 9
    # На 64-битных системах размер структуры 144 байта
    # LimitFlags находится по смещению 16 (4-й DWORD в массиве DWORD)
    limit_info = (wintypes.DWORD * 36)() # 36 * 4 = 144 bytes
    limit_info[4] = 0x2000 # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    
    res = ctypes.windll.kernel32.SetInformationJobObject(
        job, 
        9, 
        ctypes.byref(limit_info), 
        ctypes.sizeof(limit_info)
    )
    if not res:
        print(f"FAILED: SetInformationJobObject failed with error {ctypes.GetLastError()}")
        return
    print("SUCCESS: Limit Information set (KillOnJobClose)")

    # 3. Start a dummy process (ping is good for waiting)
    # Using CREATE_NO_WINDOW = 0x08000000
    cmd = ["ping", "127.0.0.1", "-n", "100"]
    proc = subprocess.Popen(cmd, creationflags=0x08000000)
    print(f"SUCCESS: Started dummy process (PID: {proc.pid})")

    # 4. Assign to Job
    res = ctypes.windll.kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle)))
    if not res:
        err = ctypes.GetLastError()
        # ERROR_ACCESS_DENIED = 5 (often happens if process is already in a job)
        print(f"FAILED: AssignProcessToJobObject failed with error {err}")
        if err == 5:
            print("HINT: Access Denied. The process might already be in another Job Object (typical for some IDEs).")
        return
    print("SUCCESS: Process assigned to Job Object")

    # 5. Check if process is running
    if proc.poll() is None:
        print("Process is running. Now closing Job Handle to see if it kills the child...")
    else:
        print("Process already finished. Test invalid.")
        return

    # 6. Close Job Handle (this should trigger the kill)
    ctypes.windll.kernel32.CloseHandle(job)
    print("Job Handle closed.")

    # 7. Check if process was killed
    time.sleep(1)
    if proc.poll() is not None:
        print("SUCCESS: Process was KILLED automatically!")
    else:
        print("FAILED: Process is STILL RUNNING!")
        proc.terminate()

if __name__ == "__main__":
    test_job_object()
