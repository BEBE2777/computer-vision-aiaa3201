"""Run a command on the course GPU box via password SSH.

Password is read from env GPU_PASS (never written to disk or repo).
Usage: GPU_PASS=xxx python ssh_run.py "remote shell command"
"""
import os, sys, paramiko

HOST = "121.12.5.46"
PORT = 10084
USER = "qzheng610"

cmd = sys.argv[1] if len(sys.argv) > 1 else "echo ok"
password = os.environ.get("GPU_PASS")
if not password:
    raise SystemExit("GPU_PASS env var not set")

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=password,
               timeout=20, allow_agent=False, look_for_keys=False)
stdin, stdout, stderr = client.exec_command(cmd, timeout=120)
out = stdout.read().decode("utf-8", "replace")
err = stderr.read().decode("utf-8", "replace")
rc = stdout.channel.recv_exit_status()
print(out, end="")
if err.strip():
    print("[stderr]", err, file=sys.stderr)
print(f"[exit {rc}]")
client.close()
sys.exit(0)
