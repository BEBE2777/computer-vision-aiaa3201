"""Upload any local file to any remote path via SSH exec + base64 (SFTP is
unavailable on this box). Usage: GPU_PASS=xxx python put_any.py <local> <remote>
"""
import os, sys, base64, paramiko

HOST, PORT, USER = "121.12.5.46", 10084, "qzheng610"
local, remote = sys.argv[1], sys.argv[2]
password = os.environ.get("GPU_PASS")
if not password:
    raise SystemExit("GPU_PASS env var not set")

content = open(local, "rb").read()
b64 = base64.b64encode(content).decode()
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=password,
               timeout=20, allow_agent=False, look_for_keys=False)
cmd = (f"mkdir -p $(dirname {remote}); echo {b64} | base64 -d > {remote}; "
       f"echo WROTE {remote} $(wc -c < {remote}) bytes")
stdin, stdout, stderr = client.exec_command(cmd, timeout=60)
print(stdout.read().decode("utf-8", "replace"), end="")
err = stderr.read().decode("utf-8", "replace")
if err.strip():
    print("[stderr]", err, file=sys.stderr)
client.close()
