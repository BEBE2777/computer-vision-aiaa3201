"""Upload a local file to the course GPU box via SFTP. Password from env GPU_PASS.
Usage: GPU_PASS=xxx python ssh_put.py <local_path> <remote_path>
"""
import os, sys, paramiko

HOST, PORT, USER = "121.12.5.46", 10084, "qzheng610"
local, remote = sys.argv[1], sys.argv[2]
password = os.environ.get("GPU_PASS")
if not password:
    raise SystemExit("GPU_PASS env var not set")

t = paramiko.Transport((HOST, PORT))
t.connect(username=USER, password=password)
sftp = paramiko.SFTPClient.from_transport(t)
sftp.put(local, remote)
print(f"uploaded {local} -> {remote} ({os.path.getsize(local)} bytes)")
sftp.close()
t.close()
