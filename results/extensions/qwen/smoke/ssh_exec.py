"""Run a multi-line shell script on GPU box by piping via stdin (no SFTP needed).
Password from env GPU_PASS. Usage: GPU_PASS=xxx python ssh_exec.py <local_script>
The script content is base64-encoded and decoded remotely to a temp file, then run.
"""
import os, sys, base64, paramiko

HOST, PORT, USER = "121.12.5.46", 10084, "qzheng610"
script_path = sys.argv[1]
password = os.environ.get("GPU_PASS")
if not password:
    raise SystemExit("GPU_PASS env var not set")

content = open(script_path, "rb").read()
b64 = base64.b64encode(content).decode()
remote = "/data/disk2/home/qzheng610/qwen_work/" + os.path.basename(script_path)

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=password,
               timeout=20, allow_agent=False, look_for_keys=False)
cmd = f"mkdir -p $(dirname {remote}); echo {b64} | base64 -d > {remote}; chmod +x {remote}; echo WROTE {remote} $(wc -c < {remote}) bytes"
stdin, stdout, stderr = client.exec_command(cmd, timeout=60)
print(stdout.read().decode("utf-8", "replace"), end="")
err = stderr.read().decode("utf-8", "replace")
if err.strip():
    print("[stderr]", err, file=sys.stderr)
client.close()
