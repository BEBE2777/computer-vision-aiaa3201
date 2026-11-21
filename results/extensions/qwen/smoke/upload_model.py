"""Stream-upload the local Qwen2-VL-2B model dir to the GPU box via tar over SSH stdin.
Password from env GPU_PASS. Resumes not supported; run once.
"""
import os, io, tarfile, paramiko, time

LOCAL = r"C:/Users/qzheng/.cache/huggingface/hub/qwen2vl2b_local"
REMOTE_PARENT = "/data/disk2/home/qzheng610/qwen_work"
HOST, PORT, USER = "121.12.5.46", 10084, "qzheng610"

t0 = time.time()
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tf:
    tf.add(LOCAL, arcname="qwen2vl2b_local")
print(f"packed {buf.tell()/1e6:.0f} MB in {time.time()-t0:.0f}s; uploading...", flush=True)
buf.seek(0)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=os.environ["GPU_PASS"],
          timeout=30, allow_agent=False, look_for_keys=False)
cmd = f"mkdir -p {REMOTE_PARENT} && tar -xz -C {REMOTE_PARENT}"
stdin, stdout, stderr = c.exec_command(cmd, timeout=600)
data = buf.getvalue()
# write in chunks to avoid blocking
for i in range(0, len(data), 1 << 20):
    stdin.write(data[i:i + (1 << 20)])
stdin.channel.shutdown_write()
rc = stdout.channel.recv_exit_status()
err = stderr.read().decode("utf-8", "replace")
print("upload rc", rc, "elapsed", round(time.time()-t0, 1), "s")
if err.strip(): print("stderr:", err[:500])
stdin2, stdout2, _ = c.exec_command(f"du -sh {REMOTE_PARENT}/qwen2vl2b_local")
print(stdout2.read().decode().strip())
c.close()
