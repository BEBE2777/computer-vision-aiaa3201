"""Wait for the 7B GPU smoke report on the course box, pull it locally,
and print a distribution analysis (no label/mask ever sent to the model).
Run with -u and PYTHONIOENCODING=utf-8.
"""
import os, time, json, sys
from collections import Counter
import paramiko

HOST, PORT, USER = "121.12.5.46", 10084, "qzheng610"
PASS = os.environ["GPU_PASS"]
REMOTE = "/data/disk2/home/qzheng610/qwen_work/smoke_out/gpu_smoke_7b_shots_1.json"
LOCAL = r"C:/Users/qzheng/Desktop/computer_vision/final_projects/aiaa3201_repo/results/extensions/qwen/smoke/gpu_smoke_7b_report.json"
AUTOLOG = "/data/disk2/home/qzheng610/qwen_work/auto_smoke_7b.log"


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=20,
              allow_agent=False, look_for_keys=False)
    return c


def exists(c, path):
    i, o, e = c.exec_command(f"test -s {path} && echo YES || echo NO")
    return o.read().decode().strip() == "YES"


def main():
    deadline = time.time() + 90 * 60  # 90 min max
    last_log = ""
    print("WAITING for 7B smoke report ...", flush=True)
    while time.time() < deadline:
        try:
            c = connect()
            if exists(c, REMOTE):
                i, o, e = c.exec_command(f"cat {REMOTE}")
                data = o.read()
                open(LOCAL, "wb").write(data)
                print(f"PULLED {len(data)} bytes -> {LOCAL}", flush=True)
                c.close()
                break
            i, o, e = c.exec_command(f"tail -6 {AUTOLOG} 2>/dev/null; echo '---SHARDS---'; du -sm /data/disk2/home/qzheng610/qwen_work/qwen2vl7b_local 2>/dev/null | cut -f1")
            last_log = o.read().decode(errors="replace")
            print(f"[{time.strftime('%H:%M:%S')}] not ready\n{last_log}", flush=True)
            c.close()
        except Exception as ex:
            print(f"conn error {type(ex).__name__}: {str(ex)[:120]}", flush=True)
        time.sleep(60)
    else:
        print("TIMEOUT waiting for 7B report", flush=True)
        sys.exit(2)

    d = json.load(open(LOCAL, encoding="utf-8"))
    recs = d["records"]
    print("\n===== 7B SMOKE ANALYSIS =====", flush=True)
    print("model_id:", d["model_id"], "prompt_sha256:", d["prompt_sha256"], flush=True)
    print("n_records:", d["n_records"], "json_valid_rate:", d["json_valid_rate"],
          "mean_lat_ms:", round(d["mean_latency_ms"]),
          "mean_peak_gb:", round(d["mean_peak_gpu_gb"], 2), flush=True)
    scores = [r["anomaly_score"] for r in recs]
    print("unique scores:", sorted(set(scores)), flush=True)
    print("decisions:", dict(Counter(r["decision"] for r in recs)), flush=True)
    good_scores = [r["anomaly_score"] for r in recs if r["true_label_for_eval_ONLY"] == "good"]
    bad_scores = [r["anomaly_score"] for r in recs if r["true_label_for_eval_ONLY"] == "bad"]
    print("good scores:", good_scores, flush=True)
    print("bad  scores:", bad_scores, flush=True)
    try:
        from sklearn.metrics import roc_auc_score
        y = [1 if r["true_label_for_eval_ONLY"] == "bad" else 0 for r in recs]
        if len(set(scores)) > 1:
            print("AUROC:", round(roc_auc_score(y, scores), 4), flush=True)
        else:
            print("AUROC: undefined (all scores identical)", flush=True)
    except Exception as ex:
        print("AUROC skip:", ex, flush=True)
    print("\n--- sample raw responses (good / bad) ---", flush=True)
    for lbl in ("good", "bad"):
        r = next((x for x in recs if x["true_label_for_eval_ONLY"] == lbl), None)
        if r:
            print(f"[{lbl}] {r['category']} {r['query_id']} score={r['anomaly_score']} dec={r['decision']}", flush=True)
            print(r["raw_response"][:500].replace("\n", " "), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
