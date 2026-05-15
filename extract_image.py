import json, base64, sys, pathlib

src = "/root/.claude/projects/-home-user-EECS486-Ethnicity-Identifier/93bb37a0-3922-4ed2-b447-20a5a6892e3a.jsonl"
out_dir = pathlib.Path("/home/user/EECS486-Ethnicity-Identifier/data")
out_dir.mkdir(exist_ok=True)

found = 0
with open(src, "r") as f:
    for line in f:
        try:
            rec = json.loads(line)
        except Exception:
            continue

        def walk(o):
            global found
            if isinstance(o, dict):
                if o.get("type") == "image" and isinstance(o.get("source"), dict):
                    s = o["source"]
                    if s.get("type") == "base64":
                        data = base64.b64decode(s["data"])
                        ext = s.get("media_type", "image/jpeg").split("/")[-1]
                        if ext == "jpeg": ext = "jpg"
                        path = out_dir / f"nuclei_{found:02d}.{ext}"
                        path.write_bytes(data)
                        print("wrote", path, len(data), "bytes")
                        found += 1
                for v in o.values(): walk(v)
            elif isinstance(o, list):
                for v in o: walk(v)
        walk(rec)
print("total images:", found)
