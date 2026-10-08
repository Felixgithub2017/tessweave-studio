"""Read untrusted model metadata without importing model code or unpickling weights."""
import hashlib
import json
import math
import os
import struct
import re
from pathlib import Path
from .structure import describe

MAX_HEADER = 32 * 1024 * 1024
FLOAT_TYPES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}
DTYPES = dict(FLOAT_TYPES, I64=8, I32=4, I16=2, I8=1, U8=1, BOOL=1,
              F8_E4M3=1, F8_E5M2=1, F8_E4M3FN=1, U16=2, U32=4, U64=8)


def permitted(path, roots):
    p = Path(path).expanduser().resolve(strict=True)
    if not any(p == r or r in p.parents for r in map(lambda x: Path(x).resolve(), roots)):
        raise ValueError("路径不在 --allow-root 授权目录内（含符号链接目标）")
    return p


def read_json(path):
    if path.stat().st_size > MAX_HEADER:
        raise ValueError("JSON 元数据超过 32 MiB 限制")
    try:
        with path.open(encoding="utf-8") as f:
            obj = json.load(f)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid UTF-8 JSON metadata: %s (%s)" % (path, exc)) from exc
    if not isinstance(obj, dict):
        raise ValueError("元数据必须是 JSON 对象")
    return obj


def safetensors_header(path):
    size = path.stat().st_size
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError("safetensors 文件过短")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= MAX_HEADER or 8 + length > size:
            raise ValueError("safetensors header 长度非法")
        raw = f.read(length)
    try:
        header = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid safetensors JSON header: %s (%s)" % (path, exc)) from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header 非对象")
    count, floats, payload = 0, 0, 0
    spans, tensors = [], []
    for name, t in header.items():
        if name == "__metadata__":
            continue
        shape, offsets, dtype = t["shape"], t["data_offsets"], t["dtype"]
        if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
            raise ValueError("非法 tensor shape")
        n = math.prod(shape)
        if len(offsets) != 2 or any(type(v) is not int for v in offsets):
            raise ValueError("非法 tensor offset")
        lo, hi = offsets
        if not 0 <= lo <= hi <= size - 8 - length:
            raise ValueError("tensor offset 超出文件")
        if dtype in DTYPES and hi - lo != n * DTYPES[dtype]:
            raise ValueError("tensor 字节数与 shape/dtype 不符")
        if hi > lo:
            spans.append((lo, hi))
        count += n
        floats += n if dtype in FLOAT_TYPES else 0
        payload += hi - lo
        tensors.append({"name": name, "shape": shape, "dtype": dtype, "elements": n, "storage_bytes": hi-lo})
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("tensor 数据区重叠")
    return {"elements": count, "float_elements": floats, "payload_bytes": payload,
            "tensors": tensors, "header_sha256": hashlib.sha256(raw).hexdigest()}


def inspect_model(directory, roots, all_tensors=False):
    root = permitted(directory, roots)
    if not root.is_dir():
        raise ValueError("请选择模型目录，而不是单个文件")
    config_file = root / "config.json"
    config = read_json(permitted(config_file, roots)) if config_file.exists() else {}
    index_file = root / "model_index.json"
    pipeline = read_json(permitted(index_file, roots)) if index_file.exists() else {}
    if not config and not pipeline:
        raise ValueError("未找到 config.json 或 Diffusers model_index.json")
    warnings, errors, manifest, tensors = [], [], [], []
    # macOS AppleDouble sidecars preserve filesystem metadata on external disks.
    # Their suffix matches the real file but their bytes are not model content.
    files = sorted(f for f in root.glob("*.safetensors") if not f.name.startswith("._"))
    if pipeline:
        files = sorted(f for f in root.rglob("*.safetensors") if not f.name.startswith("._"))
        warnings.append("Diffusers 多组件包：仅盘点，不把文本训练/聊天服务配方套给生成管线")
    indexes = sorted(f for f in root.glob("*.safetensors.index.json") if not f.name.startswith("._"))
    expected = {}
    for idx in indexes:
        meta = read_json(permitted(idx, roots))
        for name, shard in meta.get("weight_map", {}).items():
            if not isinstance(shard, str) or Path(shard).name != shard:
                raise ValueError("分片索引含路径穿越或子路径")
            expected[name] = shard
            if not (root / shard).is_file():
                errors.append("缺少分片: " + shard)
    if len(files) > 10000:
        raise ValueError("分片数量超过限制")
    names, bytes_total, elements, floats = {}, 0, 0, 0
    for f in files:
        resolved = permitted(f, roots)
        h = safetensors_header(resolved)
        rel = str(f.relative_to(root))
        manifest.append({"file": rel, "bytes": resolved.stat().st_size,
                         "mtime_ns": resolved.stat().st_mtime_ns, "header_sha256": h["header_sha256"]})
        bytes_total += resolved.stat().st_size
        elements += h["elements"]
        floats += h["float_elements"]
        for t in h["tensors"]:
            key = rel + ":" + t["name"] if pipeline else t["name"]
            if key in names:
                errors.append("重复 tensor: " + key)
            names[key] = f.name
            tensors.append(dict(t, component=str(f.parent.relative_to(root))))
    for name, shard in expected.items():
        if names.get(name) != shard:
            errors.append("索引与 tensor 不匹配: " + name)
    quant = config.get("quantization_config") or {}
    packed = bool(quant) or any(x["dtype"] not in FLOAT_TYPES for x in tensors)
    if not files:
        errors.append("未发现 safetensors 权重；不反序列化 pickle/bin，不自动执行 remote code")
    if packed:
        warnings.append("含量化/非浮点张量：存储元素数不等于原参数量；训练资源估算需要原模型审计")
    if config.get("auto_map"):
        errors.append("模型要求自定义代码；默认不信任，需审计并编写适配器")
    mt = config.get("model_type", "")
    modality = "text"
    if pipeline:
        modality = "diffusion"
    elif config.get("vision_config") or any(x in mt for x in ("vl", "vision", "llava")):
        modality = "vision-language"
    if any(x in mt for x in ("audio", "speech", "whisper", "omni")):
        modality = "audio-multimodal"
    text = config.get("text_config") or config
    experts = text.get("num_experts", text.get("n_routed_experts", text.get("num_local_experts", 0)))
    evidence = {"config": config, "pipeline": pipeline, "manifest": manifest}
    fingerprint = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    card_title=None
    try:
        card=permitted(root/'README.md',roots)
        if card.is_file() and card.stat().st_size<=2*1024*1024:
            with card.open(encoding='utf-8',errors='replace') as stream:
                match=re.search(r'^# +([^\r\n]{1,160})\s*$',stream.read(65536),re.M)
                card_title=match.group(1).strip() if match else None
    except (OSError,ValueError):
        pass
    from .architecture_inventory import architecture_inventory
    inventory=architecture_inventory(config,tensors,errors,packed)
    if inventory['outside_decoder'].get('vision') and modality=='text':modality='vision-language'
    return {"path": str(root), "name": root.name, "model_type": mt,"model_card_title":card_title,
            "architecture_inventory":inventory,
            "architectures": config.get("architectures", []), "modality": modality,
            "pipeline": pipeline.get("_class_name"), "quantization": quant,
            "experts": experts, "config": text, "weight_bytes": bytes_total,
            "stored_elements": elements, "parameter_estimate": None if packed else floats,
            "count_basis": "header 浮点存储元素；共享权重/缓冲区使其不等于严格可训练参数数",
            "tensor_count": len(tensors), "tensors": tensors if all_tensors else tensors[:150], "manifest": manifest,
            "structure": describe(config, pipeline, tensors, packed),
            "fingerprint": fingerprint, "fingerprint_scope": "配置+header+大小+mtime，不是权重内容哈希",
            "warnings": warnings, "errors": sorted(set(errors))}


def browse_directories(directory, roots):
    """Read-only server-side folder picker; never uploads browser-local files."""
    root = permitted(directory, roots)
    if not root.is_dir():
        raise ValueError("请选择文件夹，不是文件")
    items, skipped = [], 0
    with os.scandir(root) as entries:
        for entry in entries:
            if len(items) >= 500:
                break
            try:
                if not entry.is_dir():
                    continue
                permitted(entry.path, roots)
                items.append({"name":entry.name,"path":entry.path})
            except (ValueError,OSError):
                skipped += 1
    try:
        parent = str(permitted(root.parent, roots)) if root.parent != root else None
    except ValueError:
        parent = None
    return {"path":str(root),"parent":parent,"directories":sorted(items,key=lambda x:x['name'].lower()),
            "limit_reached":len(items)>=500,"skipped":skipped}


def create_directory(parent, name, roots):
    base=permitted(parent,roots)
    if not isinstance(name,str) or not name.strip() or name in ('.','..') or '/' in name or '\\' in name or '\x00' in name:
        raise ValueError('Folder name must be a single non-empty path component')
    if not base.is_dir():raise ValueError('Parent must be a directory')
    target=base/name
    target.mkdir(mode=0o700,exist_ok=False)
    return {'path':str(target),'created':True}


def scan_directory(directory, roots, limit=100, max_directories=2000):
    root = permitted(directory, roots)
    if not root.is_dir():
        raise ValueError("请选择文件夹，不是文件")
    found, visited, unreadable = [], 0, []
    for current, dirs, files in os.walk(root, followlinks=False, onerror=lambda e:unreadable.append(str(e))):
        visited += 1
        if visited > max_directories:
            break
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__"))
        if "config.json" in files or "model_index.json" in files:
            try:
                found.append(inspect_model(current, roots))
            except (ValueError, OSError, KeyError, TypeError) as e:
                found.append({"path": current, "errors": [str(e)]})
            dirs[:] = []
        if len(found) >= limit:
            break
    return {"path":str(root),"models":found,"visited_directories":min(visited,max_directories),
            "limit_reached":len(found)>=limit or visited>max_directories,
            "limits":{"models":limit,"directories":max_directories},"unreadable":unreadable,
            "scope":"扫描所选目录及子目录；跳过隐藏子目录、符号链接目录、node_modules和__pycache__；找到模型根目录后不重复扫描内部组件。HF缓存请直接选择snapshots目录。"}


def discover(directory, roots, limit=100):
    return scan_directory(directory,roots,limit)['models']
