# scripts/gen_arch_diagrams.py

import os
import json
import sys
from pathlib import Path


def load_spec(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".json":
        return json.loads(Path(path).read_text())
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError(
                "YAML input requested but PyYAML is not installed."
            ) from e
        return yaml.safe_load(Path(path).read_text())
    return json.loads(Path(path).read_text())


def mermaid_component_flowchart(spec: dict) -> str:
    comps = {c["id"]: c for c in spec.get("components", [])}
    lines = ["flowchart LR"]
    for cid, c in comps.items():
        label = f"{c.get('name', cid)}\n({c.get('layer','')})"
        lines.append(f'    {cid}["{label}"]')
    for cid, c in comps.items():
        for dep in c.get("depends_on", []):
            if dep in comps:
                lines.append(f"    {cid} --> {dep}")
    return "\n".join(lines)


def mermaid_sequence_diagram(spec: dict, flow_id: str) -> str:
    flows = {f["id"]: f for f in spec.get("flows", [])}
    if flow_id not in flows:
        raise KeyError(f"flow '{flow_id}' not found")
    flow = flows[flow_id]
    steps = flow.get("steps", [])
    participants = []
    for s in steps:
        for key in ("from", "to"):
            cid = s.get(key)
            if cid and cid not in participants:
                participants.append(cid)
    lines = ["sequenceDiagram", "  autonumber"]
    comps = {c["id"]: c for c in spec.get("components", [])}
    for pid in participants:
        label = comps.get(pid, {}).get("name", pid)
        lines.append(f"  participant {pid} as {label}")
    for s in steps:
        frm, to, msg = s["from"], s["to"], s.get("message", "")
        lines.append(f"  {frm}->>{to}: {msg}")
    return "\n".join(lines)


def write_overview(spec: dict, out_dir: Path):
    title = spec.get("meta", {}).get("title", "Architecture")
    rows = []
    for c in spec.get("components", []):
        owns = ", ".join(c.get("owns", []))
        provides = ", ".join(c.get("provides", []))
        deps = ", ".join(c.get("depends_on", []))
        rows.append(
            f"| `{c['id']}` | {c.get('name','')} | {c.get('layer','')} | {owns} | {provides} | {deps} |"
        )
    flowchart = mermaid_component_flowchart(spec)
    md = f"""# {title} — Overview

## Components
| id | name | layer | owns | provides | depends_on |
|---|---|---|---|---|---|
{os.linesep.join(rows)}

## Dependencies (Mermaid)

```mermaid
{flowchart}
```
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "arch_overview.md").write_text(md)


def write_flows(spec: dict, out_dir: Path):
    flows_dir = out_dir / "flows"
    flows_dir.mkdir(parents=True, exist_ok=True)
    for f in spec.get("flows", []):
        seq = mermaid_sequence_diagram(spec, f["id"])
        md = f"""# Flow: {f.get('title', f['id'])}

```mermaid
{seq}
```
"""
        (flows_dir / f"{f['id']}.md").write_text(md)


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python gen_arch_diagrams.py <spec.(json|yaml)> <out_dir>",
            file=sys.stderr,
        )
        sys.exit(2)
    spec_path, out_dir = sys.argv[1], Path(sys.argv[2])
    spec = load_spec(spec_path)
    write_overview(spec, out_dir)
    write_flows(spec, out_dir)
    print(f"Wrote diagrams to: {out_dir}")


if __name__ == "__main__":
    main()
