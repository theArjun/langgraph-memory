from pathlib import Path

import yaml
from jinja2 import Template

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str, **kwargs) -> dict[str, str] | str:
    path = _PROMPTS_DIR / f"{name}.yaml"
    data = yaml.safe_load(path.read_text())
    if "template" in data:
        return Template(data["template"]).render(**kwargs)
    return {key: Template(val).render(**kwargs) for key, val in data.items()}


__all__ = ["load_prompt"]
