"""Visualize the LangGraph as a Mermaid PNG."""

from pathlib import Path

from ai.graph import graph_manager

OUTPUT = Path("graph.png")

png_bytes = graph_manager._graph.get_graph().draw_mermaid_png()
OUTPUT.write_bytes(png_bytes)
print(f"Saved to {OUTPUT.resolve()}")
