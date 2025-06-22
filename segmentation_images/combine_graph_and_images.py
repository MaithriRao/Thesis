import os
import subprocess

p = subprocess.run(["identify", "-ping", "-format", "%w %h", "video.png"], capture_output=True, encoding="utf-8", check=True)

graph_width, graph_height = p.stdout.split(" ")
graph_width, graph_height = int(graph_width), int(graph_height)

subprocess.run(["inkscape", "-w", str(graph_width), "-h", str(graph_height), "graph.svg", "-o", "graph.png"], check=True)
subprocess.run(["magick", "graph.png", "video.png", "-append", "graphic.png"], check=True)

