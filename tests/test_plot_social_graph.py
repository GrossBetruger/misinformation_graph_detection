import os

import pytest
from pathlib import Path

import networkx as nx

from misinformation_graph_detection.analyze import (
    plot_social_graph,
    build_social_graph,
    PersonNode,
)


def create_sample_graph():
    # Two nodes connected by one edge, with defined attributes
    nodes = [
        PersonNode(id="314159", time=100, friends=10, followers=5),
        PersonNode(id="271828", time=200, friends=20, followers=10),
    ]
    edges = [("314159", "271828")]
    return build_social_graph(nodes, edges)


def test_plot_social_graph_creates_html(tmp_path, monkeypatch):
    # Prepare working directory
    monkeypatch.chdir(tmp_path)

    # Capture os.system calls to avoid opening a browser
    calls = []
    monkeypatch.setattr(
        "misinformation_graph_detection.analyze.os.system",
        lambda cmd: calls.append(cmd),
    )

    # Generate the graph and plot it
    G = create_sample_graph()
    title = "Test Graph"
    plot_social_graph(G, title)

    # Expected output file
    filename = tmp_path / "test_graph.html"
    # Check file was created
    assert filename.exists(), f"Missing output HTML file: {filename}"

    # Check that os.system was called with the open command
    assert calls == [
        f"open {filename.name}"
    ], "Expected os.system to open the HTML file"

    # Read content and perform basic sanity checks
    content = filename.read_text(encoding="utf-8")
    # Should contain the vis DataSet initialization for nodes and edges
    assert "new vis.DataSet" in content, "HTML missing vis DataSet"
    # Node labels and values should appear in JSON
    assert '"label": "314159"' in content, "Node label missing in HTML"
    assert '"value": 5' in content, "Node value missing in HTML"
    assert '"label": "271828"' in content, "Node label missing in HTML"
    assert '"value": 10' in content, "Node value missing in HTML"
    # Edge definition should appear in JSON
    assert '"from": "314159"' in content, "Edge source missing in HTML"
    assert '"to": "271828"' in content, "Edge target missing in HTML"


def test_plot_social_graph_empty_graph(tmp_path, monkeypatch):
    # Graph with no nodes should still create an HTML file (empty canvas)
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(
        "misinformation_graph_detection.analyze.os.system",
        lambda cmd: calls.append(cmd),
    )
    # Empty graph
    G = nx.Graph()
    title = "Empty Graph"
    # Should not raise
    plot_social_graph(G, title)
    filename = tmp_path / "empty_graph.html"
    assert filename.exists(), "Empty graph HTML file not created"
    content = filename.read_text(encoding="utf-8")
    # Even empty graph should include DataSet initializations
    assert "new vis.DataSet" in content
    # No nodes or edges data beyond the default
    # os.system should have been called once
    assert calls == [f"open {filename.name}"]
