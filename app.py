"""Standalone Shiny entrypoint. Each visitor receives a separate planner module."""
from pathlib import Path
import importlib.util
import sys
import uuid
from shiny import App
import planner

app_ui = planner.tab_multi_goal_ui("planner")

def new_planner():
    name = "_planner_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("planner.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return name, module

def server(input, output, session):
    name, module = new_planner()
    session.on_ended(lambda: sys.modules.pop(name, None))
    module.tab_multi_goal_server("planner", {}, {}, {})(input, output, session)

app = App(app_ui, server)
