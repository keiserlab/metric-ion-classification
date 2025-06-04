#init
from chimerax.core.toolshed import BundleAPI

class _MICAPI(BundleAPI):
    
    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
        from . import cmd
        if ci.name == "mic":
            func = cmd.mic
            desc = cmd.mic_desc
        elif ci.name == "mic extended":
            func = cmd.mic_extended
            desc = cmd.mic_extended_desc
        else:
            raise ValueError(f'Trying to register unknown command: {ci.name}')

        if desc.synopsis is None:
            desc.synopsis = ci.synopsis

        from chimerax.core.commands import register
        register(ci.name, desc, func)

    
    @staticmethod
    def start_tool(session, bi, ti):
        if ti.name == "MIC":
            from .tool import MICTool
            return MICTool(session, ti.name)
        raise ValueError(f"Trying to start unknown tool: {ti.name}")

    @staticmethod
    def get_class(class_name):
        if class_name == "MICTool":
            from . import tool
            return tool.MICTool
        raise ValueError(f"Unknown class name '{class_name}'")

bundle_api = _MICAPI()
