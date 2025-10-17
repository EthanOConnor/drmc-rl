
#@category DrMarioRE
"""Dump functions, symbols, and selected xrefs to JSON.
Usage: -postScript dump_symbols.py <out_json> [program_name]
"""

import json
from ghidra.app.script import GhidraScript

class DumpSymbols(GhidraScript):
    def run(self):
        args = self.getScriptArgs()
        out_path = args[0] if len(args) > 0 else "symbols.json"
        target_name = args[1] if len(args) > 1 else None

        program = getattr(self, 'currentProgram', None)
        if program is None:
            try:
                program = self.getCurrentProgram()
            except Exception:
                program = None

        if program is None:
            try:
                project = self.getCurrentProject()
                if project:
                    project_data = project.getProjectData()
                    root = project_data.getRootFolder()
                    domain_file = None
                    if target_name:
                        domain_file = root.getFile(target_name)
                    if domain_file is None:
                        files = root.getFiles()
                        if files:
                            domain_file = files[0]
                    if domain_file:
                        program = domain_file.getDomainObject(self, False, False, self.monitor)
            except Exception:
                program = None

        if program is None:
            print("No program available; exiting")
            return

        fm = program.getFunctionManager()
        sm = program.getSymbolTable()
        rm = program.getReferenceManager()

        functions = []
        fiter = fm.getFunctions(True)
        while fiter.hasNext() and not self.monitor.isCancelled():
            f = fiter.next()
            functions.append({"name": f.getName(), "entry": str(f.getEntryPoint()), "body": str(f.getBody())})

        symbols = []
        sit = sm.getAllSymbols(True)
        while sit.hasNext() and not self.monitor.isCancelled():
            s = sit.next()
            symbols.append({"name": s.getName(), "addr": str(s.getAddress()), "type": str(s.getSymbolType())})

        xrefs = {}
        for addr_val in (0x4016, 0x4017):
            try:
                to = self.toAddr(addr_val)
                refs = rm.getReferencesTo(to)
                info = []
                for r in refs:
                    info.append({"from": str(r.getFromAddress()), "opIndex": r.getOperandIndex()})
                xrefs[hex(addr_val)] = info
            except Exception:
                xrefs[hex(addr_val)] = []

        with open(out_path, 'w') as f:
            json.dump({"functions": functions, "symbols": symbols, "xrefs": xrefs}, f, indent=2)
        print("Wrote {}".format(out_path))

if __name__ == '__main__':
    DumpSymbols().run()
