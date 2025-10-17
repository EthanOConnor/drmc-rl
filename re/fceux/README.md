# FCEUX workflow (CDL + labels)

1. Open ROM, start **Code/Data Logger** (Tools → Code/Data Logger).  
2. Play multiple scenarios to mark as much code/data as possible, then **save the .cdl**.
3. Export **.nl** label files per bank (Tools → Debugger → Symbols). Keep naming per FCEUX spec.
4. Summarize the CDL coverage:
   ```bash
   python re/tools/cdl_to_json.py --cdl path/to/game.cdl --out re/out/cdl_summary.json
   ```
