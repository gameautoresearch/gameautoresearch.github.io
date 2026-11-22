#!/usr/bin/env python3
"""Pinned stock bitcode-linker adapter: omit unwanted generated debug sections.

The pinned Rust driver adds --debug even for release. It is metadata only; all
executable instructions/data/public exports and the requested LTO remain intact.
"""
import json,os,sys
from pathlib import Path
linker=Path(os.environ['TAVERN_NATIVE_REAL_LINKER'])
if not linker.is_file():raise FileNotFoundError(linker)
args=[arg for arg in sys.argv[1:] if arg!='--debug']
if path:=os.environ.get('TAVERN_NATIVE_LINK_LOG'):
    with open(path,'a') as out:out.write(json.dumps({'linker':str(linker),'args':args,'removed_debug_flag':'--debug' in sys.argv})+'\n')
os.execv(str(linker),[str(linker),*args])
