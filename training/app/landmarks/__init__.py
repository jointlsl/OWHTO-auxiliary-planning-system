import sys
from   pathlib import Path

##code import structure
_file_path      = Path(__file__) # <SRC_PROJECT>/FILE
_projsEntry_root_path = _file_path.resolve().parents[2] # <PROJECTs_ENTRY_POINT>/
print(_projsEntry_root_path)

if(_projsEntry_root_path not in sys.path):
    sys.path.insert(0,str(_projsEntry_root_path))