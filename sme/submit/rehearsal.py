"""채점 리허설: 노트북 3셀을 실제 API로 그대로 실행한다."""
import json, sys
nb = json.load(open("submission_notebook_filled.ipynb"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
src = src.replace("/kaggle/input", "/tmp/kgl/input").replace("/kaggle/working", "/tmp/kgl/working")
src = src.replace('API_KEY    = ""', f'API_KEY    = "{sys.argv[1]}"')
src = src.replace('PRED_START = "20260624"', f'PRED_START = "{sys.argv[2]}"')
src = src.replace('PRED_END   = "20260630"', f'PRED_END   = "{sys.argv[3]}"')
try:
    exec(compile(src, "notebook", "exec"), {"__name__": "__main__"})
except SystemExit as e:
    print("SystemExit:", e)
