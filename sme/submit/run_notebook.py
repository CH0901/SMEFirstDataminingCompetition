"""노트북을 실행해 출력을 파일에 박아 넣는다 (규칙 4.2 요구사항)."""
import sys, nbformat as nbf
from nbclient import NotebookClient
p = sys.argv[1]
nb = nbf.read(p, as_version=4)
NotebookClient(nb, timeout=1800, kernel_name="sme-venv",
               resources={"metadata": {"path": "."}}).execute()
nbf.write(nb, p)
n = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs"))
print(f"{p}: 코드셀 출력 {n}개 삽입")
