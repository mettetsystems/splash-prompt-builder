"""Small reproducible multilingual quality audit; retain outputs for human review."""
import json
import os
from pathlib import Path
import sys
import threading
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.models import DiffusionAdapter
from backend.agents import POLICIES, RULES

def main():
    os.environ.setdefault("HF_MODULES_CACHE", str(Path(".models/modules").resolve()))
    adapter = DiffusionAdapter({"model":"dream", "path":str(Path('.models/dream').resolve()), "device":"cuda:1", "steps":64})
    cases = [
        ("English", "splash-code", "Build a Python CLI that reads a UTF-8 file. Do not use external dependencies. Include tests.", ["Python", "UTF-8", "test"]),
        ("Spanish", "splash-code", "Crea una herramienta Python que lea un archivo UTF-8. No uses dependencias externas. Incluye pruebas.", ["Python", "UTF-8", "test"]),
        ("French", "splash-code", "Créer un outil Python qui lit un fichier UTF-8. Ne pas utiliser de dépendances externes. Inclure des tests.", ["Python", "UTF-8", "test"]),
        ("German", "splash-code", "Erstelle ein Python-Werkzeug, das eine UTF-8-Datei liest. Keine externen Abhängigkeiten verwenden. Tests einschließen.", ["Python", "UTF-8", "test"]),
        ("Chinese", "splash-code", "编写一个Python命令行工具，读取UTF-8文件。不要使用外部依赖。包含测试。", ["Python", "UTF-8", "test"]),
        ("Arabic", "splash-code", "أنشئ أداة Python لقراءة ملف UTF-8. لا تستخدم مكتبات خارجية. أضف اختبارات.", ["Python", "UTF-8", "test"]),
        ("English", "splash", "Help me explain our delayed launch to customers clearly. Do not promise a new release date.", ["not"]),
        ("English", "splash-search", "Study bilingual education and reading outcomes using peer-reviewed research from 2015 onward. Preserve informed consent.", ["2015", "consent"]),
    ]
    results=[]
    for language, agent, prompt, required in cases:
        output=adapter.generate([{"role":"system","content":POLICIES[agent]+" "+RULES},{"role":"user","content":"Prompt section:\n"+prompt}],threading.Event(),256)
        record={"language":language,"agent":agent,"prompt":prompt,**output,"literal_checks":{term:term.casefold() in output['text'].casefold() for term in required}}
        results.append(record)
        print(json.dumps(record,ensure_ascii=False),flush=True)
        Path('artifacts/enrichment-evaluation.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
