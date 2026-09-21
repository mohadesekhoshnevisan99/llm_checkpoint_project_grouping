"""Build a source-only ZIP without local dependencies, environments or outputs."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parent
files = ['optimizer.py', 'policy_bridge.py', 'summarize.py', 'test_optimizer.py',
         'test_policy_bridge.py', 'requirements.txt', 'scenario.example.json',
         'START_HERE_FA.md', 'GUIDE_FA.md', 'RESULTS.md', 'make_download.py',
         '.gitignore', '.vscode/launch.json']
target = root.parent / 'grouped_checkpoint_optimizer_download.zip'
with ZipFile(target, 'w', compression=ZIP_DEFLATED) as package:
    for name in files:
        package.write(root / name, arcname=f'grouped_checkpoint_optimizer/{name}')
with ZipFile(target) as package:
    assert package.testzip() is None
    assert len(package.namelist()) == len(files)
print(f'{target}\n{len(files)} files; {target.stat().st_size} bytes; ZIP integrity verified')
