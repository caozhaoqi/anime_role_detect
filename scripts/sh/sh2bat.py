#!/usr/bin/env python3
"""
SHBAT
ShellWindows

:
    python sh2bat.py input.sh [output.bat]
    python sh2bat.py -d /path/to/scripts/    # 
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple


# : sh -> windows
COMMAND_MAP = {
    # 
    r'\brsync\b': 'xcopy' if False else None,  # 
    r'\bcp\b': 'copy',
    r'\bmv\b': 'move',
    r'\brm\b': 'del',
    r'\brmdir\b': 'rmdir',
    r'\bmkdir\b': 'mkdir',
    r'\bln -s\b': 'mklink',
    r'\bcat\b': 'type',
    r'\btouch\b': None,  # Windows
    r'\bls\b': 'dir',
    r'\bpwd\b': 'cd',
    r'\bcd\b': 'cd /d',

    # 
    r'\bnohup\b': 'start',
    r'\bkillall\b': 'taskkill /F /IM',
    r'\bkill -9\b': 'taskkill /F /PID',
    r'\bps aux\b': 'tasklist',
    r'\bgrep\b': 'findstr',
    r'\bwhich\b': 'where',
    r'\bwhereis\b': 'where',
    r'\btop\b': 'tasklist',
    r'\bhtop\b': 'tasklist',

    # 
    r'\bsudo\b': None,  # Windows
    r'\bchmod\b': 'attrib',  # 
    r'\bchown\b': None,
    r'\bcurl\b': 'curl',  # Windows 10+curl
    r'\bwget\b': 'curl -O',
    r'\bhead\b': 'more',
    r'\btail\b': None,  # WindowsPowerShell
    r'\bwc -l\b': None,
    r'\bwatch\b': None,

    # Python/
    r'\bpython3\b': 'python',
    r'\bpip3\b': 'pip',
    r'\bpip install\b': 'pip install',
    r'\bnpm install\b': 'npm install',
    r'\bgit clone\b': 'git clone',

    # 
    r'\bexport\s+(\w+)=': r'set \1=',
    r'\bexport\s+(\w+)\s+': r'set \1=',
    r'\$HOME\b': '%USERPROFILE%',
    r'\$USER\b': '%USERNAME%',
    r'\$PWD\b': '%CD%',
    r'\$PATH\b': '%PATH%',
    r'\$(\w+)\b': r'%(\1)%',
    r'\$\{(\w+)\}': r'%(\1)%',

    # 
    r'\s+>\s+': ' > ',
    r'\s+>>\s+': ' >> ',
    r'\s+2>\s+': ' 2> ',
    r'\s+2>>\s+': ' 2>> ',
    r'\s+2>&1\b': ' 2>&1',
    r'\s+\|\s+': ' | ',

    # 
    r'\becho\s+-e\b': 'echo',
    r'\becho\s+-n\b': 'echo',
    r'\bsleep\b': 'timeout /t',
    r'\bexit 0\b': 'exit /b 0',
    r'\bexit 1\b': 'exit /b 1',
    r'\bset -e\b': None,
    r'\bset -x\b': None,
    r'\bset -u\b': None,
    r'\bsource\b': 'call',
    r'\. /.*\.sh': r'call "\1"',
    r'\.\s+': 'call ',
}

# 
SPECIAL_PATTERNS = [
    # if
    (r'\[\s+-\w+\s+["\']?([^"\'\]]+)["\']?\s+\]', r'IF EXIST "\1"'),  # 
    (r'\[\s+["\']?([^"\'\]]+)["\']?\s+=\s+["\']?([^"\'\]]+)["\']?\s+\]', r'IF "\1"=="\2"'),
    (r'\[\s+["\']?([^"\'\]]+)["\']?\s+!=\s+["\']?([^"\'\]]+)["\']?\s+\]', r'IF NOT "\1"=="\2"'),
    (r'\[\s+(\d+)\s+-eq\s+(\d+)\s+\]', r'IF \1 EQU \2'),
    (r'\[\s+(\d+)\s+-ne\s+(\d+)\s+\]', r'IF \1 NEQ \2'),
    (r'\[\s+(\d+)\s+-gt\s+(\d+)\s+\]', r'IF \1 GTR \2'),
    (r'\[\s+(\d+)\s+-lt\s+(\d+)\s+\]', r'IF \1 LSS \2'),
    (r'\[\s+(\d+)\s+-ge\s+(\d+)\s+\]', r'IF \1 GEQ \2'),
    (r'\[\s+(\d+)\s+-le\s+(\d+)\s+\]', r'IF \1 LEQ \2'),

    # then/fi 
    (r'\bthen\b', ''),
    (r'\bfi\b', ''),

    # for
    (r'\bfor\s+(\w+)\s+in\s+(.+?)\s*;\s*do\b', r'FOR %%(\1) IN (\2) DO'),
    (r'\bfor\s+(\w+)\s+in\s+(.+?)\s+do\b', r'FOR %%(\1) IN (\2) DO'),
    (r'\bdone\b', ''),

    # while
    (r'\bwhile\s+(.+?)\s*;\s*do\b', r'DO WHILE \1'),
    (r'\bwhile\s+(.+?)\s+do\b', r'DO WHILE \1'),

    # 
    (r'(\w+)\(\)\s*\{', r'CALL :\1'),
    (r'function\s+(\w+)\s*\{', r'CALL :\1'),

    # 
    (r'#.*$', ''),  # 
]

# Windows
UNSUPPORTED = [
    'shebang',
    'select',
    'case',
    'function',  # 
    'trap',
    'exec',
    'eval',
    'let',
    'local',
]


class Sh2BatConverter:
    """SHBAT"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.warnings: List[str] = []

    def log(self, msg: str):
        if self.verbose:
            print(f"  [INFO] {msg}")

    def warn(self, msg: str):
        self.warnings.append(msg)
        if self.verbose:
            print(f"  [WARN] {msg}")

    def convert_file(self, sh_path: str, bat_path: Optional[str] = None) -> str:
        """"""
        sh_path = Path(sh_path)

        if not sh_path.exists():
            raise FileNotFoundError(f": {sh_path}")

        if bat_path is None:
            # bat
            bat_path = sh_path.with_suffix('.bat')
        else:
            bat_path = Path(bat_path)

        self.log(f": {sh_path}")

        with open(sh_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        converted = self.convert_content(content)

        # bat
        with open(bat_path, 'w', encoding='utf-8') as f:
            f.write(converted)

        self.log(f": {bat_path}")

        return str(bat_path)

    def convert_content(self, content: str) -> str:
        """"""
        lines = content.split('\n')
        result_lines = []

        # 
        result_lines.append('@echo off')
        result_lines.append('chcp 65001 >nul')
        result_lines.append('setlocal enabledelayedexpansion')
        result_lines.append('')

        # shebang
        skip_next = False
        for i, line in enumerate(lines):
            original_line = line

            # shebang
            if line.startswith('#!') and i == 0:
                self.warn("shebang")
                continue

            # 
            stripped = line.strip()
            if stripped.startswith('#') and not stripped.startswith('#!'):
                # 
                if any(kw in stripped for kw in ['TODO', 'FIXME', 'NOTE', 'WARNING']):
                    comment_text = stripped.lstrip('#').strip()
                    result_lines.append(f'recho {comment_text}')
                continue

            # 
            if not stripped:
                result_lines.append('')
                continue

            # 
            converted_line = self.convert_line(line)

            if converted_line:
                result_lines.append(converted_line)

        # 
        result_lines.append('')
        result_lines.append('pause')

        return '\n'.join(result_lines)

    def convert_line(self, line: str) -> str:
        """"""
        result = line

        # 
        for feature in UNSUPPORTED:
            pattern = rf'\b{feature}\b'
            if re.search(pattern, result, re.IGNORECASE):
                self.warn(f": {feature}")

        # 
        replacements = [
            # 
            (r'\$HOME', '%USERPROFILE%'),
            (r'\$(\w+)', r'%\1%'),
            (r'\$\{(\w+)\}', r'%\1%'),

            # 
            (r'\s*#.*$', ''),

            # exportset
            (r'export\s+(\w+)=(.+)', r'set \1=\2'),
            (r'export\s+(\w+)\s+(.+)', r'set \1=\2'),

            # echo
            (r'echo\s+-e\s+', 'echo '),
            (r'echo\s+-n\s+', 'echo '),
            (r'echo\s+"(.*)"', r'echo \1'),
            (r"echo\s+'(.*)'", r'echo \1'),

            # 
            (r'-f\s+([^)\s]+)', r'IF EXIST "\1"'),
            (r'-d\s+([^)\s]+)', r'IF EXIST "\1"'),

            # Python - 
            (r'python3\b', 'python'),
            (r'pip3\b', 'pip'),
            (r'pip install\b', 'pip install'),
            ('.venv/bin/python', lambda m: '.venv\\Scripts\\python.exe'),
            ('.venv/bin/pip', lambda m: '.venv\\Scripts\\pip.exe'),
            ('.venv/bin/', lambda m: '.venv\\Scripts\\'),

            # Git
            (r'git\s+pull\b', 'git pull'),
            (r'git\s+push\b', 'git push'),
            (r'git\s+clone\b', 'git clone'),
            (r'git\s+status\b', 'git status'),

            # 
            (r'ls\s+', 'dir '),
            (r'ls$', 'dir'),
            (r'cd\s+', 'cd /d '),
            (r'rm\s+-rf\s+', 'rmdir /s /q '),
            (r'rm\s+-r\s+', 'rmdir /s /q '),
            (r'rm\s+-f\s+', 'del /f /q '),
            (r'mkdir\s+-p\s+', 'mkdir '),
            (r'cp\s+', 'copy '),
            (r'mv\s+', 'move '),

            # 
            (r'taskkill\s+/F\s+/PID\b', 'taskkill /F /PID'),
            (r'taskkill\s+/F\s+/IM\b', 'taskkill /F /IM'),

            # 
            (r'\s+>\s+', ' > '),
            (r'\s+>>\s+', ' >> '),
            (r'\s+2>\s+', ' 2> '),
            (r'\s*\|\s*grep\b', '| findstr'),
            (r'\|\s*wc\s+-l\b', ''),

            # 
            (r'\[\s+', 'IF '),
            (r'\s+\]\s*', ''),
            (r'\s+-eq\s+', ' EQU '),
            (r'\s+-ne\s+', ' NEQ '),
            (r'\s+-gt\s+', ' GTR '),
            (r'\s+-lt\s+', ' LSS '),
            (r'\s+-ge\s+', ' GEQ '),
            (r'\s+-le\s+', ' LEQ '),
            (r'\s+&&\s+', ' && '),
            (r'\s+\|\|\s+', ' || '),

            # 
            (r'for\s+(\w+)\s+in\s+', r'FOR %%1 IN ('),
            (r'\s*;\s*do\b', ') DO'),
            (r'\s+do\b', ') DO'),
            (r'\bdone\b', ''),
            (r'\bthen\b', ''),
            (r'\bfi\b', ''),

            # 
            (r'(\w+)\(\)\s*\{', r'CALL :\1'),
            (r'function\s+(\w+)', r':\1'),

            # 
            (r'sleep\s+(\d+)', r'timeout /t \1'),
            (r'sleep\s+(\d+)m', r'timeout /t \1'),
            (r'\bcurl\b\s+-s\b', 'curl'),
            (r'curl\s+-fsSL\b', 'curl -fsSL'),
            (r'timeout\s+/t\s+\d+\s+/nobreak', ''),

            #  - 
            (r'/Users/[^/\s]+', lambda m: m.group(0).replace('/', '\\')),
            (r'/', lambda m: '\\'),
        ]

        for pattern, replacement in replacements:
            if callable(replacement):
                result = re.sub(pattern, replacement, result)
            else:
                try:
                    result = re.sub(pattern, replacement, result)
                except re.error:
                    # 
                    pass

        # 
        result = re.sub(r'\s+', ' ', result)
        result = result.strip()

        # Windows
        unsupported_cmds = ['source ', 'nohup ', 'sudo ', 'which ', 'whereis ']
        for cmd in unsupported_cmds:
            if result.startswith(cmd):
                result = result.replace(cmd, '')

        return result

    def convert_directory(self, dir_path: str, output_dir: Optional[str] = None) -> List[Tuple[str, str]]:
        """sh"""
        dir_path = Path(dir_path)

        if not dir_path.is_dir():
            raise NotADirectoryError(f": {dir_path}")

        output_dir = Path(output_dir) if output_dir else dir_path

        results = []

        for sh_file in dir_path.rglob('*.sh'):
            rel_path = sh_file.relative_to(dir_path)
            bat_file = output_dir / rel_path.with_suffix('.bat')

            # 
            bat_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                self.warnings = []
                self.convert_file(str(sh_file), str(bat_file))
                results.append((str(sh_file), str(bat_file)))

                if self.warnings:
                    for w in self.warnings:
                        print(f"  [WARN] {w}")

            except Exception as e:
                print(f"  [ERROR]  {sh_file}: {e}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description='SHBAT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
:
    python sh2bat.py script.sh              # 
    python sh2bat.py script.sh out.bat       # 
    python sh2bat.py -d ./scripts/          # 
    python sh2bat.py -d ./scripts/ -o ./bat # 
    python sh2bat.py -l                      # 
        '''
    )

    parser.add_argument('input', nargs='?', help='sh')
    parser.add_argument('output', nargs='?', help='bat')
    parser.add_argument('-d', '--directory', help='')
    parser.add_argument('-o', '--output-dir', help='')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-l', '--list-commands', action='store_true', help='')

    args = parser.parse_args()

    # 
    if args.list_commands:
        print("")
        print("\n:")
        print("  cp -> copy, mv -> move, rm -> del")
        print("  mkdir -> mkdir, ls -> dir, cat -> type")
        print("\n:")
        print("  killall -> taskkill, ps -> tasklist")
        print("  nohup -> start ()")
        print("\nPython/:")
        print("  python3 -> python, pip3 -> pip")
        print("  .venv/bin/python -> .venv\\Scripts\\python.exe")
        print("\n:")
        print("  curl -> curl, sleep N -> timeout /t N")
        print("  export VAR=value -> set VAR=value")
        print("  $HOME -> %USERPROFILE%")
        print("\n:")
        for feat in UNSUPPORTED:
            print(f"  - {feat}")
        return

    # 
    if args.directory:
        converter = Sh2BatConverter(verbose=args.verbose)
        print(f": {args.directory}")

        if args.output_dir:
            print(f": {args.output_dir}")

        results = converter.convert_directory(args.directory, args.output_dir)

        print(f"\n!  {len(results)} ")
        for src, dst in results:
            print(f"  {src} -> {dst}")
        return

    # 
    if args.input:
        converter = Sh2BatConverter(verbose=args.verbose)
        try:
            output = converter.convert_file(args.input, args.output)
            print(f"\n!")
            print(f": {output}")

            if converter.warnings:
                print("\n:")
                for w in converter.warnings:
                    print(f"  - {w}")

        except Exception as e:
            print(f"\n: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
