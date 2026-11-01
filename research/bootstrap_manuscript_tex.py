"""Fetch pinned official manuscript classes/styles into a new local directory."""
import argparse
import hashlib
import io
from pathlib import Path
import subprocess
from urllib.request import urlopen
from zipfile import ZipFile


SOURCES=[
    ('springer',
     'https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/18782940/data/v12',
     '812e76dcaa9c28dc1bff1fb6065d51729b67d4ea140552a05088317414a3ecae',
     {'sn-jnl.cls','sn-basic.bst'}),
    ('elsarticle','https://mirrors.ctan.org/macros/latex/contrib/elsarticle.zip',
     'f25a3e1b006e0f86abb091a02f3afa7c6d60491665c7aa89f70cc5609e14d9dc',
     {'elsarticle.dtx','elsarticle.ins','elsarticle-num.bst','elsarticle-harv.bst'})]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise SystemExit('Output exists; choose a new dependency directory')
    archives=[]
    for name,url,expected,members in SOURCES:
        with urlopen(url,timeout=30) as response:
            data=response.read(16*1024*1024+1)
        if len(data)>16*1024*1024 or hashlib.sha256(data).hexdigest()!=expected:
            raise SystemExit('Official archive hash changed or size invalid: '+name)
        with ZipFile(io.BytesIO(data)) as archive:
            selected={Path(n).name:archive.read(n) for n in archive.namelist() if Path(n).name in members}
        if set(selected)!=members:
            raise SystemExit('Official archive is missing required files: '+name)
        archives.append((name,data,selected))
    args.out.mkdir(parents=True)
    for name,data,selected in archives:
        (args.out/(name+'.zip')).write_bytes(data)
        for filename,content in selected.items():
            (args.out/filename).write_bytes(content)
    with (args.out/'docstrip.log').open('w') as log:
        subprocess.run(['latex','-no-shell-escape','-interaction=batchmode','elsarticle.ins'],
                       cwd=args.out,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Official class/style dependencies verified in '+str(args.out))


if __name__=='__main__':
    main()
