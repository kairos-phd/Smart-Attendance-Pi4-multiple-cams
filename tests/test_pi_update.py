import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


def module():
    path = Path(__file__).parents[1] / 'scripts' / 'update_pi.py'
    spec = importlib.util.spec_from_file_location('update_pi', path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def test_gate_prevents_activation_for_incompatible_catalog():
    m = module()
    report = {'catalog': {'incompatible_users':3}}
    with pytest.raises(RuntimeError, match='incompatible'): m.require_compatible(report, False)
    m.require_compatible(report, True)


def test_untracked_overwrite_candidates_are_preserved(tmp_path):
    m = module()
    repo = tmp_path / 'repo'; repo.mkdir()
    def git(*args): return subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True)
    git('init', '-q'); git('config', 'user.email', 'test@example.org'); git('config', 'user.name', 'Test')
    (repo / 'old.py').write_text('old')
    git('add', '.'); git('commit', '-qm', 'base')
    base = git('rev-parse', 'HEAD').stdout.decode().strip()
    (repo / 'incoming.py').write_text('upstream')
    git('add', '.'); git('commit', '-qm', 'incoming')
    target = git('rev-parse', 'HEAD').stdout.decode().strip()
    git('switch', '--detach', base)
    (repo / 'incoming.py').write_text('local Pi edit')
    (repo / 'spare.bak').write_text('keep')
    backup = tmp_path / 'backup'; backup.mkdir()
    moved = m.preserve_collisions(repo, target, backup)
    assert moved == ['incoming.py']
    assert (backup / 'collisions' / 'incoming.py').read_text() == 'local Pi edit'
    assert (repo / 'spare.bak').read_text() == 'keep'
    assert not (repo / 'incoming.py').exists()


def test_private_config_updates_only_requested_keys(tmp_path):
    m = module()
    path = tmp_path / '.env'
    original = 'ADMIN_SHARED_TOKEN=secret\nMODEL_PACK=original\nPREVIEW_COOKIE_SECURE=true\n'
    path.write_text(original)
    m.update_config(path, {'PREVIEW_COOKIE_SECURE':'false'})
    assert path.read_text() == original.replace('PREVIEW_COOKIE_SECURE=true','PREVIEW_COOKIE_SECURE=false')
    assert path.stat().st_mode & 0o777 == 0o600


def test_completed_moves_are_journaled_when_later_move_fails(tmp_path, monkeypatch):
    m = module()
    repo = tmp_path/'repo'; repo.mkdir()
    backup = tmp_path/'backup'; backup.mkdir()
    for name in ('a.py', 'b.py'): (repo/name).write_text('local')
    monkeypatch.setattr(m, 'git', lambda *a: 'a.py\0b.py\0')
    original_move = m.shutil.move
    def fail_second(source, destination):
        if str(source).endswith('b.py'): raise OSError('forced second move failure')
        return original_move(source, destination)
    monkeypatch.setattr(m.shutil, 'move', fail_second)
    journal = []
    with pytest.raises(OSError): m.preserve_collisions(repo, 'commit', backup, journal)
    assert journal == ['a.py']
    assert (backup/'collisions'/'a.py').read_text() == 'local'
    assert (backup/'moved-untracked.json').exists()
    assert (repo/'b.py').read_text() == 'local'


@pytest.mark.parametrize('failure', ['pip', 'late_enrollment'])
def test_activation_failure_restores_local_edits_and_restarts_old_service(tmp_path, monkeypatch, failure):
    import json
    m = module()
    repo = tmp_path/'repo'; repo.mkdir()
    real_run = subprocess.run
    def repo_git(*args): return real_run(['git', *args], cwd=repo, capture_output=True, check=True)
    repo_git('init','-q'); repo_git('config','user.email','test@example.org'); repo_git('config','user.name','Test')
    (repo/'source.py').write_text('original')
    repo_git('add','.'); repo_git('commit','-qm','old')
    old = repo_git('rev-parse','HEAD').stdout.decode().strip()
    (repo/'source.py').write_text('new')
    (repo/'incoming.py').write_text('upstream module')
    repo_git('add','.')
    repo_git('commit','-qm','new')
    new = repo_git('rev-parse','HEAD').stdout.decode().strip()
    repo_git('switch','--detach',old)
    (repo/'source.py').write_text('local original edit')
    (repo/'incoming.py').write_text('local untracked module')
    (repo/'.env').write_text('PREVIEW_COOKIE_SECURE=true\n')
    (repo/'.venv'/'bin').mkdir(parents=True)
    (repo/'.venv'/'bin'/'python').touch()
    events = []; installs = []
    monkeypatch.setattr(m, 'service_environment', lambda _: {})
    def mocked_run(cmd, **kwargs):
        if str(cmd[0]) == str(repo/'.venv'/'bin'/'python'):
            if 'pi4_attendance.tools.model_info' in cmd:
                path = Path(cmd[cmd.index('--output')+1])
                path.write_text(json.dumps({'catalog':{'incompatible_users':0},'model_info':{'selection':'pack'},
                    'data_root':str(tmp_path/'data'),'api_enabled':False,'api_port':8080,'model_version':'new'}))
            if '-c' in cmd and 'catalog_compatibility' in cmd[cmd.index('-c')+1]:
                return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({'incompatible_users':1 if failure == 'late_enrollment' else 0}))
            if 'install' in cmd:
                installs.append(cmd)
                if kwargs.get('check'): raise subprocess.CalledProcessError(1, cmd)
                return subprocess.CompletedProcess(cmd, 1)
            return subprocess.CompletedProcess(cmd, 0)
        if cmd[0] in ('sudo','systemctl'):
            events.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)
        return real_run(cmd, **kwargs)
    monkeypatch.setattr(m.subprocess, 'run', mocked_run)
    error_type = RuntimeError if failure == 'late_enrollment' else subprocess.CalledProcessError
    with pytest.raises(error_type):
        m.main(['--repo',str(repo),'--ref',new,'--apply','--preview-transport','http'])
    assert (repo/'source.py').read_text() == 'local original edit'
    assert (repo/'incoming.py').read_text() == 'local untracked module'
    if failure == 'late_enrollment': assert not installs
    assert (repo/'.env').read_text() == 'PREVIEW_COOKIE_SECURE=true\n'
    assert events[-1] == ['sudo','systemctl','start','sic-pi4-attendance']
    assert all('--no-index' in cmd and '--no-build-isolation' in cmd for cmd in installs)
