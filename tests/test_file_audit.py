import importlib.util
from pathlib import Path
import sys


def audit_module():
    path=Path(__file__).parents[1]/'scripts'/'check_pi_files.py'
    spec=importlib.util.spec_from_file_location('check_pi_files',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def context(module,tmp_path):
    return module.AuditContext(repo_root=tmp_path/'repo',model_root=tmp_path/'models',
                               data_root=tmp_path/'data',active_models=set(),references={})


def test_private_files_and_biometric_backups_are_always_kept(tmp_path):
    m=audit_module(); ctx=context(m,tmp_path)
    for path in (ctx.repo_root/'.env.bak',ctx.data_root/'attendance.sqlite.bak',
                 ctx.data_root/'enrollment'/'old.jpg'):
        assert m.classify(path,ctx)[0].startswith('KEEP')


def test_active_model_and_its_alias_are_kept(tmp_path):
    m=audit_module(); ctx=context(m,tmp_path)
    ctx.model_root.mkdir()
    model=ctx.model_root/'active.onnx'; model.write_bytes(b'model')
    alias=ctx.model_root/'alias.onnx'; alias.symlink_to(model)
    ctx.active_models.add(model.resolve())
    assert m.classify(model,ctx)[0]=='KEEP_ACTIVE_MODEL'
    assert m.classify(alias,ctx)[0]=='KEEP_ACTIVE_MODEL'


def test_unreferenced_code_backup_is_only_an_archive_candidate(tmp_path):
    m=audit_module(); ctx=context(m,tmp_path)
    path=ctx.repo_root/'src'/'api.py.bak'
    assert m.classify(path,ctx)[0]=='ARCHIVE_CANDIDATE'
    ctx.references[path.name]=['scripts/uses_backup.sh']
    assert m.classify(path,ctx)[0]=='KEEP_REFERENCED'


def test_unused_model_is_reviewed_instead_of_automatically_deleted(tmp_path):
    m=audit_module(); ctx=context(m,tmp_path)
    assert m.classify(ctx.model_root/'backup.onnx',ctx)[0]=='REVIEW_MODEL'
