import runpy
from pathlib import Path
import pytest


def test_dashboard_updater_preserves_model_and_rolls_back_interruption(tmp_path, monkeypatch):
    module=runpy.run_path(str(Path(__file__).parents[1]/'scripts/update_dashboard_only.py'))
    root=tmp_path/'src'/'pi4_attendance';root.mkdir(parents=True)
    api=root/'api.py';api.write_text('def create_app(runtime):\n    require_admin = None\n    app = None\n    return app\n')
    engine=root/'face_engine.py';engine.write_text('unchanged')
    before=api.read_bytes()
    candidate=Path(__file__).parents[1]/'src/pi4_attendance/dashboard_feed.py'
    assert module['update'](tmp_path,candidate)['changed']
    replace=module['os'].replace
    def interrupt(source,target):
        if target==api: raise KeyboardInterrupt()
        replace(source,target)
    monkeypatch.setattr(module['os'],'replace',interrupt)
    with pytest.raises(KeyboardInterrupt):module['update'](tmp_path,candidate,True)
    assert api.read_bytes()==before and not (root/'dashboard_feed.py').exists()
    monkeypatch.setattr(module['os'],'replace',replace)
    result=module['update'](tmp_path,candidate,True)
    assert (Path(result['backup_directory'])/'api.py').read_bytes()==before
    assert engine.read_text()=='unchanged'
    assert module['update'](tmp_path,candidate,True)['changed'] is False
