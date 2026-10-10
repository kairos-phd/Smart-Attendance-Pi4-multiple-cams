import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from pi4_attendance.face_engine import FaceEngine


def recognizer(path, quantized=True, zero_point=3):
    vector=np.zeros((1,512),np.float32)
    vector[0,:2]=[0.5,1.0]
    nodes=[helper.make_node('Constant',[],['float_embedding'],value=numpy_helper.from_array(vector))]
    initializers=[]
    if quantized:
        initializers=[numpy_helper.from_array(np.array(0.125,np.float32),'scale'),
                      numpy_helper.from_array(np.array(zero_point,np.int8),'zero')]
        nodes.append(helper.make_node('QuantizeLinear',['float_embedding','scale','zero'],['embedding']))
        output_type=TensorProto.INT8
    else:
        nodes.append(helper.make_node('Identity',['float_embedding'],['embedding']))
        output_type=TensorProto.FLOAT
    graph=helper.make_graph(nodes,'fixture',
        [helper.make_tensor_value_info('image',TensorProto.FLOAT,[1,3,112,112])],
        [helper.make_tensor_value_info('embedding',output_type,[1,512])],initializers)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)],ir_version=10)
    onnx.save(model,str(path))
    return vector.reshape(-1)


@pytest.mark.parametrize('quantized',[True,False])
def test_external_model_reads_graph_quantization_and_normalizes(tmp_path,monkeypatch,quantized):
    pack=tmp_path/'models'/'pack'
    pack.mkdir(parents=True)
    detector=pack/'det.onnx'
    detector.write_bytes(b'detector-fixture')
    model=tmp_path/'models'/'external.onnx'
    vector=recognizer(model,quantized)
    class Analysis:
        def __init__(self,**kwargs): self.models={'detection':SimpleNamespace(model_file=str(detector))}
        def prepare(self,**kwargs): pass
    monkeypatch.setitem(sys.modules,'insightface.app',SimpleNamespace(FaceAnalysis=Analysis))
    monkeypatch.setitem(sys.modules,'insightface.utils',SimpleNamespace(
        face_align=SimpleNamespace(norm_crop=lambda image,**kwargs:np.zeros((112,112,3),np.uint8))))
    engine=FaceEngine(tmp_path,'pack',320,0.45,0.06,recognizer_model=model)
    engine.start()
    output=engine._embedding_float(np.zeros((32,32,3),np.uint8),
                                   SimpleNamespace(kps=np.zeros((5,2),np.float32)))
    np.testing.assert_allclose(output,vector/np.linalg.norm(vector),atol=0.005)
    assert str(model) in engine.model_info['files']
    if quantized:
        assert engine.model_info['output_scale']==0.125
        assert engine.model_info['output_zero_point']==3


def test_explicit_missing_recognizer_fails_before_silent_fallback(tmp_path):
    engine=FaceEngine(tmp_path,'pack',320,0.45,0.06,recognizer_model=tmp_path/'missing.onnx')
    with pytest.raises(FileNotFoundError,match='recognizer'):
        engine.start()
