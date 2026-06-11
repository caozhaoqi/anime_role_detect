"""验证完整模型加载 + 标签生成"""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath('.'))
os.environ['PYTORCH_MPS_DISABLE'] = '1'

print('1: loading tagger module...', flush=True)
from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
print('2: module loaded', flush=True)

t = WDViTV3Tagger.get_instance()
print(f'3: instance OK, device={t.device}', flush=True)

print('4: loading model...', flush=True)
ok = t.load_model()
print(f'5: model loaded: {ok}', flush=True)

if ok:
    # Test with a sample image
    from PIL import Image
    img = Image.new('RGB', (448, 448), color='red')
    tags = t.generate_tags(img, threshold=0.1)
    print(f'6: generated {len(tags)} tags', flush=True)
    if tags:
        print(f'7: top tag: {tags[0]}', flush=True)
else:
    print('6: model not loaded, skipping inference test', flush=True)

print('7: DONE', flush=True)