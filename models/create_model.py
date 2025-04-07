from models.general_model import BaseModel
import timm
from models.emb_model import *
from models.classifiers import *

def create_model(name,**kwargs):
    num_classes = kwargs.get('num_classes')
    if name=='MLP':
        backbone = Mlp(**kwargs)
        num_features = kwargs.get('num_features')
        head = SeparateClassifier(in_features = num_features, num_classes = num_classes)
        model = BaseModel(model = backbone,head = head,**kwargs)
    elif name=='Linear':
        backbone = nn.Identity()
        num_features = kwargs.get('num_features')
        head = SeparateClassifier(in_features = num_features, num_classes = num_classes)
        model = BaseModel(model = backbone,head = head,**kwargs)
    elif name=='Dinov2':
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        if hasattr(backbone, 'num_features'):
            num_features = backbone.num_features
        else:
            raise ValueError(f"model {name} has no 'num_features'")
        head = SeparateClassifier(in_features=num_features, num_classes=num_classes)
        model = BaseModel(model = backbone, num_features = num_features, head = head,**kwargs)

    elif name=='VisionLSTM':
        backbone = torch.hub.load("nx-ai/vision-lstm", "vil2-base")
        backbone.head = None
        if hasattr(backbone, 'head_dim'):
            num_features = backbone.head_dim
        else:
            raise ValueError(f"model {name} has no 'num_features'")
        head = SeparateClassifier(in_features=num_features, num_classes=num_classes)
        model = BaseModel(model = backbone, num_features = num_features, head = head,**kwargs)
        
    else:
        backbone = timm.create_model(name, pretrained=True, num_classes=0)
        if hasattr(backbone, 'num_features'):
            num_features = backbone.num_features
        else:
            raise ValueError(f"model {name} has no 'num_features'")
        head = SeparateClassifier(in_features=num_features, num_classes=num_classes)
        model = BaseModel(model = backbone, num_features = num_features, head = head,**kwargs)

    
    return model