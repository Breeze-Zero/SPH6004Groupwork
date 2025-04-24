from models.general_model import BaseModel,Img_text_Model
import timm
from models.emb_model import *
from models.img_text_model import *
from models.classifiers import *
from models.LT_VIT import *
# from monai.networks.nets import Transchex
# from ikan.GroupKAN import GroupKAN

def create_model(name,**kwargs):
    # name = name.split('_')[0]
    num_classes = kwargs.get('num_classes')
    if name=='MLP':
        backbone = Mlp(**kwargs)
        num_features = kwargs.get('num_features')
        head = kwargs.get('Classifier')
        if 'SeparateClassifier' in head:
            head = SeparateClassifier(in_features=num_features, num_classes=num_classes)
        else:
            head = Classifier(in_features=num_features, num_classes=num_classes)
        model = BaseModel(model = backbone,head = head,**kwargs)
    elif name=='SwiGLUFFN':
        backbone = SwiGLUFFN(**kwargs)
        num_features = kwargs.get('num_features')
        head = kwargs.get('Classifier')
        if 'SeparateClassifier' in head:
            head = SeparateClassifier(in_features=num_features, num_classes=num_classes)
        else:
            head = Classifier(in_features=num_features, num_classes=num_classes)
        model = BaseModel(model = backbone,head = head,**kwargs)
    elif name=='KAN':
        num_features = kwargs.get('num_features')
        backbone = GroupKAN([num_features,4*num_features,num_features],drop=kwargs.get('drop'))
        head = kwargs.get('Classifier')
        if 'SeparateClassifier' in head:
            head = SeparateClassifier(in_features=num_features, num_classes=num_classes)
        else:
            head = Classifier(in_features=num_features, num_classes=num_classes)
        model = BaseModel(model = backbone,head = head,**kwargs)
    elif name=='Linear':
        backbone = nn.Identity()
        num_features = kwargs.get('num_features')
        head = kwargs.get('Classifier')
        if 'SeparateClassifier' in head:
            head = SeparateClassifier(in_features=num_features, num_classes=num_classes)
        else:
            head = Classifier(in_features=num_features, num_classes=num_classes)
        model = BaseModel(model = backbone,head = head,**kwargs)
    elif name =='Transchex':
        model =  Transchex(
            in_channels=3,
            img_size=(224, 224),
            num_classes=13,
            patch_size=(14, 14),
            num_language_layers=2,
            num_vision_layers=2,
            num_mixed_layers=2,
        )

    elif name=='LTViT':
        model = LabelGuidedTransformer(4,num_classes)
    elif 'img_text' in name:
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        if hasattr(backbone, 'num_features'):
            num_features = backbone.num_features
        else:
            raise ValueError(f"model {name} has no 'num_features'")
        head = kwargs.get('Classifier')
        if 'SeparateClassifier' in head:
            head = SeparateClassifier(in_features=512, num_classes=num_classes)
        else:
            head = Classifier(in_features=512, num_classes=num_classes)
        model = Img_text_Model(backbone,FusionStack(num_layers=4,img_dim = num_features,text_dim=768),num_features = num_features, head = head)

    elif 'Dinov2' in name:
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        if hasattr(backbone, 'num_features'):
            num_features = backbone.num_features
        else:
            raise ValueError(f"model {name} has no 'num_features'")
        # for param in backbone.parameters():
        #     param.requires_grad = False
        head = kwargs.get('Classifier')
        if 'SeparateClassifier' in head:
            head = SeparateClassifier(in_features=num_features, num_classes=num_classes)
        else:
            head = Classifier(in_features=num_features, num_classes=num_classes)
        model = BaseModel(model = backbone, num_features = num_features, head = head,**kwargs)

    elif name=='VisionLSTM':
        backbone = torch.hub.load("nx-ai/vision-lstm", "vil2-base")
        backbone.head = None
        num_features = 1536
        head = kwargs.get('Classifier')
        if 'SeparateClassifier' in head:
            head = SeparateClassifier(in_features=num_features, num_classes=num_classes)
        else:
            head = Classifier(in_features=num_features, num_classes=num_classes)
        model = BaseModel(model = backbone, num_features = num_features, head = head,**kwargs)
        
    else:
        pretrained = kwargs.get('pretrained')
        backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
        if hasattr(backbone, 'num_features'):
            num_features = backbone.num_features
        else:
            raise ValueError(f"model {name} has no 'num_features'")
        head = kwargs.get('Classifier')
        if 'SeparateClassifier' in head:
            head = SeparateClassifier(in_features=num_features, num_classes=num_classes)
        else:
            head = Classifier(in_features=num_features, num_classes=num_classes)
        model = BaseModel(model = backbone, num_features = num_features, head = head,**kwargs)

    
    return model