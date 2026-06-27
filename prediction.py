import torch
from PIL import Image
from torch import nn
from torchvision import transforms
import torchvision.models as models

trained_model = None
fruit_classes = ['Fresh Banana','Fresh Lemon','Fresh Lulo','Fresh Mango','Fresh Orange','Fresh Strawberry','Fresh Tamarillo','Fresh Tomato',
                 'Spoiled Banana','Spoiled Lemon','Spoiled Lulo','Spoiled Mango','Spoiled Orange','Spoiled Strawberry','Spoiled Tamarillo','Spoiled Tomato']
class FreshnessClassifier(nn.Module):
    def __init__(self,num_classes):
        super().__init__()

        self.model = models.resnet50(weights = 'DEFAULT')
        in_features = self.model.fc.in_features

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def predict(uploaded_image):
    image = Image.open(uploaded_image).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    global trained_model
    if trained_model is None:
        trained_model = FreshnessClassifier(len(fruit_classes))
        trained_model.load_state_dict(torch.load("./resnet50_model.pth"))
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _,predicted = torch.max(output,1)
        return fruit_classes[predicted.item()]