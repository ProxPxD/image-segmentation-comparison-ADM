from torchsummary import summary
from model import UNet
from parameters import Parameters

models = {
    'UNet': UNet(Parameters.permutated_image_size[0], Parameters.n_classes).to(Parameters.device),
}

for model_name, model in models.items():
    print(f'{model_name}_summary: ', summary(model, Parameters.permutated_image_size))
    print('\n\n' + '*'*40)
