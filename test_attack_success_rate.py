import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
from CLGAN_rl import CLGAN_Attack, AdvGAN_Attack
import models
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader

use_cuda = True
image_nc = 1
batch_size = 128
max_samples = 100

gen_input_nc = image_nc

# Define what device we are using
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Load the target model and MNIST dataset
pretrained_model = "./MNIST_target_model.pth"
targeted_model = models.MNISTClassifier().to(device)
targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.eval()

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))  # Normalize the image data
# ])
# dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# dataset = torch.utils.data.Subset(dataset, range(max_samples))

mnist_dataset_test = torchvision.datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)
test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=True, num_workers=1)

# Define the attack methods and parameters
# attack_methods = ['FGSM', 'PGD', 'CLGAN']
attack_methods = ['CLGAN']
attack_params = {
    'FGSM': {
        'epsilon': 0.1
    },
    'PGD': {
        'epsilon': 0.1,
        'alpha': 0.01,
        'num_steps': 40
    },
    'CLGAN': {
        'epsilon': 0.3,
        #'model_A': YourDistilledModel(),
        #'model_B': target_model
    }
}

# load the generator of adversarial examples
pretrained_generator_path = './models/netG_epoch_900.pth'
pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

# # Define the evaluation function
# def evaluate_attack_success_rate(attack_method, attack_params):
#     success_rates = []
#     for i in range(len(dataset)):
#         # Get the original image and label
#         image, label = dataset[i]
#         image = image.to(device)

#         # Generate the adversarial example
#         if attack_method == 'CLGAN':
#             epsilon = 0.3
#             perturbation = pretrained_G(image.unsqueeze(0))
#             perturbation = torch.clamp(perturbation[0], -epsilon, epsilon)
#             adv_image = image + perturbation
#             adv_image = torch.clamp(adv_image, 0, 1)
#             adv_image = adv_image.cpu()
#             image1 = Image.fromarray((adv_image[0].detach().numpy() * 255).astype(np.uint8))
#             image1 = image1.convert("L")
#             image1.save('image.png')

#         if attack_method == 'FGSM':
#             epsilon = attack_params['epsilon']
#             image.requires_grad = True
#             output = targeted_model(image.unsqueeze(0).to(device))
#             loss = nn.CrossEntropyLoss()(output, torch.tensor([label]).to(device))
#             loss.backward()
#             perturbation = torch.sign(image.grad.data)
#             adv_image = torch.clamp(image + epsilon * perturbation, 0, 1)

#         elif attack_method == 'PGD':
#             epsilon = attack_params['epsilon']
#             alpha = attack_params['alpha']
#             num_steps = attack_params['num_steps']
#             adv_image = image.clone().detach().to(device)
#             adv_image.requires_grad = True

#             for _ in range(num_steps):
#                 output = targeted_model(adv_image.unsqueeze(0))
#                 loss = nn.CrossEntropyLoss()(output, torch.tensor([label]).to(device))
#                 grad = torch.autograd.grad(loss, adv_image)[0]
#                 perturbation = alpha * torch.sign(grad.data)
#                 perturbation = torch.clamp(perturbation, -epsilon, epsilon)
#                 adv_image = torch.clamp(adv_image + perturbation, 0, 1)

#         image1 = Image.fromarray((adv_image[0].detach().cpu().numpy() * 255).astype(np.uint8))
#         image1 = image1.convert("L")
#         image1.save('image.png')

#         # Perform predictions on the target model
#         original_output = targeted_model(image.unsqueeze(0).to(device))
#         adv_output = targeted_model(adv_image.unsqueeze(0).to(device))

#         # Check if the attack is successful
#         original_pred = torch.argmax(original_output, dim=1)
#         adv_pred = torch.argmax(adv_output, dim=1)
#         success = (original_pred != adv_pred).item()
#         print(perturbation)

#         success_rates.append(success)

#         true_model = targeted_model(image.unsqueeze(0).to(device))
#         true_probs_model = F.softmax(true_model, dim=1)
#         true_label = torch.argmax(true_probs_model, axis=1)
#         true_label = torch.unsqueeze(true_label, dim=1)
#         true_probability = (torch.gather(true_probs_model, dim=1, index=true_label)).reshape(image.shape[0], )
#         adv_model = targeted_model(adv_image.unsqueeze(0).to(device))
#         adv_probs_model = F.softmax(adv_model, dim=1)
#         adv_probability = (torch.gather(adv_probs_model, dim=1, index=true_label)).reshape(image.shape[0], )
#         attack = true_probability - adv_probability
#         attack = torch.clamp(attack, min=0)

#     attack_success_rate = sum(success_rates) / len(dataset)
#     return attack_success_rate

# Define the evaluation function
def evaluate_attack_success_rate(attack_method, attack_params):
    success_rates = []
    for i, data in enumerate(test_dataloader, 0):
        # Get the original image and label
        image, label = data
        image = image.to(device)

        # Generate the adversarial example
        if attack_method == 'CLGAN':
            epsilon = 0.1
            perturbation = pretrained_G(image)
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)
            adv_image = image + perturbation
            adv_image = torch.clamp(adv_image, 0, 1)
            adv_image = adv_image.cpu()
            # image1 = Image.fromarray((adv_image[0].detach().numpy() * 255).astype(np.uint8))
            # image1 = image1.convert("L")
            # image1.save('image.png')

        if attack_method == 'FGSM':
            epsilon = attack_params['epsilon']
            image.requires_grad = True
            output = targeted_model(image.unsqueeze(0).to(device))
            loss = nn.CrossEntropyLoss()(output, torch.tensor([label]).to(device))
            loss.backward()
            perturbation = torch.sign(image.grad.data)
            adv_image = torch.clamp(image + epsilon * perturbation, 0, 1)

        elif attack_method == 'PGD':
            epsilon = attack_params['epsilon']
            alpha = attack_params['alpha']
            num_steps = attack_params['num_steps']
            adv_image = image.clone().detach().to(device)
            adv_image.requires_grad = True

            for _ in range(num_steps):
                output = targeted_model(adv_image.unsqueeze(0))
                loss = nn.CrossEntropyLoss()(output, torch.tensor([label]).to(device))
                grad = torch.autograd.grad(loss, adv_image)[0]
                perturbation = alpha * torch.sign(grad.data)
                perturbation = torch.clamp(perturbation, -epsilon, epsilon)
                adv_image = torch.clamp(adv_image + perturbation, 0, 1)

        # image1 = Image.fromarray((adv_image[0].detach().cpu().numpy() * 255).astype(np.uint8))
        # image1 = image1.convert("L")
        # image1.save('image.png')

        # Perform predictions on the target model
        original_output = targeted_model(image.to(device))
        adv_output = targeted_model(adv_image.to(device))

        # Check if the attack is successful
        original_pred = torch.argmax(original_output, dim=1)
        adv_pred = torch.argmax(adv_output, dim=1).to(device)
        success = torch.sum(label.to(device) != adv_pred).item()
        print(original_pred)
        print(adv_pred)

        success_rates.append(success)

        # true_model = targeted_model(image.unsqueeze(0).to(device))
        # true_probs_model = F.softmax(true_model, dim=1)
        # true_label = torch.argmax(true_probs_model, axis=1)
        # true_label = torch.unsqueeze(true_label, dim=1)
        # true_probability = (torch.gather(true_probs_model, dim=1, index=true_label)).reshape(image.shape[0], )
        # adv_model = targeted_model(adv_image.unsqueeze(0).to(device))
        # adv_probs_model = F.softmax(adv_model, dim=1)
        # adv_probability = (torch.gather(adv_probs_model, dim=1, index=true_label)).reshape(image.shape[0], )
        # attack = true_probability - adv_probability
        # attack = torch.clamp(attack, min=0)

    attack_success_rate = sum(success_rates) / len(mnist_dataset_test)
    return attack_success_rate

# Evaluate each attack method
for attack_method in attack_methods:
    params = attack_params[attack_method]
    attack_success_rate = evaluate_attack_success_rate(attack_method, params)
    print(f'{attack_method}: Attack Success Rate = {attack_success_rate}')
