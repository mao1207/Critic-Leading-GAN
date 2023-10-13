import torch.nn as nn
import torch
import numpy as np
import models
import torch.nn.functional as F
import torchvision
import os
import random
from PIL import Image
from math import*

models_path = './models/'


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.05)
        nn.init.constant_(m.bias.data, 0)

# def normalize_tensor(tensor):
#     min_value = tensor.min()
#     max_value = tensor.max()
#     normalized_tensor = (tensor - min_value) / (max_value - min_value)
#     return normalized_tensor

class CLGAN_Attack:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 box_min,
                 box_max):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max

        self.gen_input_nc = image_nc
        self.netG = models.Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = models.Discriminator(image_nc).to(device)
        self.critic = models.Critic(image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)
        self.critic.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.01)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.001)
        self.optimizer_C = torch.optim.Adam(self.critic.parameters(),
                                            lr=0.0001)

        self.epsilon = 0.3

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, x, epoch):
        acc_suc = 0
        acc_sum = 0

        # optimize D
        for i in range(1):
            self.epsilon = 0.3
            perturbation = self.netG(x)
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)

            # add a clipping trick
            adv_images = x + perturbation
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            self.optimizer_G.zero_grad()
            loss_G_fake.backward()
            #self.optimizer_G.step()

            # calculate perturbation norm
            C = 0.1
            loss_perturb = torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1).to(self.device)
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            # cal adv loss
            true_model = self.model(x)
            true_probs_model = F.softmax(true_model, dim=1)
            true_label = torch.argmax(true_probs_model, axis=1)
            true_label = torch.unsqueeze(true_label, dim=1)
            true_probability = (torch.gather(true_probs_model, dim=1, index=true_label)).reshape(x.shape[0], )
            adv_model = self.model(adv_images)
            adv_probs_model = F.softmax(adv_model, dim=1)
            adv_probability = (torch.gather(adv_probs_model, dim=1, index=true_label)).reshape(x.shape[0], )
            if epoch < 10:
                attack = true_probability - adv_probability + torch.norm(true_probs_model - adv_probs_model, dim=1) * torch.norm(true_probs_model - adv_probs_model, dim=1)
            else:
                # loss_perturb = torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1)
                # attack = -loss_perturb
                attack = true_probability - adv_probability
                attack = torch.where(loss_perturb < torch.median(loss_perturb), attack * 1.1, attack)

            attack_label = torch.where(attack > torch.median(attack), 0, 1)
            attack_mean = torch.mean(attack)
            loss_perturb_mean = torch.mean(loss_perturb)

            original_output = self.model(x)
            adv_output = self.model(x + perturbation)

            original_pred = torch.argmax(original_output, dim=1)
            adv_pred = torch.argmax(adv_output, dim=1)
            success = (original_pred != adv_pred)
            acc_suc += torch.sum(success).item()
            acc_sum += success.shape[0]

            self.optimizer_G.zero_grad()
            perturbation = self.netG(x)
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
            adv_images = x + perturbation
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)
            judge = self.critic(x.detach(), adv_images)
            top20_probs_0, top20_indices_0 = torch.topk(judge[:, 0], k=40)
            top20_p0_0 = judge[top20_indices_0, 0]
            top20_p1_0 = judge[top20_indices_0, 1]
            top20_probs_1, top20_indices_1 = torch.topk(judge[:, 1], k=40)
            top20_p0_1 = judge[top20_indices_1, 0]
            top20_p1_1 = judge[top20_indices_1, 1]
            mark_mean = torch.mean(torch.sum(top20_p0_0 - top20_p1_0) + torch.sum(top20_p0_1 - top20_p1_1))
            # print('0:', torch.argmax(judge[top20_indices_0], dim = 1))
            # print('1:', torch.argmax(judge[top20_indices_1], dim = 1))
            # mark_mean = F.mse_loss(judge.squeeze(), torch.zeros_like(judge[:,0], device=self.device))
            if epoch >= 0:
                loss_perturb = torch.clamp(torch.norm(perturbation.view(perturbation.shape[0], -1), 1.8, dim=1), min=3, max=float('inf'))
                loss_perturb = torch.mean(loss_perturb)
                pert_lambda = torch.from_numpy(np.full((x.shape[0],), 1)).to(self.device)
                mark_mean -= torch.mean(pert_lambda * loss_perturb)
            loss_G = -mark_mean
            # if epoch < 3:
            #     loss_G -= torch.var(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1), dim=0)

            loss_G.backward()
            self.optimizer_G.step()

            self.optimizer_C.zero_grad()
            judge = self.critic(x.detach(), adv_images.detach()).to(self.device)

            # loss_cl = F.mse_loss(judge.squeeze(), attack_label.float())
            loss_cl = F.cross_entropy(judge, attack_label)
            loss_cl.backward()
            self.optimizer_C.step()

            adv_image = adv_images[0].cpu()
            image1 = Image.fromarray((adv_image[0].detach().numpy() * 255).astype(np.uint8))
            image1 = image1.convert("L")
            image1.save('image.png')

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb_mean.item(), attack_mean.item(), mark_mean.item(), loss_cl, acc_suc/acc_sum

    def train(self, train_dataloader, epochs):
        # pretrained_model = "./models/netC_pretrain_epoch_60.pth"
        # self.critic.load_state_dict(torch.load(pretrained_model))
        # self.critic.eval()
        # pretrained_model = "./models/netG_epoch_4.pth"
        # self.netG.load_state_dict(torch.load(pretrained_model))
        # self.netG.eval()

        # G_model = "./models/netG_epoch_100.pth"
        # self.netG.load_state_dict(torch.load(G_model))
        # self.netG.eval()
        # D_model = "./models/netD_epoch_100.pth"
        # self.netDisc.load_state_dict(torch.load(D_model))
        # self.netDisc.eval()
        # critic_model = "./models/netC_epoch_100.pth"
        # self.critic.load_state_dict(torch.load(critic_model))
        # self.critic.eval()

        # Pre-training
        # for epoch in range(0, 61):
        #     for batch in train_dataloader:
        #         image_batch, label_batch = batch
        #         mark_sum = 0
        #         loss_sum = 0
        #         mark, loss_cl = self.pre_train_batch(image_batch.to(self.device), label_batch.to(self.device))
        #         mark_sum += mark
        #         loss_sum += loss_cl
        #
        #     if epoch % 20 == 0:
        #         netC_file_name = models_path + 'netC_pretrain_epoch_' + str(epoch) + '.pth'
        #         torch.save(self.critic.state_dict(), netC_file_name)
        #
        #     print("(pre-train) epoch %d / %d: critic_mark: %.3f, loss_critic: %.3f\n" %
        #       (epoch, epochs + 1, mark_sum/len(batch), loss_sum/len(batch) ))

        for epoch in range(1, 10001):
            if epoch == 200:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001)
                self.optimizer_C = torch.optim.Adam(self.critic.parameters(),
                                                    lr=0.0001)
            if epoch == 400:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.00001)
                self.optimizer_C = torch.optim.Adam(self.critic.parameters(),
                                                    lr=0.00001)
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            critic_mark_sum = 0
            loss_critic_sum = 0
            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch, critic_mark_batch, loss_critic_batch, attack_success_rate = \
                    self.train_batch(images, epoch)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch
                critic_mark_sum += critic_mark_batch
                loss_critic_sum += loss_critic_batch
                # print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
                #  \nloss_perturb: %f, loss_adv: %f, critic_mark: %f, loss_critic: %f, acc: %f\n" %
                #       (epoch, loss_D_batch, loss_G_fake_batch,
                #        loss_perturb_batch, loss_adv_batch, critic_mark_batch,
                #        loss_critic_batch, attack_success_rate))

            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %f, loss_adv: %f, critic_mark: %f, loss_critic: %f, acc: %f\n" %
                  (epoch, loss_D_sum / num_batch, loss_G_fake_sum / num_batch,
                   loss_perturb_sum / num_batch, loss_adv_sum / num_batch, critic_mark_sum / num_batch,
                   loss_critic_sum / num_batch, attack_success_rate))

            # save generator
            if epoch % 20 == 0:
                netG_file_name = models_path + 'netG_epoch_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)
                netD_file_name = models_path + 'netD_epoch_' + str(epoch) + '.pth'
                torch.save(self.netDisc.state_dict(), netD_file_name)
                netC_file_name = models_path + 'netC_epoch_' + str(epoch) + '.pth'
                torch.save(self.critic.state_dict(), netC_file_name)

class AdvGAN_Attack:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 box_min,
                 box_max):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max

        self.gen_input_nc = image_nc
        self.netG = models.Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = models.Discriminator(image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.001)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, x, labels):
        # optimize D
        for i in range(1):
            perturbation = self.netG(x)

            # add a clipping trick
            adv_images = torch.clamp(perturbation, -0.3, 0.3) + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 0.1
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            # cal adv loss
            logits_model = self.model(adv_images)
            probs_model = F.softmax(logits_model, dim=1)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(real - other, zeros)
            loss_adv = torch.sum(loss_adv)

            # maximize cross_entropy loss
            # loss_adv = -F.mse_loss(logits_model, onehot_labels)
            # loss_adv = - F.cross_entropy(logits_model, labels)

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()

    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):

            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001)
            if epoch == 80:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.00001)
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = \
                    self.train_batch(images, labels)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))

            # save generator
            if epoch%20==0:
                netG_file_name = models_path + 'netG_epoch_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)
