import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from networks import Generator, Critic
from utility import generate_latent
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pickle
import numpy as np
from networks import Generator, Critic
from utility import generate_latent

from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class WGAN(object):
    def __init__(self, image_size, input_channels, hidden_channels, output_channels, latent_dimension, lr, device, clamp=0.01, gp_weight=10):
        self.image_size = image_size
        self.input_channels = input_channels
        self.hidden_chanels = hidden_channels
        self.output_channels = output_channels
        self.latent_dimension = latent_dimension
        self.device = device
        self.clamp = clamp
        self.gp_weight = gp_weight

        self.critic = Critic(image_size, hidden_channels,
                             input_channels).to(device)
        self.generator = Generator(
            image_size, latent_dimension, hidden_channels, output_channels).to(device)

        self.critic.apply(self.weights_init)
        self.generator.apply(self.weights_init)

        self.optimizer_critic = torch.optim.RMSprop(
            self.critic.parameters(), lr)
        self.optimizer_gen = torch.optim.RMSprop(
            self.generator.parameters(), lr)

        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(), lr, betas=(0, 0.9))
        self.optimizer_gen = torch.optim.Adam(
            self.generator.parameters(), lr, betas=(0, 0.9))

        self.critic_losses = []
        self.gen_losses = []

        self.losses = []

    def critique(self, x):
        return self.critic(x)

    def generate(self, z):
        return self.generator(z)

    def load_model(self, load_name):
        self.critic.load_state_dict(torch.load(
            load_name + '_critic', map_location='cpu'))
        self.generator.load_state_dict(torch.load(
            load_name + '_generator', map_location='cpu'))

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)

        if self.device != 'cpu':
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + \
            (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.device != 'cpu':
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.critique(interpolated).to(self.device)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device), create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    def normalise_pixels(self, images):
        normalised_images = (images - 0.5)/0.5
        return normalised_images

    def denormalise_pixels(self, images):
        denormalised_images = (images * 0.5) + 0.5
        #denormalised_images = np.clip(denormalised_images, 0, 1)
        return denormalised_images

    def train(self, data_loader, num_epochs, n_iter=5, log_iter=1, test_iter=1, test_latents=None, save_name=None, load_name=None):
        if load_name is not None:
            self.load_model(load_name)

        d_err = 0
        g_err = 0
        for epoch in range(num_epochs):
            if epoch % test_iter == 0 and test_latents is not None:
                images = self.generate(test_latents)
                self.display_images(images)

            if epoch < 0:
                critic_iters = 100

            else:
                critic_iters = n_iter

            j = 1
            for i, batch in enumerate(data_loader):
                real_images, _ = batch
                real_images = self.normalise_pixels(real_images)

                real_images = real_images.to(self.device)
                d_real = self.critique(real_images)

                self.optimizer_critic.zero_grad()
                # Generate a batch of latents
                latent_zs = generate_latent(
                    self.latent_dimension, len(real_images)).to(self.device)

                # Transform latents into images using the generator
                fake_images = self.generate(latent_zs).to(self.device)

                # Classify fakes with discriminator
                d_fake = self.critique(fake_images.detach())

                gradient_penalty = self._gradient_penalty(
                    real_images, fake_images)

                d_err = -(torch.mean(d_real) - torch.mean(d_fake)) + \
                    gradient_penalty
                d_err.backward()

                # Gradient descent step for discriminator
                self.optimizer_critic.step()

#        for p in self.critic.parameters():
#          p.data.clamp_(-self.clamp, self.clamp)

                j += 1

                if j % critic_iters == 0:
                    ### Generator ###
                    self.optimizer_gen.zero_grad()

                    # Generate a batch of latents
                    latent_zs = generate_latent(
                        self.latent_dimension, len(real_images)).to(self.device)
                    # Transform latents into images using the generator
                    fake_images = self.generate(latent_zs).to(self.device)
                    d_fake = self.critique(fake_images)
                    g_err = -torch.mean(d_fake)
                    g_err.backward()

                    self.optimizer_gen.step()

                    j = 1

            if epoch % log_iter == 0:
                self.critic_losses.append(d_err)
                self.gen_losses.append(g_err)
                print('Epoch %d: Wasserstein Loss: %.4f\t Generator Loss: %.4f' % (
                    epoch, -d_err, g_err))

                if save_name is not None:
                    torch.save(self.critic.state_dict(), save_name + '_critic')
                    torch.save(self.generator.state_dict(),
                               save_name + '_generator')

                    with open(save_name + '_critic_losses.pkl', 'wb') as f:
                        pickle.dump(self.critic_losses, f)

                    with open(save_name + '_gen_losses.pkl', 'wb') as f:
                        pickle.dump(self.gen_losses, f)

    def display_images(self, images):
        k = 0

        fig, ax = plt.subplots(1, 5)

        for i in range(5):
            image = self.denormalise_pixels(
                images[k]).cpu().detach().permute(1, 2, 0)

            ax[i].imshow(image)

            ax[i].spines["top"].set_visible(False)
            ax[i].spines["right"].set_visible(False)
            ax[i].spines["bottom"].set_visible(False)
            ax[i].spines["left"].set_visible(False)
            ax[i].set_xticks([])
            ax[i].set_yticks([])

            k += 1

        fig.set_figheight(20)
        fig.set_figwidth(20)

        plt.show()
