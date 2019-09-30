import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from networks import Generator, Discriminator
from utility import generate_latent


class DCGAN(object):
    def __init__(self, image_size, input_channels, hidden_channels, output_channels, latent_dimension, lr, device):
        self.image_size = image_size
        self.input_channels = input_channels
        self.hidden_chanels = hidden_channels
        self.output_channels = output_channels
        self.latent_dimension = latent_dimension
        self.device = device

        self.discriminator = Discriminator(
            image_size, hidden_channels, input_channels).to(device)
        self.generator = Generator(
            image_size, latent_dimension, hidden_channels, output_channels).to(device)

        self.discriminator.apply(self.weights_init)
        self.generator.apply(self.weights_init)

        self.optimizer_dis = torch.optim.RMSprop(
            self.discriminator.parameters(), lr)
        self.optimizer_gen = torch.optim.RMSprop(
            self.generator.parameters(), lr)

        self.dis_losses = []
        self.gen_losses = []

    def load_model(self, load_name):
        self.discriminator.load_state_dict(torch.load(
            load_name + '_discriminator', map_location='cpu'))
        self.generator.load_state_dict(torch.load(
            load_name + '_generator', map_location='cpu'))

    def discriminate(self, x):
        return self.discriminator(x)

    def generate(self, z):
        return self.generator(z)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    def normalise_pixels(self, images):
        normalised_images = (images - 0.5)/0.5
        return normalised_images

    def denormalise_pixels(self, images):
        denormalised_images = (images * 0.5) + 0.5
        return denormalised_images

    def train(self, data_loader, criterion, num_epochs, log_iter=1, test_iter=1, test_latents=None, save_name=None, load_name=None):
        if load_name is not None:
            self.load_model(load_name)

        for epoch in range(num_epochs):
            if epoch % test_iter == 0 and test_latents is not None:
                images = self.generate(test_latents)
                self.display_images(images)

            for i, batch in enumerate(data_loader):
                real_images, _ = batch
                real_images = self.normalise_pixels(real_images)

                real_images = real_images.to(self.device)

                d_real = self.discriminate(real_images)

                # Uniform noise [-0.2, 0.2]
                real_noise = ((torch.rand(d_real.size()) * 2) - 1)/10
                y_real = (torch.ones(d_real.size()) +
                          real_noise).to(self.device)

                # BCE loss for discriminator on real images
                self.optimizer_dis.zero_grad()

                d_real_err = criterion(d_real, y_real)
                d_real_err.backward()

                # Generate a batch of latents
                latent_zs = generate_latent(
                    self.latent_dimension, len(real_images)).to(self.device)

                # Transform latents into images using the generator
                fake_images = self.generate(latent_zs).to(self.device)

                # Classify fakes with discriminator
                d_fake = self.discriminate(fake_images.detach())

                # Uniform noise [0, 0.3]
                fake_noise = (torch.rand(d_fake.size()) * 3)/10
                y_fake = (torch.zeros(d_fake.size()) +
                          fake_noise).to(self.device)

                # BCE loss for discriminator on fake images
                d_fake_err = criterion(d_fake, y_fake)
                d_fake_err.backward()

                d_err = d_real_err + d_fake_err

                # Gradient descent step for discriminator
                self.optimizer_dis.step()

                ### Generator ###
                self.optimizer_gen.zero_grad()
                d_fake = self.discriminate(fake_images)

                # Uniform noise [-0.2, 0.2]
                real_noise = ((torch.rand(d_real.size()) * 2) - 1)/10
                y_real = (torch.ones(d_real.size()) +
                          real_noise).to(self.device)

                g_err = criterion(d_fake, y_real)
                g_err.backward()

                self.optimizer_gen.step()

            if epoch % log_iter == 0:
                self.dis_losses.append(d_err)
                self.gen_losses.append(g_err)
                print('Epoch %d: Discriminator Loss: %.4f\t Generator Loss: %.4f\t Total Loss: %.4f\t' % (
                    epoch, d_err, g_err, d_err + g_err))

                if save_name is not None:
                    torch.save(self.discriminator.state_dict(),
                               save_name + '_discriminator')
                    torch.save(self.generator.state_dict(),
                               save_name + '_generator')

    def display_images(self, images):
        k = 0

        fig, ax = plt.subplots(1, 5)

        for i in range(5):
            ax[i].imshow(transforms.ToPILImage()(
                (self.denormalise_pixels(images[k].data.cpu().detach()))))
            k += 1

        fig.set_figheight(20)
        fig.set_figwidth(20)
        plt.show()
